"""LIBERO-Plus evaluation entrypoint.

This is a sibling of `main.py`, dedicated to the LIBERO-Plus benchmark. It
adds two capabilities on top of the original LIBERO `main.py`:

1. Task-range slicing via `--start_idx` / `--end_idx`, so that a single suite
   can be split into multiple shards and run in parallel across many GPUs
   (see `run_libero_plus_eval.sh`).
2. Per-shard result dumping in the LIBERO-Plus format: aggregated success
   counts grouped by the `category` field from
   `libero/libero/benchmark/task_classification.json`, written to
   `<output_dir>/results/<suite>/<start>_<end>.json`. These shards are merged
   afterwards by `aggregate_libero_plus.py`.

The standard LIBERO `main.py` is intentionally left untouched.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task (LIBERO-Plus default = 1)

    #################################################################################################################
    # Sharding (for multi-GPU parallel evaluation)
    #################################################################################################################
    # Half-open range [start_idx, end_idx) of task ids to run. -1 means "all tasks".
    start_idx: int = -1
    end_idx: int = -1

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"  # Path to save videos
    # Where to dump per-shard json results aggregated by LIBERO-Plus `category`.
    # Final layout: <results_out_dir>/<task_suite_name>/<start>_<end>.json
    results_out_dir: str = "data/libero_plus/results"

    seed: int = 7  # Random Seed (for reproducibility)


def _load_task_id_to_category(task_suite_name: str) -> dict[int, tuple[str, str]]:
    """Load `task_id -> (category, name)` mapping from LIBERO-Plus.

    `LIBERO_PLUS_DIR` (or `LIBERO_HOME`) must point to the LIBERO-Plus repo
    root (the directory that contains `libero/libero/benchmark/...`).

    Note: LIBERO-Plus uses 1-indexed task ids in the json, while LIBERO's
    Python API uses 0-indexed `task_id`. We store the mapping under the
    0-indexed key (i.e. json `id` minus 1).
    """
    libero_root = os.environ.get("LIBERO_PLUS_DIR") or os.environ.get("LIBERO_HOME")
    if not libero_root:
        raise RuntimeError(
            "LIBERO-Plus root not found. Please set LIBERO_PLUS_DIR (preferred) "
            "or LIBERO_HOME to the LIBERO-Plus repo root so that "
            "libero/libero/benchmark/task_classification.json can be located."
        )
    cls_path = pathlib.Path(libero_root) / "libero" / "libero" / "benchmark" / "task_classification.json"
    with open(cls_path) as f:
        all_mapping = json.load(f)
    if task_suite_name not in all_mapping:
        raise KeyError(f"task_suite_name={task_suite_name!r} not found in {cls_path}")
    id2cat: dict[int, tuple[str, str]] = {}
    for item in all_mapping[task_suite_name]:
        id2cat[int(item["id"]) - 1] = (item["category"], item["name"])
    return id2cat


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (n_tasks={num_tasks_in_suite})")

    # Resolve shard range.
    if args.start_idx < 0:
        args.start_idx = 0
    if args.end_idx < 0:
        args.end_idx = num_tasks_in_suite
    args.start_idx = max(0, args.start_idx)
    args.end_idx = min(num_tasks_in_suite, args.end_idx)
    if args.start_idx >= args.end_idx:
        logging.warning(
            f"Empty shard for {args.task_suite_name}: [{args.start_idx}, {args.end_idx}). Nothing to do."
        )
        return
    logging.info(f"Processing tasks [{args.start_idx}, {args.end_idx}) of {num_tasks_in_suite}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_out_dir) / args.task_suite_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    id2category = _load_task_id_to_category(args.task_suite_name)
    # Initialize per-category counters for every category in this suite, so that
    # an empty shard still emits a well-formed json with all keys present.
    disturb_res: dict[str, dict[str, int]] = {}
    for cat, _ in id2category.values():
        disturb_res.setdefault(cat, {"total_count": 0, "success_count": 0})

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(args.start_idx, args.end_idx)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        if task_id not in id2category:
            # Should not happen if task_classification.json matches the suite,
            # but degrade gracefully so a missing entry doesn't crash the shard.
            logging.warning(f"task_id={task_id} not in task_classification.json; using 'unknown' category")
            id2category[task_id] = ("unknown", f"task_{task_id}")
        category, _task_name = id2category[task_id]
        disturb_res.setdefault(category, {"total_count": 0, "success_count": 0})

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            try:
                env.reset()
            except Exception as e:
                logging.error(f"env.reset() failed, skipping episode: {e}")
                task_episodes += 1
                total_episodes += 1
                disturb_res[category]["total_count"] += 1
                continue
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "__rollout_task_id__": int(task_id),
                            "__rollout_episode_id__": int(episode_idx),
                            "__rollout_step__": int(t - args.num_steps_wait),
                            "__rollout_task_name__": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        disturb_res[category]["success_count"] += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            disturb_res[category]["total_count"] += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_name = f"rollout_{task_segment}_{suffix}.mp4"
            if len(video_name.encode("utf-8")) > 250:
                name_hash = hashlib.md5(task_segment.encode()).hexdigest()[:12]
                max_seg_len = 250 - len(f"rollout__{name_hash}_{suffix}.mp4")
                task_segment = task_segment[:max_seg_len]
                video_name = f"rollout_{task_segment}_{name_hash}_{suffix}.mp4"
            try:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / video_name,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            except Exception as e:
                logging.error(f"Failed to save video {video_name}: {e}")

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        if task_episodes > 0:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        if total_episodes > 0:
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Dump per-shard results in the same per-category schema used by LIBERO-Plus,
    # so that aggregate_libero_plus.py can merge them across shards.
    shard_path = results_dir / f"{args.start_idx}_{args.end_idx}.json"
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(disturb_res, f, indent=2)
    logging.info(f"Wrote shard results to {shard_path}")

    if total_episodes > 0:
        logging.info(f"[shard {args.start_idx}-{args.end_idx}] Total success rate: {total_successes / total_episodes:.4f}")
    logging.info(f"[shard {args.start_idx}-{args.end_idx}] Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
