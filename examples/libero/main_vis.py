"""LIBERO eval entry point used for *attention visualization*.

Differences vs. ``main.py``:
  * runs only the **selected** task(s) inside the chosen suite (``--args.task-id``
    for a single task, or ``--args.task-ids`` for a list), instead of looping
    over all tasks.
  * defaults to fewer trials (``--args.num-trials-per-task=3``) since we only
    want a handful of rollouts to look at.
  * everything else — observation dict, replanning, video saving, the
    ``__rollout_*__`` metadata that drives attention-vis folder layout — is
    identical to ``main.py``.

When multiple task IDs are passed they're evaluated sequentially against the
same policy server. The server already partitions attention dumps by task
(``<PI0_ATTN_VIS_DIR>/task<NN>_<name>/ep<NNN>/...``) using the rollout context
metadata sent in each request, and replay videos are filename-disambiguated by
``task_id``, so a single shared output directory is safe.
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Set, Tuple

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
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90."""

    task_id: int = 0
    """Index of the task to run inside the chosen task suite (0-indexed).

    Ignored if ``task_ids`` is non-empty. Kept for backward compatibility with
    callers that pass a single task index.
    """

    task_ids: Tuple[int, ...] = ()
    """Indices of the tasks to run, evaluated sequentially (0-indexed).

    If empty, falls back to ``[task_id]``. Pass on the CLI as e.g.
    ``--args.task-ids 0 3 8``.
    """

    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""

    num_trials_per_task: int = 3
    """Number of rollouts of the selected task. Kept small for visualization."""

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_vis"
    """Path to save replay videos."""

    seed: int = 7
    """Random seed (for reproducibility)."""


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (total tasks: {num_tasks_in_suite})")

    task_ids = list(args.task_ids) if args.task_ids else [args.task_id]
    seen: Set[int] = set()
    deduped: List[int] = []
    for tid in task_ids:
        if tid not in seen:
            seen.add(tid)
            deduped.append(tid)
    task_ids = deduped

    for tid in task_ids:
        if not 0 <= tid < num_tasks_in_suite:
            raise ValueError(
                f"task id {tid} out of range; suite '{args.task_suite_name}' "
                f"has {num_tasks_in_suite} tasks (valid range: 0..{num_tasks_in_suite - 1})."
            )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    logging.info(f"Will evaluate {len(task_ids)} task(s): {task_ids}")

    total_episodes, total_successes = 0, 0
    per_task_stats: List[Tuple[int, int, int]] = []  # (task_id, successes, episodes)

    for task_id in task_ids:
        logging.info("=" * 60)
        logging.info(f"Starting task #{task_id}")
        logging.info("=" * 60)

        task_successes, task_episodes, total_successes, total_episodes = _eval_single_task(
            args=args,
            task_suite=task_suite,
            task_id=task_id,
            max_steps=max_steps,
            client=client,
            total_episodes=total_episodes,
            total_successes=total_successes,
        )
        per_task_stats.append((task_id, task_successes, task_episodes))

    logging.info("=" * 60)
    logging.info("Per-task summary:")
    for tid, succ, eps in per_task_stats:
        rate = (float(succ) / float(eps)) if eps else 0.0
        logging.info(f"  task {tid}: {succ}/{eps} = {rate:.3f}")
    if total_episodes:
        logging.info(
            f"Overall success rate: {float(total_successes) / float(total_episodes):.3f} "
            f"({total_successes}/{total_episodes})"
        )
    logging.info(f"Total episodes: {total_episodes}")


def _eval_single_task(
    *,
    args: Args,
    task_suite,
    task_id: int,
    max_steps: int,
    client,
    total_episodes: int,
    total_successes: int,
) -> Tuple[int, int, int, int]:
    """Evaluate one task; returns (task_successes, task_episodes, total_successes, total_episodes)."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    logging.info(f"Selected task #{task_id}: {task_description}")

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        logging.info(f"\nTask: {task_description}")

        env.reset()
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
                        # Rollout metadata for attention-vis folder layout.
                        # (Stripped server-side by Policy.infer; harmless on
                        # older servers that don't know about these keys.)
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
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        task_episodes += 1
        total_episodes += 1

        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path)
            / f"vis_task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=10,
        )

        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(
            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
        )

    if task_episodes:
        logging.info(
            f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.3f}"
        )

    return task_successes, task_episodes, total_successes, total_episodes


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite. https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55"""
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
