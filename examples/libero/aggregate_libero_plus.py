"""Aggregate per-shard LIBERO-Plus results into a single summary.

Each shard is produced by `main_plus.py` at:
    <results_dir>/<suite>/<start>_<end>.json

with the per-category schema:
    {
      "<category>": {"total_count": int, "success_count": int},
      ...
    }

This script merges all shards across all suites into one `overall_results.json`
that contains:
  - per-suite breakdown by category, plus a per-suite "overall"
  - per-category aggregation across all suites (both micro- and macro-average)
  - a top-level "overall" aggregating every suite

It mirrors `starvla-LIBERO-plus/eval_files/parallel_eval/aggregate_results.py`
in spirit, but is suite-aware and explicit about its inputs (no env vars).

Per-category aggregation explained:
  - `success_rate` is the *micro* average: sum of success_count divided by sum
    of total_count across all suites for that category. Sample-weighted.
  - `macro_success_rate` is the *macro* average: arithmetic mean of the
    per-suite `success_rate` for that category. Each suite weighted equally.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _safe_rate(succ: int, total: int) -> float:
    return float(succ) / float(total) if total > 0 else 0.0


# Display order for per_category aggregation in the output json and stdout.
# Mirrors the column order used in the LIBERO-Plus paper / report tables:
#   Camera | Robot | Language | Light | Background | Noise | Layout
# Categories not in this list (e.g. future additions) are appended in
# alphabetical order so they remain present without breaking the layout.
PER_CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def _ordered_categories(cats) -> list:
    seen = set()
    ordered = []
    for c in PER_CATEGORY_ORDER:
        if c in cats:
            ordered.append(c)
            seen.add(c)
    for c in sorted(cats):
        if c not in seen:
            ordered.append(c)
    return ordered


def aggregate(results_dir: str, suites: list[str]) -> dict:
    out: dict = {
        "overall": {"total_count": 0, "success_count": 0},
        "per_category": {},
        "per_suite": {},
    }

    for suite in suites:
        suite_dir = os.path.join(results_dir, suite)
        suite_agg: dict[str, dict] = {"overall": {"total_count": 0, "success_count": 0}}
        json_files = sorted(glob.glob(os.path.join(suite_dir, "*.json")))
        if not json_files:
            print(f"[warn] no shard files under {suite_dir}", file=sys.stderr)

        for path in json_files:
            with open(path) as f:
                shard = json.load(f)
            for cat, vals in shard.items():
                if cat == "overall":
                    continue  # ignore stray totals if a shard ever produced them
                tot = int(vals.get("total_count", 0))
                suc = int(vals.get("success_count", 0))
                bucket = suite_agg.setdefault(cat, {"total_count": 0, "success_count": 0})
                bucket["total_count"] += tot
                bucket["success_count"] += suc
                suite_agg["overall"]["total_count"] += tot
                suite_agg["overall"]["success_count"] += suc

        for cat, vals in suite_agg.items():
            vals["success_rate"] = _safe_rate(vals["success_count"], vals["total_count"])

        out["per_suite"][suite] = suite_agg
        out["overall"]["total_count"] += suite_agg["overall"]["total_count"]
        out["overall"]["success_count"] += suite_agg["overall"]["success_count"]

    out["overall"]["success_rate"] = _safe_rate(
        out["overall"]["success_count"], out["overall"]["total_count"]
    )

    # Build per-category aggregation across suites.
    # We reuse the already-aggregated per-suite numbers so a category that's
    # absent from some suite (total_count=0 there) is naturally skipped from
    # the macro mean to avoid biasing it toward zero.
    cat_to_per_suite: dict[str, dict[str, dict]] = {}
    for suite, suite_agg in out["per_suite"].items():
        for cat, vals in suite_agg.items():
            if cat == "overall":
                continue
            cat_to_per_suite.setdefault(cat, {})[suite] = vals

    for cat in _ordered_categories(cat_to_per_suite.keys()):
        suite_entries = cat_to_per_suite[cat]
        micro_total = sum(v["total_count"] for v in suite_entries.values())
        micro_succ = sum(v["success_count"] for v in suite_entries.values())
        # Macro average only over suites that actually have samples for this category.
        macro_rates = [v["success_rate"] for v in suite_entries.values() if v["total_count"] > 0]
        out["per_category"][cat] = {
            "total_count": micro_total,
            "success_count": micro_succ,
            "success_rate": _safe_rate(micro_succ, micro_total),
            "macro_success_rate": (sum(macro_rates) / len(macro_rates)) if macro_rates else 0.0,
            "num_suites": len(macro_rates),
            "per_suite_success_rate": {
                suite: v["success_rate"] for suite, v in suite_entries.items()
            },
        }

    return out


def _print_summary(agg: dict) -> None:
    print("=" * 78)
    print("LIBERO-Plus aggregated results")
    print("=" * 78)
    for suite, suite_agg in agg["per_suite"].items():
        s = suite_agg["overall"]
        print(f"[{suite}] {s['success_count']}/{s['total_count']} = {s['success_rate']:.4f}")
        for cat, vals in suite_agg.items():
            if cat == "overall":
                continue
            print(f"    - {cat:24s} {vals['success_count']:>5d}/{vals['total_count']:<5d} = {vals['success_rate']:.4f}")

    if agg.get("per_category"):
        print("-" * 78)
        print("Per-category averages across suites:")
        print(f"  {'category':<24} {'micro':>9} {'macro':>9}  per-suite rates")
        for cat, vals in agg["per_category"].items():
            ps = vals.get("per_suite_success_rate", {})
            ps_str = ", ".join(f"{s.replace('libero_','')}={r:.3f}" for s, r in ps.items())
            print(
                f"  {cat:<24} {vals['success_rate']:>9.4f} {vals['macro_success_rate']:>9.4f}  {ps_str}"
            )

    o = agg["overall"]
    print("-" * 78)
    print(f"OVERALL : {o['success_count']}/{o['total_count']} = {o['success_rate']:.4f}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LIBERO-Plus per-shard results")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing <suite>/<start>_<end>.json shard files",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="Suites to aggregate (default: all four)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output json path (default: <results_dir>/overall_results.json)",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(args.results_dir, "overall_results.json")
    agg = aggregate(args.results_dir, args.suites)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    _print_summary(agg)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
