#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from batch_rep_boundary_report import (
    _iter_session_dirs,
    _json_default,
    _session_metadata,
    analyze_with_finalrep,
    analyze_with_plot_multi,
)


FIXED_PERSON_REP_COUNTS = {
    "Abhinav": 15,
    "Salvador": 15,
}


TRUTH_PATTERNS_SETS_X_REPS = [
    re.compile(r"(?i)(?<!\d)(\d+)x(\d+)(?!\d)"),
]

TRUTH_PATTERNS_REPS_ONLY = [
    re.compile(r"(?i)(?<!\d)(\d+)x(?:_|-|$)"),
]


def infer_ground_truth_reps(session_name: str) -> Optional[int]:
    name = session_name.strip()

    for pattern in TRUTH_PATTERNS_SETS_X_REPS:
        match = pattern.search(name)
        if not match:
            continue
        sets = int(match.group(1))
        reps = int(match.group(2))
        if 1 <= sets <= 10 and 1 <= reps <= 40:
            return int(sets * reps)

    for pattern in TRUTH_PATTERNS_REPS_ONLY:
        match = pattern.search(name)
        if not match:
            continue
        reps = int(match.group(1))
        if 1 <= reps <= 40:
            return reps

    return None


def infer_ground_truth_for_session(meta: Dict[str, Any]) -> tuple[Optional[int], str]:
    person = str(meta.get("person", "")).strip()
    if person in FIXED_PERSON_REP_COUNTS:
        return FIXED_PERSON_REP_COUNTS[person], "manual_person_rule"

    inferred = infer_ground_truth_reps(str(meta.get("session", "")))
    if inferred is not None:
        return inferred, "inferred_ground_truth"

    return None, ""


def _winner(finalrep_error: Optional[int], plot_multi_error: Optional[int]) -> str:
    if finalrep_error is None or plot_multi_error is None:
        return ""
    if finalrep_error < plot_multi_error:
        return "FINALREP"
    if plot_multi_error < finalrep_error:
        return "plot_multi_accel_updated"
    return "tie"


def _metrics(rows: Sequence[Dict[str, Any]], method_key: str, truth_key: str = "ground_truth_reps") -> Dict[str, Any]:
    scored = [r for r in rows if r.get(truth_key) is not None and r.get(method_key) is not None]
    if not scored:
        return {
            "sessions_scored": 0,
            "exact_match_count": 0,
            "exact_match_rate": None,
            "mean_absolute_error": None,
            "median_absolute_error": None,
        }

    errors = [abs(int(r[method_key]) - int(r[truth_key])) for r in scored]
    exact = sum(1 for e in errors if e == 0)
    errors_sorted = sorted(errors)
    mid = len(errors_sorted) // 2
    if len(errors_sorted) % 2 == 0:
        median = 0.5 * (errors_sorted[mid - 1] + errors_sorted[mid])
    else:
        median = errors_sorted[mid]

    return {
        "sessions_scored": len(scored),
        "exact_match_count": exact,
        "exact_match_rate": exact / len(scored),
        "mean_absolute_error": sum(errors) / len(scored),
        "median_absolute_error": median,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FINALREP.py and plot_multi_accel_updated.py on the same session folders.")
    parser.add_argument("root", type=str, help="Dataset root directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for comparison reports")
    parser.add_argument("--template-reps", type=int, default=5, help="Template rep count for plot_multi_accel_updated sessions")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "rep_analysis_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for session_dir in _iter_session_dirs(root):
        if session_dir.name == "rep_analysis_reports":
            continue
        meta = _session_metadata(root, session_dir)
        gt, truth_source = infer_ground_truth_for_session(meta)
        print(f"Comparing {meta['relative_path']} ...")

        finalrep_result: Optional[Dict[str, Any]] = None
        plot_result: Optional[Dict[str, Any]] = None
        finalrep_error_text = ""
        plot_error_text = ""

        try:
            finalrep_result = analyze_with_finalrep(session_dir)
        except Exception as exc:
            finalrep_error_text = str(exc)

        try:
            plot_result = analyze_with_plot_multi(
                session_dir,
                template_reps=args.template_reps,
                expected_reps=gt,
                exercise_name=meta["exercise"],
            )
        except Exception as exc:
            plot_error_text = str(exc)

        if finalrep_result is None and plot_result is None:
            errors.append({**meta, "error": "Both methods failed", "finalrep_error": finalrep_error_text, "plot_multi_error": plot_error_text})
            print("  -> both methods failed")
            continue

        finalrep_reps = finalrep_result["estimated_total_reps"] if finalrep_result is not None else None
        plot_reps = plot_result["estimated_total_reps"] if plot_result is not None else None
        finalrep_abs_error = abs(finalrep_reps - gt) if (gt is not None and finalrep_reps is not None) else None
        plot_abs_error = abs(plot_reps - gt) if (gt is not None and plot_reps is not None) else None
        reference_reps = gt if gt is not None else plot_reps
        reference_source = truth_source if gt is not None else ("plot_multi_baseline" if plot_reps is not None else "")
        finalrep_reference_abs_error = abs(finalrep_reps - reference_reps) if (reference_reps is not None and finalrep_reps is not None) else None
        plot_reference_abs_error = abs(plot_reps - reference_reps) if (reference_reps is not None and plot_reps is not None) else None

        row = {
            **meta,
            "ground_truth_reps": gt,
            "reference_reps": reference_reps,
            "reference_source": reference_source,
            "finalrep_reps": finalrep_reps,
            "plot_multi_reps": plot_reps,
            "finalrep_abs_error": finalrep_abs_error,
            "plot_multi_abs_error": plot_abs_error,
            "winner": _winner(finalrep_abs_error, plot_abs_error),
            "finalrep_reference_abs_error": finalrep_reference_abs_error,
            "plot_multi_reference_abs_error": plot_reference_abs_error,
            "reference_winner": _winner(finalrep_reference_abs_error, plot_reference_abs_error),
            "finalrep_analyzer": finalrep_result["analyzer"] if finalrep_result is not None else "",
            "plot_multi_analyzer": plot_result["analyzer"] if plot_result is not None else "",
            "finalrep_error_text": finalrep_error_text,
            "plot_multi_error_text": plot_error_text,
        }
        rows.append(row)
        print(f"  -> FINALREP={finalrep_reps} | plot_multi={plot_reps} | truth={gt}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "sessions_compared": len(rows),
        "sessions_with_ground_truth": sum(1 for r in rows if r.get("ground_truth_reps") is not None),
        "sessions_with_reference": sum(1 for r in rows if r.get("reference_reps") is not None),
        "finalrep": _metrics(rows, "finalrep_reps"),
        "plot_multi_accel_updated": _metrics(rows, "plot_multi_reps"),
        "reference_inclusive": {
            "finalrep": _metrics(rows, "finalrep_reps", truth_key="reference_reps"),
            "plot_multi_accel_updated": _metrics(rows, "plot_multi_reps", truth_key="reference_reps"),
            "head_to_head": {
                "finalrep_wins": sum(1 for r in rows if r.get("reference_winner") == "FINALREP"),
                "plot_multi_wins": sum(1 for r in rows if r.get("reference_winner") == "plot_multi_accel_updated"),
                "ties": sum(1 for r in rows if r.get("reference_winner") == "tie"),
            },
        },
        "head_to_head": {
            "finalrep_wins": sum(1 for r in rows if r.get("winner") == "FINALREP"),
            "plot_multi_wins": sum(1 for r in rows if r.get("winner") == "plot_multi_accel_updated"),
            "ties": sum(1 for r in rows if r.get("winner") == "tie"),
        },
        "errors": errors,
    }

    comparison_json = out_dir / "method_comparison.json"
    comparison_csv = out_dir / "method_comparison.csv"

    comparison_json.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2, default=_json_default))
    _write_csv(comparison_csv, rows)

    print("")
    print(f"Wrote comparison JSON: {comparison_json}")
    print(f"Wrote comparison CSV: {comparison_csv}")
    print(f"Sessions compared: {len(rows)}")
    print(f"Sessions with inferable ground truth: {metrics['sessions_with_ground_truth']}")
    print(f"Sessions with reference count: {metrics['sessions_with_reference']}")
    print(f"FINALREP MAE: {metrics['finalrep']['mean_absolute_error']}")
    print(f"plot_multi_accel_updated MAE: {metrics['plot_multi_accel_updated']['mean_absolute_error']}")


if __name__ == "__main__":
    main()
