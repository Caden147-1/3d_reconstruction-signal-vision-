#!/usr/bin/env python3
"""Write reconstruction-side ablation commands without running them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def command_line(parts: Sequence[str]) -> str:
    return " ".join(ps_quote(part) if " " in part or "\\" in part or "/" in part else part for part in parts)


def scenario_dirs(base_dir: Path, scenario_name: str) -> Dict[str, str]:
    scenario_root = base_dir / scenario_name
    return {
        "scenario_root": scenario_root.as_posix(),
        "processed": (scenario_root / "processed").as_posix(),
        "reconstruction": (scenario_root / "reconstruction_dataset").as_posix(),
        "workspace": (scenario_root / "colmap_workspace").as_posix(),
    }


def build_main_command(
    scenario: Dict[str, Any],
    include_wifi_positioning: bool,
) -> List[str]:
    dirs = scenario["dirs"]
    command = [
        "python",
        "main.py",
        "--output",
        dirs["processed"],
        "--reconstruction-output",
        dirs["reconstruction"],
        "--colmap-workspace",
        dirs["workspace"],
    ]
    if not include_wifi_positioning:
        command.append("--skip-wifi-positioning")
    command.extend(scenario.get("args", []))
    return command


def build_scenarios(base_dir: Path) -> List[Dict[str, Any]]:
    specs = [
        {
            "name": "pdr_enabled",
            "purpose": "Current reconstruction-side default: PDR-assisted matching and PDR alignment refs.",
            "args": [],
        },
        {
            "name": "visual_only",
            "purpose": "Compare against visual/order constraints without PDR distance fallback or PDR alignment refs.",
            "args": ["--visual-only"],
        },
        {
            "name": "no_video_bridge",
            "purpose": "Measure whether route video bridge frames improve sparse connectivity.",
            "args": ["--skip-video-frames"],
        },
        {
            "name": "blur_threshold_0",
            "purpose": "Keep all readable video frames and test whether current blur filtering is too strict.",
            "args": ["--video-blur-threshold", "0"],
        },
        {
            "name": "blur_threshold_40",
            "purpose": "Moderate video blur filtering.",
            "args": ["--video-blur-threshold", "40"],
        },
        {
            "name": "matcher_exhaustive",
            "purpose": "Compare constrained matching against COLMAP exhaustive matching.",
            "args": ["--colmap-matcher", "exhaustive"],
        },
        {
            "name": "matcher_sequential",
            "purpose": "Compare constrained matching against COLMAP sequential matching.",
            "args": ["--colmap-matcher", "sequential"],
        },
        {
            "name": "pdr_distance_4m",
            "purpose": "Tighter PDR distance window to test over-restriction.",
            "args": ["--constrained-max-pdr-distance-m", "4"],
        },
        {
            "name": "pdr_distance_12m",
            "purpose": "Looser PDR distance window to test drift tolerance and false-pair risk.",
            "args": ["--constrained-max-pdr-distance-m", "12"],
        },
        {
            "name": "slot_window_1",
            "purpose": "Tighter angular slot window inside and across WiFi checkpoints.",
            "args": ["--constrained-slot-window", "1"],
        },
    ]
    scenarios: List[Dict[str, Any]] = []
    for spec in specs:
        scenario = dict(spec)
        scenario["dirs"] = scenario_dirs(base_dir, scenario["name"])
        scenarios.append(scenario)
    return scenarios


def render_powershell(scenarios: Iterable[Dict[str, Any]], include_wifi_positioning: bool) -> str:
    lines = [
        "# Reconstruction ablation commands",
        "# These commands prepare isolated workspaces. Sparse/dense runs are listed after each prepare step.",
        "",
    ]
    for scenario in scenarios:
        workspace = scenario["dirs"]["workspace"]
        main_command = command_line(build_main_command(scenario, include_wifi_positioning))
        report_command = command_line(
            [
                "python",
                "reconstruction_quality.py",
                "--processed",
                scenario["dirs"]["processed"],
                "--reconstruction",
                scenario["dirs"]["reconstruction"],
                "--workspace",
                workspace,
                "--json-output",
                f"{workspace}/quality_report.json",
                "--markdown-output",
                f"{workspace}/QUALITY_REPORT.md",
            ]
        )
        lines.extend(
            [
                f"# {scenario['name']}: {scenario['purpose']}",
                main_command,
                command_line(["python", "run_colmap.py", "reset", "--workspace", workspace]),
                command_line(["python", "run_colmap.py", "sparse", "--workspace", workspace]),
                command_line(["python", "run_colmap.py", "dense", "--workspace", workspace]),
                report_command,
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("ablation_runs"))
    parser.add_argument("--json-output", type=Path, default=Path("ablation_runs") / "ablation_plan.json")
    parser.add_argument("--powershell-output", type=Path, default=Path("ablation_runs") / "run_ablation_commands.ps1")
    parser.add_argument(
        "--include-wifi-positioning",
        action="store_true",
        help="Do not add --skip-wifi-positioning to generated preprocessing commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = build_scenarios(args.base_dir)
    for scenario in scenarios:
        scenario["prepare_command"] = build_main_command(scenario, args.include_wifi_positioning)
        scenario["colmap_commands"] = [
            ["python", "run_colmap.py", "reset", "--workspace", scenario["dirs"]["workspace"]],
            ["python", "run_colmap.py", "sparse", "--workspace", scenario["dirs"]["workspace"]],
            ["python", "run_colmap.py", "dense", "--workspace", scenario["dirs"]["workspace"]],
        ]
        scenario["quality_report_command"] = [
            "python",
            "reconstruction_quality.py",
            "--processed",
            scenario["dirs"]["processed"],
            "--reconstruction",
            scenario["dirs"]["reconstruction"],
            "--workspace",
            scenario["dirs"]["workspace"],
        ]

    plan = {
        "description": "Reconstruction-side ablation plan. Generated commands do not run automatically.",
        "include_wifi_positioning": args.include_wifi_positioning,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    write_json(args.json_output, plan)
    write_text(args.powershell_output, render_powershell(scenarios, args.include_wifi_positioning))
    print(f"[done] wrote {args.json_output}")
    print(f"[done] wrote {args.powershell_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
