#!/usr/bin/env python3
"""Estimate position from a new WiFi scan using the generated fingerprint library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wifi_positioning import (
    estimate_position,
    load_fingerprints,
    parse_scan_csv,
    parse_scan_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("reconstruction_dataset/wifi_fingerprint_library.csv"))
    parser.add_argument("--scan-json", type=Path, help="JSON array using wifiScanner.dat AP objects.")
    parser.add_argument("--scan-csv", type=Path, help="CSV with mac/dbm or mac/rssi columns.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ap_fields, fingerprints = load_fingerprints(args.library)
    if not fingerprints:
        raise SystemExit(f"No fingerprints loaded from {args.library}")
    if args.scan_json:
        scan_vector, known_fields, unknown_macs = parse_scan_json(args.scan_json, ap_fields)
        scan_source = str(args.scan_json)
    elif args.scan_csv:
        scan_vector, known_fields, unknown_macs = parse_scan_csv(args.scan_csv, ap_fields)
        scan_source = str(args.scan_csv)
    else:
        raise SystemExit("Provide --scan-json or --scan-csv")

    estimate = estimate_position(
        scan_vector=scan_vector,
        ap_fields=ap_fields,
        fingerprints=fingerprints,
        k=args.k,
        active_fields=known_fields,
    )
    result = {
        "scan_source": scan_source,
        "library": str(args.library),
        "known_ap_count": len(known_fields),
        "unknown_ap_count": len(unknown_macs),
        "unknown_macs": unknown_macs[:20],
        **estimate,
    }
    if args.output_json:
        write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
