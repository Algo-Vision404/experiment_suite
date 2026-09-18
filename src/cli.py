"""Command-line interface for EXSS."""
from __future__ import annotations

import argparse
import json
from .ingestion import DataIngestor
from .quality import DataHealthScout
from .monitoring import DriftMonitor

def main() -> None:
    parser = argparse.ArgumentParser(prog="exss", description="ML Experimental Standardization Suite")
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile", help="inspect dataset health")
    profile.add_argument("path")
    drift = sub.add_parser("drift", help="compare numeric distributions")
    drift.add_argument("reference")
    drift.add_argument("current")
    drift.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "profile":
        df = DataIngestor.load(args.path)
        print(json.dumps({
            "inspection": DataIngestor.inspect(df),
            "health": DataHealthScout.calculate_health_score(df),
        }, indent=2, default=str))
    elif args.command == "drift":
        ref, cur = DataIngestor.load(args.reference), DataIngestor.load(args.current)
        print(json.dumps(DriftMonitor.calculate_drift(ref, cur, args.threshold), indent=2, default=str))
