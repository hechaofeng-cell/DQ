"""Run the DTA Imagenette phase-one reliability gate."""

import argparse
from pathlib import Path

from .pipeline import run_phase1
from .review_web import serve_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    phase1 = commands.add_parser("phase1", help="Run or resume the complete phase-one pipeline")
    phase1.add_argument("--config", type=Path, required=True)
    phase1.add_argument("--output", type=Path, required=True)
    phase1.add_argument("--resume", action="store_true")
    phase1.add_argument("--smoke-only", action="store_true")
    review = commands.add_parser("review", help="Serve the local human-review interface")
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "review":
        serve_review(args.output, args.host, args.port)
        return
    try:
        result = run_phase1(args.config, args.output, args.resume, args.smoke_only)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"dta: {exc}\n")
    print(f"DTA phase 1: {result.get('status', result.get('decision'))}")


if __name__ == "__main__":
    main()
