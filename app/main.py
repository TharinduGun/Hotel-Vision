"""
Hotel Vision — Modular Analytics Entry Point
=============================================
Usage:
    cd d:/Work/jwinfotech/Videoanalystics/video-analytics
    python -m app.main
    python -m app.main --config path/to/custom_config.yaml
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.orchestrator.engine import Engine


def main():
    parser = argparse.ArgumentParser(
        description="Hotel Vision — Modular Video Analytics Engine"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to system_config.yaml (uses default if not specified)",
    )
    args = parser.parse_args()

    engine = Engine(config_path=args.config)
    engine.run()


if __name__ == "__main__":
    main()
