"""
Entry point for the employee tracking pipeline.
Reads camera config and launches one processing thread per camera.
"""

import argparse
from pipeline.pipeline_manager import PipelineManager


def main():
    parser = argparse.ArgumentParser(description="Employee Tracking System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cameras.json",
        help="Path to cameras config JSON",
    )
    args = parser.parse_args()

    manager = PipelineManager(config_path=args.config)
    manager.start()


if __name__ == "__main__":
    main()