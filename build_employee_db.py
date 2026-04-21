"""
One-off script to generate face embeddings for all employee images.

Usage:
    python build_employee_db.py
    python build_employee_db.py --image-dir data/employees --output data/employee_embeddings.pkl
"""

import argparse
from Identity.employee_database import EmployeeDatabase


def main():
    parser = argparse.ArgumentParser(description="Build employee face embedding database")
    parser.add_argument("--image-dir", type=str, default="data/employees",
                        help="Directory containing employee face images")
    parser.add_argument("--output", type=str, default="data/employee_embeddings.pkl",
                        help="Output path for the embeddings pickle file")
    args = parser.parse_args()

    db = EmployeeDatabase(employee_dir=args.image_dir, output_path=args.output)
    db.build()


if __name__ == "__main__":
    main()