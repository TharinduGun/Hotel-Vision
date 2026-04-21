"""
Matches a face embedding against the stored employee embedding database.

Returns the employee ID from employees.json if the closest match is
within the distance threshold, otherwise returns 'unknown'.
"""

import os
import json
import pickle
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FaceRecognizer:

    def __init__(self, embeddings_path=None,
                 employees_config=None, threshold=0.9):
        if embeddings_path is None:
            embeddings_path = os.path.join(_PROJECT_ROOT, "data", "employee_embeddings.pkl")
        if employees_config is None:
            employees_config = os.path.join(_PROJECT_ROOT, "configs", "employees.json")
        """
        Args:
            embeddings_path: Path to the pickled embeddings dict {name: embedding}.
            employees_config: Path to the name -> employee_id JSON mapping.
            threshold: Maximum L2 distance to accept a match.
        """
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}. "
                "Run build_employee_db.py first."
            )

        with open(embeddings_path, "rb") as f:
            self.database = pickle.load(f)

        with open(employees_config) as f:
            self.employee_map = json.load(f)

        self.threshold = threshold

    def recognize(self, embedding):
        """
        Find the closest employee match for a given face embedding.

        Args:
            embedding: 512-d numpy array from InsightFace.

        Returns:
            Employee ID string (e.g. 'E001') or 'unknown'.
        """
        if len(self.database) == 0:
            return "unknown"

        best_match = None
        best_distance = float("inf")

        for name, db_embedding in self.database.items():
            dist = np.linalg.norm(embedding - db_embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = name

        if best_distance < self.threshold and best_match is not None:
            return self.employee_map.get(best_match, best_match)

        return "unknown"