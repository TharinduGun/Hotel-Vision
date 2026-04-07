"""
Builds a pickle file of face embeddings from employee images.

Each image file in the employee directory should contain one face and be
named with the employee's name (matching keys in employees.json).
"""

import os
import cv2
import pickle
from insightface.app import FaceAnalysis


class EmployeeDatabase:

    def __init__(self, employee_dir="data/employees", output_path="data/employee_embeddings.pkl"):
        self.employee_dir = employee_dir
        self.output_path = output_path
        self.app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0)

    def build(self):
        """Scan employee images, extract embeddings, and save to pickle."""
        if not os.path.isdir(self.employee_dir):
            print(f"Error: Employee image directory not found: {self.employee_dir}")
            return

        database = {}
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for filename in sorted(os.listdir(self.employee_dir)):
            if not filename.lower().endswith(image_extensions):
                continue

            name = os.path.splitext(filename)[0]
            img_path = os.path.join(self.employee_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue

            faces = self.app.get(img)

            if len(faces) == 0:
                print(f"Warning: No face detected in {filename}, skipping.")
                continue

            if len(faces) > 1:
                print(f"Warning: Multiple faces in {filename}, using the first one.")

            database[name] = faces[0].embedding
            print(f"  Registered: {name}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "wb") as f:
            pickle.dump(database, f)

        print(f"Employee embeddings saved ({len(database)} employees) -> {self.output_path}")