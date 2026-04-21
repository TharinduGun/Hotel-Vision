"""
Builds a pickle file of face embeddings from employee images.

For each employee image, generates multiple augmented versions to simulate
real-world variations (lighting, angle, blur), extracts an embedding from
each, and stores the averaged embedding for more robust recognition.
"""

import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


class EmployeeDatabase:

    def __init__(self, employee_dir="data/employees",
                 output_path="data/employee_embeddings.pkl",
                 use_augmentation=True):
        """
        Args:
            employee_dir: Directory containing employee face images.
            output_path: Where to save the embeddings pickle.
            use_augmentation: If True, generate augmented variants per image
                              and average their embeddings.
        """
        self.employee_dir = employee_dir
        self.output_path = output_path
        self.use_augmentation = use_augmentation
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def _augment(self, img):
        """
        Generate augmented variants of a face image.

        Returns a list of images including the original plus augmented versions.
        Augmentations target real CCTV conditions: lighting, slight rotation,
        blur, noise, and contrast normalisation.
        """
        variants = [img]
        h, w = img.shape[:2]

        # Horizontal flip
        variants.append(cv2.flip(img, 1))

        # Brightness variations
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
        dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
        variants.extend([bright, dark])

        # Contrast variations
        high_contrast = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        low_contrast = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
        variants.extend([high_contrast, low_contrast])

        # Slight rotations (±10 degrees)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            variants.append(rotated)

        # Gaussian blur (simulates low-quality CCTV)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        variants.append(blurred)

        # Gaussian noise (simulates sensor noise)
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        variants.append(noisy)

        # CLAHE — improves contrast in poorly-lit faces
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        clahe_img = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        variants.append(clahe_img)

        return variants

    def _get_embedding(self, img):
        """Extract a single face embedding from an image, or None if no face found."""
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].embedding

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

            if self.use_augmentation:
                variants = self._augment(img)
                embeddings = []

                for variant in variants:
                    emb = self._get_embedding(variant)
                    if emb is not None:
                        embeddings.append(emb)

                if len(embeddings) == 0:
                    print(f"Warning: No face detected in any variant of {filename}, skipping.")
                    continue

                # Average all valid embeddings, then normalise
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                database[name] = avg_embedding
                print(f"  Registered: {name} ({len(embeddings)}/{len(variants)} variants used)")

            else:
                emb = self._get_embedding(img)
                if emb is None:
                    print(f"Warning: No face detected in {filename}, skipping.")
                    continue
                database[name] = emb
                print(f"  Registered: {name}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "wb") as f:
            pickle.dump(database, f)

        print(f"Employee embeddings saved ({len(database)} employees) -> {self.output_path}")