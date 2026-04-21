import pickle
import os
import numpy as np

path = "data/employee_embeddings.pkl"
print(f"File size: {os.path.getsize(path)} bytes")

with open(path, "rb") as f:
    db = pickle.load(f)

print(f"Total employees: {len(db)}")
first_name = list(db.keys())[0]
print(f"Embedding shape: {db[first_name].shape}")
print(f"Embedding dtype: {db[first_name].dtype}")
print()

for name in sorted(db.keys()):
    emb = db[name]
    norm = float(np.linalg.norm(emb))
    print(f"  {name:15s}  norm={norm:.4f}  min={emb.min():.4f}  max={emb.max():.4f}")
