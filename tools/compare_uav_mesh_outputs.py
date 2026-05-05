import json
import sys

import numpy as np


def load(path):
    with open(path) as f:
        return json.loads(f.readline())


old = load(sys.argv[1])
new = load(sys.argv[2])

old_vertices = np.asarray(old["vertices"], dtype=np.float64)
new_vertices = np.asarray(new["vertices"], dtype=np.float64)
old_faces = np.asarray(old["faces"], dtype=np.int64)
new_faces = np.asarray(new["faces"], dtype=np.int64)

if old_vertices.shape != new_vertices.shape:
    raise SystemExit(f"vertex shape mismatch: {old_vertices.shape} != {new_vertices.shape}")
if old_faces.shape != new_faces.shape:
    raise SystemExit(f"face shape mismatch: {old_faces.shape} != {new_faces.shape}")

face_equal = bool(np.array_equal(old_faces, new_faces))
vertex_abs = np.abs(old_vertices - new_vertices)

summary = {
    "vertex_count": int(old_vertices.shape[0]),
    "face_index_count": int(old_faces.shape[0]),
    "faces_equal": face_equal,
    "max_abs_vertex_delta": float(vertex_abs.max(initial=0.0)),
    "mean_abs_vertex_delta": float(vertex_abs.mean() if vertex_abs.size else 0.0),
    "old_min": old["min"],
    "new_min": new["min"],
    "old_max": old["max"],
    "new_max": new["max"],
    "old_sum": old["sum"],
    "new_sum": new["sum"],
}
print(json.dumps(summary, indent=2, sort_keys=True))

if not face_equal or summary["max_abs_vertex_delta"] > 1e-5:
    raise SystemExit(1)
