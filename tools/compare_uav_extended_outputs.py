import json
import sys

import numpy as np


TOL = 1e-5


def load(path):
    with open(path) as f:
        return json.loads(f.readline())


def compare_scalar(name, old, new, failures):
    if old != new:
        failures.append(f"{name}: {old!r} != {new!r}")


def compare_array(name, old, new, failures, summary):
    old_arr = np.asarray(old, dtype=np.float64)
    new_arr = np.asarray(new, dtype=np.float64)
    if old_arr.shape != new_arr.shape:
        failures.append(f"{name}: shape mismatch {old_arr.shape} != {new_arr.shape}")
        return
    delta = np.abs(old_arr - new_arr)
    max_delta = float(delta.max(initial=0.0))
    summary[f"{name}_max_abs_delta"] = max_delta
    if max_delta > TOL:
        failures.append(f"{name}: max abs delta {max_delta} > {TOL}")


def compare_mesh(name, old, new, failures, summary):
    for key in ["vertex_count", "face_index_count"]:
        compare_scalar(f"{name}.{key}", old[key], new[key], failures)
        summary[f"{name}_{key}"] = old[key]
    for key in ["vertices_sha256", "faces_sha256"]:
        compare_scalar(f"{name}.{key}", old[key], new[key], failures)
        summary[f"{name}_{key}"] = old[key]
    for key in ["min", "max", "sum", "mean"]:
        compare_array(f"{name}.{key}", old[key], new[key], failures, summary)
    if "vertices" in old or "vertices" in new:
        compare_array(f"{name}.vertices", old["vertices"], new["vertices"], failures, summary)
    if "faces" in old or "faces" in new:
        old_faces = np.asarray(old["faces"])
        new_faces = np.asarray(new["faces"])
        faces_equal = bool(np.array_equal(old_faces, new_faces))
        summary[f"{name}_faces_equal"] = faces_equal
        if not faces_equal:
            failures.append(f"{name}.faces differ")


def compare_record(name, old, new, failures, summary):
    compare_scalar(f"{name}.is_out_bound", bool(old["is_out_bound"]), bool(new["is_out_bound"]), failures)
    compare_array(f"{name}.hit_pos", old["hit_pos"], new["hit_pos"], failures, summary)


def compare_tree(name, old, new, failures, summary):
    if isinstance(old, dict):
        if set(old) != set(new):
            failures.append(f"{name}: key mismatch {sorted(old)} != {sorted(new)}")
            return
        if set(old) == {"hit_pos", "is_out_bound"}:
            compare_record(name, old, new, failures, summary)
            return
        for key in sorted(old):
            compare_tree(f"{name}.{key}", old[key], new[key], failures, summary)
    elif isinstance(old, list):
        if len(old) != len(new):
            failures.append(f"{name}: length mismatch {len(old)} != {len(new)}")
            return
        for ix, (old_item, new_item) in enumerate(zip(old, new)):
            compare_tree(f"{name}[{ix}]", old_item, new_item, failures, summary)
    elif isinstance(old, (int, float)):
        compare_array(name, old, new, failures, summary)
    else:
        compare_scalar(name, old, new, failures)


old = load(sys.argv[1])
new = load(sys.argv[2])

failures = []
summary = {}

for key in ["scene", "scene_dataset", "enable_physics", "create_renderer", "objects"]:
    compare_tree(key, old[key], new[key], failures, summary)

compare_mesh("scene_mesh", old["scene_mesh"], new["scene_mesh"], failures, summary)
compare_mesh("initial_object_mesh", old["initial_object_mesh"], new["initial_object_mesh"], failures, summary)
compare_tree("collision", old["collision"], new["collision"], failures, summary)
compare_tree("moved_points", old["moved_points"], new["moved_points"], failures, summary)
compare_record("refresh_query", old["refresh_query"], new["refresh_query"], failures, summary)

old_t = old.get("timings", {})
new_t = new.get("timings", {})
summary["timings_seconds_old"] = old_t
summary["timings_seconds_new"] = new_t
summary["timing_ratio_new_over_old"] = {
    key: new_t[key] / old_t[key]
    for key in old_t.keys() & new_t.keys()
    if old_t[key] > 0
}

print(json.dumps({"summary": summary, "failures": failures}, indent=2, sort_keys=True))
if failures:
    raise SystemExit(1)
