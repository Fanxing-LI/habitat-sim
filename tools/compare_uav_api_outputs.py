import json
import sys

import numpy as np


TOL = 1e-5


def load(path):
    with open(path) as f:
        return json.loads(f.readline())


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
        if old[key] != new[key]:
            failures.append(f"{name}.{key}: {old[key]} != {new[key]}")
        summary[f"{name}_{key}"] = old[key]
    if np.array_equal(np.asarray(old["faces"]), np.asarray(new["faces"])):
        summary[f"{name}_faces_equal"] = True
    else:
        summary[f"{name}_faces_equal"] = False
        failures.append(f"{name}.faces differ")
    for key in ["vertices", "min", "max", "sum", "mean"]:
        compare_array(f"{name}.{key}", old[key], new[key], failures, summary)


def compare_record(name, old, new, failures, summary):
    if bool(old["is_out_bound"]) != bool(new["is_out_bound"]):
        failures.append(
            f"{name}.is_out_bound: {old['is_out_bound']} != {new['is_out_bound']}"
        )
    compare_array(f"{name}.hit_pos", old["hit_pos"], new["hit_pos"], failures, summary)


old = load(sys.argv[1])
new = load(sys.argv[2])

failures = []
summary = {}

compare_mesh("scene_mesh", old["scene_mesh"], new["scene_mesh"], failures, summary)
compare_mesh("object_mesh", old["object_mesh"], new["object_mesh"], failures, summary)

old_collision = old["collision"]
new_collision = new["collision"]
if old_collision.keys() != new_collision.keys():
    failures.append("collision point key sets differ")
else:
    for point_name in old_collision:
        if old_collision[point_name].keys() != new_collision[point_name].keys():
            failures.append(f"collision flag key sets differ for {point_name}")
            continue
        for flag_name in old_collision[point_name]:
            compare_record(
                f"collision.{point_name}.{flag_name}",
                old_collision[point_name][flag_name],
                new_collision[point_name][flag_name],
                failures,
                summary,
            )

compare_record(
    "refresh_near_object",
    old["refresh_near_object"],
    new["refresh_near_object"],
    failures,
    summary,
)

print(json.dumps({"summary": summary, "failures": failures}, indent=2, sort_keys=True))
if failures:
    raise SystemExit(1)
