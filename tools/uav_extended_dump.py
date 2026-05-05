import argparse
import hashlib
import json
import time

import numpy as np

import habitat_sim


def as_list(value):
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def mesh_summary(mesh, include_full_mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    summary = {
        "vertex_count": int(vertices.shape[0]),
        "face_index_count": int(faces.shape[0]),
        "vertices_sha256": hashlib.sha256(
            np.ascontiguousarray(vertices).view(np.uint8)
        ).hexdigest(),
        "faces_sha256": hashlib.sha256(
            np.ascontiguousarray(faces).view(np.uint8)
        ).hexdigest(),
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
        "sum": vertices.sum(axis=0).tolist(),
        "mean": vertices.mean(axis=0).tolist(),
    }
    if include_full_mesh:
        summary["vertices"] = vertices.tolist()
        summary["faces"] = faces.tolist()
    return summary


def make_sim(args):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = args.scene
    if args.scene_dataset:
        cfg.scene_dataset_config_file = args.scene_dataset
    cfg.enable_physics = args.enable_physics
    cfg.create_renderer = args.create_renderer
    cfg.use_semantic_textures = False
    cfg.load_semantic_mesh = False
    cfg.force_separate_semantic_scene_graph = False
    cfg.enable_gfx_replay_save = True
    cfg.leave_context_with_background_renderer = True
    cfg.random_seed = 7

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    sim.seed(7)
    return sim


def quat_from_yaw(yaw_degrees):
    return habitat_sim.utils.common.quat_to_magnum(
        habitat_sim.utils.common.quat_from_angle_axis(
            np.deg2rad(yaw_degrees), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
    )


def add_object(sim, object_handle, spec):
    obj_mgr = sim.get_rigid_object_manager()
    template_mgr = sim.get_object_template_manager()
    template_id = template_mgr.load_configs(object_handle)[0]
    obj = obj_mgr.add_object_by_template_id(template_id)
    obj.motion_type = getattr(habitat_sim.physics.MotionType, spec["motion_type"])
    obj.translation = np.array(spec["translation"], dtype=np.float32)
    obj.rotation = quat_from_yaw(spec["yaw_degrees"])
    return obj


def collision_record_to_dict(record):
    return {
        "hit_pos": as_list(record.hit_pos),
        "is_out_bound": bool(record.is_out_bound),
    }


def query_collision(sim, point, radius, object_collidable, scene_collidable, shape):
    arr = np.array(point, dtype=np.float32)
    if shape == "column":
        arr = arr.reshape(3, 1)
    return collision_record_to_dict(
        sim.get_closest_collision_point(
            arr,
            radius,
            object_collidable,
            scene_collidable,
        )
    )


def timed(label, timings, fn):
    start = time.perf_counter()
    result = fn()
    timings[label] = time.perf_counter() - start
    return result


def run_case(args):
    sim = make_sim(args)
    try:
        dynamic_specs = [
            {
                "handle": args.object[0],
                "translation": [2.0, 1.0, 0.0],
                "yaw_degrees": 0.0,
                "motion_type": "DYNAMIC",
            },
            {
                "handle": args.object[min(1, len(args.object) - 1)],
                "translation": [-1.25, 1.4, 1.75],
                "yaw_degrees": 43.0,
                "motion_type": "DYNAMIC",
            },
        ]
        static_spec = {
            "handle": args.object[0],
            "translation": [0.0, 1.0, 3.0],
            "yaw_degrees": 17.0,
            "motion_type": "STATIC",
        }

        dynamic_objects = [
            add_object(sim, spec["handle"], spec) for spec in dynamic_specs
        ]
        add_object(sim, static_spec["handle"], static_spec)

        timings = {}
        timed("update_KDtree", timings, sim.update_KDtree)
        scene_mesh = mesh_summary(sim.scene_mesh, args.include_full_mesh)

        timed("initial_update_dynamic_KDtree", timings, sim.update_dynamic_KDtree)
        initial_object_mesh = mesh_summary(sim.object_mesh, args.include_full_mesh)

        points = {
            "dynamic_0_center": dynamic_specs[0]["translation"],
            "dynamic_1_center": dynamic_specs[1]["translation"],
            "static_object_center": static_spec["translation"],
            "near_floor": [0.0, 0.05, 0.0],
            "scene_wall": [29.75, 1.0, 0.0],
            "outside_x": [100.0, 1.0, 0.0],
            "outside_y": [0.0, 100.0, 0.0],
            "outside_z": [0.0, 1.0, 100.0],
        }
        flags = {
            "default_like": [True, True],
            "scene_and_object": [True, True],
            "scene_only": [False, True],
            "object_only": [True, False],
            "neither": [False, False],
        }
        shapes = ["flat", "column"]

        collision = {}
        for point_name, point in points.items():
            collision[point_name] = {}
            for shape in shapes:
                collision[point_name][shape] = {}
                for flag_name, (object_collidable, scene_collidable) in flags.items():
                    collision[point_name][shape][flag_name] = query_collision(
                        sim,
                        point,
                        args.max_search_radius,
                        object_collidable,
                        scene_collidable,
                        shape,
                    )

        moved_points = []
        for step in range(3):
            for ix, obj in enumerate(dynamic_objects):
                base = np.array(dynamic_specs[ix]["translation"], dtype=np.float32)
                delta = np.array([0.35 * (step + 1), 0.1 * ix, -0.2 * (step + 1)])
                obj.translation = base + delta
                obj.rotation = quat_from_yaw(dynamic_specs[ix]["yaw_degrees"] + 15.0 * step)
            timed(f"move_{step}_update_dynamic_KDtree", timings, sim.update_dynamic_KDtree)
            moved_points.append(
                {
                    "step": step,
                    "object_mesh": mesh_summary(sim.object_mesh, False),
                    "queries": {
                        f"dynamic_{ix}_center": query_collision(
                            sim,
                            np.asarray(obj.translation, dtype=np.float32).tolist(),
                            args.max_search_radius,
                            True,
                            True,
                            "flat",
                        )
                        for ix, obj in enumerate(dynamic_objects)
                    },
                }
            )

        timed("refresh_update_KDtree", timings, sim.update_KDtree)
        timed("refresh_update_dynamic_KDtree", timings, sim.update_dynamic_KDtree)
        refresh_query = query_collision(
            sim, np.asarray(dynamic_objects[0].translation).tolist(), args.max_search_radius, True, True, "flat"
        )

        return {
            "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
            "scene": args.scene,
            "scene_dataset": args.scene_dataset,
            "enable_physics": args.enable_physics,
            "create_renderer": args.create_renderer,
            "objects": {
                "dynamic": dynamic_specs,
                "static": static_spec,
            },
            "scene_mesh": scene_mesh,
            "initial_object_mesh": initial_object_mesh,
            "collision": collision,
            "moved_points": moved_points,
            "refresh_query": refresh_query,
            "timings": timings,
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset", default="")
    parser.add_argument("--object", action="append", required=True)
    parser.add_argument("--max-search-radius", type=float, default=10.0)
    parser.add_argument("--enable-physics", action="store_true")
    parser.add_argument("--create-renderer", action="store_true")
    parser.add_argument("--include-full-mesh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_case(args), sort_keys=True))


if __name__ == "__main__":
    main()
