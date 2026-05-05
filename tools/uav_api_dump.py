import argparse
import json

import numpy as np

import habitat_sim


def as_list(value):
    return np.asarray(value, dtype=np.float64).tolist()


def mesh_summary(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return {
        "vertex_count": int(vertices.shape[0]),
        "face_index_count": int(faces.shape[0]),
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
        "sum": vertices.sum(axis=0).tolist(),
        "mean": vertices.mean(axis=0).tolist(),
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
    }


def make_sim(scene):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))


def add_dynamic_object(sim, object_handle, translation, yaw_degrees):
    obj_mgr = sim.get_rigid_object_manager()
    template_mgr = sim.get_object_template_manager()
    template_id = template_mgr.load_configs(object_handle)[0]
    obj = obj_mgr.add_object_by_template_id(template_id)
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    obj.translation = np.array(translation, dtype=np.float32)
    obj.rotation = habitat_sim.utils.common.quat_to_magnum(
        habitat_sim.utils.common.quat_from_angle_axis(
            np.deg2rad(yaw_degrees), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
    )
    return obj


def collision_record_to_dict(record):
    return {
        "hit_pos": as_list(record.hit_pos),
        "is_out_bound": bool(record.is_out_bound),
    }


def run_case(scene, object_handle):
    sim = make_sim(scene)
    try:
        add_dynamic_object(sim, object_handle, [1.25, 0.75, -0.5], 37.0)

        sim.update_KDtree()
        scene_mesh = mesh_summary(sim.scene_mesh)

        sim.update_dynamic_KDtree()
        object_mesh = mesh_summary(sim.object_mesh)

        points = {
            "origin": [0.0, 0.5, 0.0],
            "object_center": [1.25, 0.75, -0.5],
            "near_object": [1.25, 0.75, -0.25],
            "near_floor": [0.0, 0.05, 0.0],
            "outside_x": [100.0, 0.5, 0.0],
            "outside_y": [0.0, 100.0, 0.0],
            "outside_z": [0.0, 0.5, 100.0],
        }
        flags = {
            "scene_and_object": [True, True],
            "scene_only": [False, True],
            "object_only": [True, False],
            "neither": [False, False],
        }

        collision = {}
        for point_name, point in points.items():
            collision[point_name] = {}
            for flag_name, (object_collidable, scene_collidable) in flags.items():
                record = sim.get_closest_collision_point(
                    np.array(point, dtype=np.float32),
                    10.0,
                    object_collidable,
                    scene_collidable,
                )
                collision[point_name][flag_name] = collision_record_to_dict(record)

        # Exercise repeated refresh: this caught regressions in tree reset behavior.
        sim.update_KDtree()
        sim.update_dynamic_KDtree()
        refresh_record = collision_record_to_dict(
            sim.get_closest_collision_point(
                np.array(points["near_object"], dtype=np.float32), 10.0, True, True
            )
        )

        return {
            "scene": scene,
            "object": object_handle,
            "scene_mesh": scene_mesh,
            "object_mesh": object_mesh,
            "collision": collision,
            "refresh_near_object": refresh_record,
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data/test_assets/scenes/simple_room.glb")
    parser.add_argument("--object", default="data/test_assets/objects/chair.object_config.json")
    args = parser.parse_args()
    print(json.dumps(run_case(args.scene, args.object), sort_keys=True))


if __name__ == "__main__":
    main()
