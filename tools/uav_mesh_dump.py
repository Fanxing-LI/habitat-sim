import argparse
import json

import numpy as np

import habitat_sim


def make_sim(scene):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))


def object_mesh_summary(sim, object_handle):
    obj_mgr = sim.get_rigid_object_manager()
    template_mgr = sim.get_object_template_manager()
    template_id = template_mgr.load_configs(object_handle)[0]
    obj = obj_mgr.add_object_by_template_id(template_id)
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    obj.translation = np.array([1.25, 0.75, -0.5], dtype=np.float32)
    obj.rotation = habitat_sim.utils.common.quat_to_magnum(
        habitat_sim.utils.common.quat_from_angle_axis(
        np.deg2rad(37.0), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
    )

    sim.update_dynamic_KDtree()
    mesh = sim.object_mesh
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data/test_assets/scenes/simple_room.glb")
    parser.add_argument("--object", default="data/test_assets/objects/chair.object_config.json")
    args = parser.parse_args()

    sim = make_sim(args.scene)
    try:
        print(json.dumps(object_mesh_summary(sim, args.object), sort_keys=True))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
