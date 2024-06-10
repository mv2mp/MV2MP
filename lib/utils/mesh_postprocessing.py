import numpy as np
import pymeshlab
import trimesh
import trimesh.intersections
import trimesh.proximity


MINIMAL_HOOF_FACES_COUNT = 40

LEFT_FOOT_FACE_INDICES = np.unique(
    [
        5520,
        5521,
        5530,
        5531,
        5536,
        5537,
        5546,
        5547,
        5554,
        5554,
        5555,
        5564,
        5565,
        5566,
        5566,
        5567,
        5567,
        5584,
        5585,
        5586,
        5587,
        5597,
        5598,
        5599,
        5616,
        5618,
        5619,
        5632,
        5635,
        5774,
        5775,
        5778,
        5779,
        5780,
        5781,
        5782,
        5783,
        5784,
        5785,
        5910,
        5911,
        5912,
        5913,
        5914,
        5915,
        5916,
        5917,
        5918,
        5919,
        5920,
        5921,
        5922,
        5923,
        5924,
        5925,
        5926,
        5927,
        5928,
        5929,
        5930,
        5931,
        5932,
        5933,
        5934,
        5935,
        5936,
        5937,
        5938,
        5939,
        5940,
        5941,
        5942,
        5943,
        5944,
        5945,
        5946,
        5947,
        5948,
        5949,
        5950,
        5951,
        5952,
        5953,
        5954,
        5955,
        5956,
        5957,
        5958,
        5959,
        5960,
        5961,
        5962,
        5963,
        5964,
        5965,
        5966,
        5967,
        5972,
        5973,
        5978,
        5979,
        5981,
        5983,
        5986,
        5987,
        5988,
        5989,
        5990,
        5991,
        5992,
        5993,
        5994,
        5995,
        6436,
        6437,
        6438,
        6439,
        6440,
        6441,
        6442,
        6444,
        6445,
        5969,
        5968,
        5971,
        5970,
        5974,
        5975,
        5976,
        5977,
        5982,
        6443,
        5980,
        5634,
    ]
)

RIGHT_FOOT_FACE_INDICES = np.unique(
    [
        12408,
        12409,
        12455,
        12475,
        12474,
        12418,
        12419,
        12424,
        12486,
        12487,
        12506,
        12522,
        12883,
        12881,
        12879,
        12877,
        12874,
        12875,
        12876,
        12879,
        12878,
        12880,
        12882,
        12673,
        12670,
        12671,
        12668,
        12669,
        12666,
        12667,
        12662,
        12663,
        12801,
        12800,
        12803,
        12802,
        12805,
        12807,
        12804,
        12810,
        12811,
        12813,
        12822,
        12825,
        12824,
        12831,
        12823,
        12821,
        12820,
        12835,
        12832,
        12830,
        12845,
        12844,
        13329,
        13328,
        12858,
        12863,
        12862,
        13330,
        12860,
        12849,
        12838,
        12840,
        12861,
        12863,
        12858,
        12839,
        12849,
        12848,
        13324,
        13325,
        12850,
        12852,
        12868,
    ]
)


def get_faces_vertices(mesh: trimesh.Trimesh, faces_ids: np.ndarray):
    verts_ids = mesh.faces[faces_ids].reshape(-1)
    return mesh.vertices[np.unique(verts_ids)]


def get_plane_fitted_to_faces(mesh: trimesh.Trimesh, faces_ids: np.ndarray):
    some_face_normal = mesh.face_normals[faces_ids[0]]
    vertices = get_faces_vertices(mesh, faces_ids)
    plane_origin, plane_normal = trimesh.points.plane_fit(vertices)
    # ensure thar plane_normal and some_face_normal are directed in the same way
    if np.dot(plane_normal, some_face_normal) < 0:
        plane_normal = -plane_normal
    return plane_origin, plane_normal


def compute_mesh_difference(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(mesh_b.vertices, mesh_b.faces), mesh_name="mesh_b")
    mesh_set.add_mesh(pymeshlab.Mesh(mesh_a.vertices, mesh_a.faces), mesh_name="mesh_a")
    mesh_set.generate_boolean_difference(first_mesh=1, second_mesh=0)  #
    return trimesh.Trimesh(
        vertices=mesh_set.current_mesh().vertex_matrix(), faces=mesh_set.current_mesh().face_matrix(), process=True
    )


def compute_mesh_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(mesh_b.vertices, mesh_b.faces), mesh_name="mesh_b")
    mesh_set.add_mesh(pymeshlab.Mesh(mesh_a.vertices, mesh_a.faces), mesh_name="mesh_a")
    mesh_set.generate_boolean_intersection(first_mesh=1, second_mesh=0)
    return trimesh.Trimesh(
        vertices=mesh_set.current_mesh().vertex_matrix(), faces=mesh_set.current_mesh().face_matrix(), process=True
    )


def cut_hoof_by_foot_plane(reconstructed_mesh: trimesh.Trimesh, smpl_mesh: trimesh.Trimesh, foot_indices: np.ndarray):
    lp_origin, lp_normal = get_plane_fitted_to_faces(smpl_mesh, foot_indices)

    # WARNING: this call will fail without shapely and triangle being installed
    # pip install shapely
    # pip install --extra-index-url https://pypi.python.org/simple triangle
    sliced_mesh: trimesh.Trimesh = trimesh.intersections.slice_mesh_plane(
        reconstructed_mesh, lp_normal, lp_origin, cap=True
    )

    if len(sliced_mesh.faces) < MINIMAL_HOOF_FACES_COUNT:
        return reconstructed_mesh

    connected_components = sliced_mesh.split()
    points = get_faces_vertices(smpl_mesh, foot_indices)

    min_distance = np.inf
    closest_connected_component = None
    for component in connected_components:
        closest, distance, _ = trimesh.proximity.closest_point(component, points)
        if np.array(distance).mean() < min_distance:
            min_distance = np.array(distance).mean()
            closest_connected_component = component
    assert closest_connected_component is not None, f"Can't find the closest component, {len(connected_components)}"
    cut_mesh = compute_mesh_difference(reconstructed_mesh, closest_connected_component.convex_hull)
    return cut_mesh


def remove_hooves(reconstructed_mesh: trimesh.Trimesh, smpl_mesh: trimesh.Trimesh):
    cut_mesh_a = cut_hoof_by_foot_plane(reconstructed_mesh, smpl_mesh, LEFT_FOOT_FACE_INDICES)
    cut_mesh_b = cut_hoof_by_foot_plane(reconstructed_mesh, smpl_mesh, RIGHT_FOOT_FACE_INDICES)
    cut_mesh = compute_mesh_intersection(cut_mesh_a, cut_mesh_b)
    return cut_mesh
