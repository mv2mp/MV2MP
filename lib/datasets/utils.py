import numpy as np

from scipy.spatial.transform import Rotation as R


def extract_pose_and_rodrigues(world_to_cam):
    # Ensure the matrix is a numpy array
    world_to_cam = np.array(world_to_cam)

    # Extract translation vector
    translation = world_to_cam[:3, 3]

    # Extract rotation matrix
    rotation_matrix = world_to_cam[:3, :3]

    # Convert rotation matrix to Rodriguez rotation vector
    rotation = R.from_matrix(rotation_matrix)
    rodrigues_rotation = rotation.as_rotvec()

    return translation.astype(np.float32), rodrigues_rotation.astype(np.float32)
