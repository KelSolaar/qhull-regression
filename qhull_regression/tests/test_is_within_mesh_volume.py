from qhull_regression.common import *


def test_is_within_mesh_volume():
    vertices = np.load("vertices.npz")["arr_0"]

    assert is_within_mesh_volume(np.array([0.3205, 0.4131, 0.5100]), vertices)

    assert not is_within_mesh_volume(np.array([-0.0005, 0.0031, 0.0010]), vertices)

    assert is_within_mesh_volume(np.array([0.4325, 0.3788, 0.1034]), vertices)

    assert not is_within_mesh_volume(np.array([0.0025, 0.0088, 0.0340]), vertices)
