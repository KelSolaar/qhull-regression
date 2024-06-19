import numpy as np
from scipy.spatial import Delaunay


RANDOM_STATE = np.random.RandomState(32)


def vecmul(m, v):
    return np.matmul(m, v[..., None]).squeeze(-1)


def tstack(a):
    a = np.asarray(a)

    if a.ndim <= 2:
        return np.transpose(a)

    return np.concatenate([x[..., None] for x in a], axis=-1)


def random_triplet_generator(
    size,
    limits=np.array([[0, 1], [0, 1], [0, 1]]),
    random_state=RANDOM_STATE,
):
    limit_x, limit_y, limit_z = limits

    return tstack(
        [
            random_state.uniform(limit_x[0], limit_x[1], size=size),
            random_state.uniform(limit_y[0], limit_y[1], size=size),
            random_state.uniform(limit_z[0], limit_z[1], size=size),
        ]
    )


def is_within_mesh_volume(points, mesh, tolerance=100 * np.finfo(np.float_).eps):
    triangulation = Delaunay(mesh)

    simplex = triangulation.find_simplex(points, tol=tolerance)
    simplex = np.where(simplex >= 0, True, False)

    return simplex


def RGB_colourspace_volume_coverage_MonteCarlo(samples: int = 1000000) -> float:
    XYZ = random_triplet_generator(samples, random_state=RANDOM_STATE)
    vertices = np.load("vertices.npz")["arr_0"]
    XYZ_vs = XYZ[is_within_mesh_volume(XYZ, vertices)]

    RGB = vecmul(
        np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ]
        ),
        XYZ_vs,
    )

    RGB_c = RGB[np.logical_and(np.min(RGB, axis=-1) >= 0, np.max(RGB, axis=-1) <= 1)]

    return 100 * RGB_c.size / XYZ_vs.size
