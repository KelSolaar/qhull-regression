from qhull_regression.common import *


def test_RGB_colourspace_volume_coverage_MonteCarlo():
    np.testing.assert_allclose(
        RGB_colourspace_volume_coverage_MonteCarlo(int(10e3)),
        47.25326992,
    )
