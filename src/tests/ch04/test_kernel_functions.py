import numpy as np
import sys; sys.path.append('./src/'); sys.path.append('../../')

from ch04 import gaussian_kernel, kernel_density_estimate


def test_kernel_functions():
    data = np.array([0.1, 0.1, 0.1,
                    0.15,
                    0.2, 0.2, 0.2,
                    0.25, 0.25,
                    0.3, 0.3, 0.3, 0.3,
                    0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                    0.45, 0.45, 0.45,
                    0.5,
                    0.55, 0.55, 0.55, 0.55, 0.55,
                    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                    0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                    0.75, 0.75,
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                    0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,
                    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                    0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
                    1.00, 1.00, 1.00, 1.00, 1.00,
                    1.05, 1.05, 1.05,
                    1.10, 1.10,
                    1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15,
                    1.20, 1.20, 1.20, 1.20,
                    1.25,
                    1.30,
                    1.35, 1.35, 1.35,
                    1.40,
                    1.50,
                    1.55, 1.55, 1.55,
                    1.60])
    t = np.linspace(0, 2, 100)

    for b in [0.01, 0.05, 0.1, 0.2]:
        try:
            kde = kernel_density_estimate(gaussian_kernel(b), data)
            y = np.array([kde(t_i) for t_i in t])
        except Exception as exc:
            assert False, "b = {b}, {exc}"
