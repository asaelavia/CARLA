"""
Based on https://github.com/amirhk/recourse/blob/master/loadSCM.py
"""
from carla.data.load_scm.distributions import *


def sanity_3_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -x1 + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical = []
    immutables = []

    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )

def sanity_5_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -x1,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.3 * x1 + 0.7 * x2) ,
        "x4": lambda n_samples, x2, x3: 0.4 * x2 - 0.6 * x3,
        "x5": lambda n_samples, x1, x3, x4: 0.2 * x1 + 0.3 * x3 - 0.5 * x4,
        # "x1": lambda n_samples: n_samples,
        # "x2": lambda n_samples, x1: -x1 + n_samples,
        # "x3": lambda n_samples, x1, x2: 0.5 * (0.3 * x1 + 0.7 * x2) + n_samples,
        # "x4": lambda n_samples, x2, x3: 0.4 * x2 - 0.6 * x3 + n_samples,
        # "x5": lambda n_samples, x1, x3, x4: 0.2 * x1 + 0.3 * x3 - 0.5 * x4 + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
        "u5": Normal(0, 1),
    }
    continuous = ["x1", "x2", "x3", "x4", "x5"]
    categorical = []
    immutables = []


    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )



import numpy as np
from typing import Dict, List, Tuple


def german_credit() -> Tuple[Dict, Dict, Dict, List, List, List]:
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples: n_samples,
        "x3": lambda n_samples: n_samples,
        "x4": lambda n_samples, x1, x3: 1000 + 50 * x1 + 500 * x3 + n_samples,
        "x5": lambda n_samples, x1, x3: np.maximum(0, x1 - 18 - 3 * x3 + n_samples),
        "x6": lambda n_samples, x1, x4: np.maximum(0, 0.1 * x4 * (x1 - 18) / 10 + n_samples),
        "x7": lambda n_samples, x4, x3, x5: np.maximum(0,0.2 * x4 + 1000 * x3 - 500 * x5 + n_samples),
        # "x8": lambda n_samples, x4, x6, x5, x7, x1, x3: (
        #         0.3 * x4 / 1000 +
        #         0.2 * x6 / 1000 +
        #         0.2 * x5 -
        #         0.3 * x7 / 1000 +
        #         0.1 * x1 / 10 +
        #         0.1 * x3 +
        #         n_samples
        # ),
    }

    structural_equations_ts = structural_equations_np

    noises_distributions = {
        "u1": Uniform(18, 71),
        "u2": Bernoulli(0.6),
        "u3": Uniform(0, 3),
        "u4": Normal(0, 500),
        "u5": Normal(0, 2),
        "u6": Normal(0, 1000),
        "u7": Normal(0, 2000),
        # "u8": Normal(0, 0.5),
    }

    # continuous = ["x1", "x4", "x5", "x6", "x7", "x8"]
    continuous = ["x1", "x4", "x5", "x6", "x7"]
    categorical = ["x2", "x3"]
    immutables = ["x2"]

    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )