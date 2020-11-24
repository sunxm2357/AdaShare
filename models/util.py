import sys
sys.path.insert(0, '..')
import torch
import numpy as np
from utils.flops_benchmark import add_flops_counting_methods


def count_params(model):
    num_params = 0.
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        v_shape = v.shape
        num_params += np.prod(v_shape)

    print('Number of Parameters = %.2f M' % (num_params/1e6))


def compute_flops(model, input, kwargs_dict):
    model = add_flops_counting_methods(model)
    model = model.cuda().train()

    model.start_flops_count()

    _ = model(input, **kwargs_dict)
    gflops = model.compute_average_flops_cost()

    return gflops