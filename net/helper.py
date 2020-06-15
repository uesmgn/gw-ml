import numpy as np

def get_middle_dim(input_dim, pool_kernels):
    prod = int(np.prod(pool_kernels))
    if input_dim % prod == 0:
        middle_dim = input_dim
        for p in pool_kernels:
            middle_dim //= p
        return middle_dim
    else:
        raise ValueError(f'The input dimention must be a multiple of {prod}')
