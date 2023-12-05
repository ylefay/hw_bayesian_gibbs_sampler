import pickle

import jax

from gibbs import gibbs_per_block
from simulate_data import generate_dataset, initialize_parameters
import jax.numpy as jnp

k = 100
s_list = [5]
Ry_list = [0.6]
no_datasets = 1
datasetsX, datasets = generate_dataset(s_list, Ry_list, no_datasets)

ITERATION = 2000

res = dict()
for i in datasets.keys():
    X = datasetsX[i]
    dataset = datasets[i]
    for (s, Ry) in dataset.keys():
        Y, beta_v, z_v, sigma2_v, q_v = dataset[(s, Ry)]
        # init = z_v, beta_v, sigma2_v, q_v
        q, z, beta, sigma2 = initialize_parameters(X, Y)
        X = jnp.array(X)
        Y = jnp.array(Y)
        z = jnp.array(z)
        beta = jnp.array(beta)
        init = (q, z, beta, sigma2)
        with jax.disable_jit(False):
            res_gibbs = gibbs_per_block(X, Y, init, ITERATION=ITERATION)
        with open(f'out_{i}_{s}_{Ry}.pickle', 'wb') as handle:
            pickle.dump(res_gibbs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # res[i, s, Ry] = res_gibbs
        # whateveryouwant

# with open('out.pickle', 'wb') as handle:
#    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
