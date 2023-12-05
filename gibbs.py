import jax
import jax.numpy as jnp
import cond_posteriors as cp
from progress_bar import progress_bar

k = 100

OP_key = jax.random.PRNGKey(1)


def gibbs_per_block(X, Y, init, ITERATION=5000):
    @progress_bar(num_samples=ITERATION, message=f"iter={ITERATION}")
    def iter_gibbs(carry, inps):
        _, key = inps
        R2_v, q_v, z_v, beta_v, sigma2_v = carry
        key1, key2, key3, key4 = jax.random.split(key, 4)
        R2_v, q_v = cp.R2q(key1, X, z_v, beta_v, sigma2_v)
        z_v = cp.z(key2, Y, X, R2_v, q_v, z_v)
        sigma2_v = cp.sigma2(key3, Y, X, R2_v, q_v, z_v)
        beta_v = cp.betatilde(key4, Y, X, R2_v, q_v, sigma2_v, z_v)
        return (R2_v, q_v, z_v, beta_v, sigma2_v), (R2_v, q_v, z_v, beta_v, sigma2_v)

    keys = jax.random.split(OP_key, ITERATION)
    init = (1.0, *init)
    _, out = jax.lax.scan(iter_gibbs, init, (jnp.arange(ITERATION), keys))
    return out
