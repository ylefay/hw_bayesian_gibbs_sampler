import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)
k = 100
T = 200
l = 0.
a = 1.
b = 1.
A = 1.
B = 1.
grid = jnp.array([0.001, 0.002, 0.003, 0.004, 0.005,
                  0.006, 0.007, 0.008, 0.009, 0.01,
                  0.011, 0.012, 0.013, 0.014, 0.015,
                  0.016, 0.017, 0.018, 0.019, 0.02,
                  0.021, 0.022, 0.023, 0.024, 0.025,
                  0.026, 0.027, 0.028, 0.029, 0.03,
                  0.031, 0.032, 0.033, 0.034, 0.035,
                  0.036, 0.037, 0.038, 0.039, 0.04,
                  0.041, 0.042, 0.043, 0.044, 0.045,
                  0.046, 0.047, 0.048, 0.049, 0.05,
                  0.051, 0.052, 0.053, 0.054, 0.055,
                  0.056, 0.057, 0.058, 0.059, 0.06,
                  0.061, 0.062, 0.063, 0.064, 0.06500001,
                  0.06600001, 0.067, 0.068, 0.06900001, 0.07000001,
                  0.071, 0.072, 0.07300001, 0.07400001, 0.075,
                  0.07600001, 0.07700001, 0.07800001, 0.079, 0.08000001,
                  0.08100001, 0.082, 0.083, 0.08400001, 0.08500001,
                  0.086, 0.087, 0.08800001, 0.08900001, 0.09,
                  0.09100001, 0.09200001, 0.09300001, 0.094, 0.09500001,
                  0.09600001, 0.097, 0.098, 0.09900001, 0.10000001,
                  0.11, 0.12, 0.13, 0.13999999, 0.14999999,
                  0.16, 0.16999999, 0.17999998, 0.18999998, 0.19999999,
                  0.20999998, 0.21999997, 0.22999997, 0.23999996, 0.24999997,
                  0.26, 0.26999998, 0.27999997, 0.28999996, 0.29999995,
                  0.30999994, 0.31999993, 0.32999995, 0.33999997, 0.34999996,
                  0.35999995, 0.36999995, 0.37999994, 0.38999993, 0.39999992,
                  0.40999997, 0.41999996, 0.42999995, 0.43999994, 0.44999993,
                  0.45999992, 0.4699999, 0.4799999, 0.48999995, 0.49999994,
                  0.50999993, 0.5199999, 0.5299999, 0.5399999, 0.5499999,
                  0.5599999, 0.56999993, 0.5799999, 0.5899999, 0.5999999,
                  0.6099999, 0.6199999, 0.6299999, 0.6399999, 0.6499999,
                  0.6599999, 0.6699999, 0.6799999, 0.6899999, 0.69999987,
                  0.7099999, 0.7199999, 0.7299999, 0.7399999, 0.7499999,
                  0.7599999, 0.76999986, 0.77999985, 0.78999984, 0.7999999,
                  0.8099999, 0.8199999, 0.82999986, 0.83999985, 0.84999985,
                  0.85999984, 0.8699999, 0.8799999, 0.88999987, 0.89999986,
                  0.901, 0.902, 0.903, 0.904, 0.905,
                  0.90599996, 0.90699995, 0.90799993, 0.9089999, 0.9099999,
                  0.9109999, 0.9119999, 0.91299987, 0.91399986, 0.91499984,
                  0.9159998, 0.9169998, 0.9179998, 0.9189998, 0.9199998,
                  0.92099977, 0.92199975, 0.92299974, 0.9239997, 0.9249997,
                  0.9259997, 0.9269997, 0.9279997, 0.92899966, 0.92999965,
                  0.93099964, 0.9319996, 0.9329996, 0.9339996, 0.9349996,
                  0.9359996, 0.93699956, 0.93799955, 0.93899953, 0.9399995,
                  0.9409995, 0.9419995, 0.9429995, 0.94399947, 0.94499946,
                  0.94599944, 0.94699943, 0.9479994, 0.9489994, 0.9499994,
                  0.9509994, 0.95199937, 0.95299935, 0.95399934, 0.9549993,
                  0.9559993, 0.9569993, 0.9579993, 0.9589993, 0.95999926,
                  0.96099925, 0.96199924, 0.9629992, 0.9639992, 0.9649992,
                  0.9659992, 0.9669992, 0.96799916, 0.96899915, 0.96999913,
                  0.9709991, 0.9719991, 0.9729991, 0.9739991, 0.97499907,
                  0.97599906, 0.97699904, 0.97799903, 0.978999, 0.979999,
                  0.980999, 0.981999, 0.98299897, 0.98399895, 0.98499894,
                  0.9859989, 0.9869989, 0.9879989, 0.9889989, 0.9899989,
                  0.99099886, 0.99199885, 0.99299884, 0.9939988, 0.9949988,
                  0.9959988, 0.9969988, 0.9979988, 0.99899876])
surface = jnp.array([0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.000998, 0.000998, 0.000998, 0.000998, 0.000998,
                     0.00099799, 0.000998, 0.000998, 0.000998, 0.00099799,
                     0.000998, 0.000998, 0.000998, 0.00099799, 0.000998,
                     0.000998, 0.000998, 0.00099799, 0.000998, 0.000998,
                     0.00099799, 0.000998, 0.000998, 0.000998, 0.00099799,
                     0.000998, 0.000998, 0.000998, 0.00099799, 0.000998,
                     0.000998, 0.000998, 0.00099799, 0.000998, 0.000998,
                     0.00099799, 0.000998, 0.000998, 0.000998, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00997999, 0.00997999,
                     0.00997998, 0.00997998, 0.00997999, 0.00997999, 0.00997998,
                     0.00997998, 0.00997999, 0.00997998, 0.00997999, 0.00998001,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00997998, 0.00998001, 0.00998001, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00998004,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00998004, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00998004, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00998004, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00998004,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00998004, 0.00997998,
                     0.00997998, 0.00997998, 0.00997998, 0.00997998, 0.00997998,
                     0.00998004, 0.00997998, 0.00997998, 0.00997998, 0.00099816,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799, 0.00099799,
                     0.00099799, 0.00099799, 0.00099799, 0.00099799])


def sz(z_v):
    return jnp.sum(z_v)


def vbar(X):
    return jnp.mean(jnp.var(X, axis=0))


def gamma2(R2_v, q_v, vbarX_v):
    return R2_v / (q_v * k * vbarX_v * (1 - R2_v))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    prod = Xtilde_v.T @ Xtilde_v
    return prod + jnp.eye(k) / gamma2_v


def Xtilde(X, z_v):
    def iter(inps, i):
        j, _X = inps
        _X, i, j = jax.lax.cond(z_v.at[i].get() == True, lambda _: (_X.at[:, j].set(X.at[:, i].get()), i + 1, j + 1),
                                lambda _i: (_X, i + 1, j), None)
        return (j, _X), None

    res, _ = jax.lax.scan(iter, (0, jnp.zeros((T, k))), jnp.arange(k))
    _, X_out = res
    return X_out


def betahat(Wtilde_v, Xtilde_v, Y):
    return jnp.linalg.solve(Wtilde_v, Xtilde_v.T @ Y)


@jax.jit
def R2q(OP_key, X, z, beta_v, sigma2_v):
    sz_v = sz(z)
    bz = beta_v @ jnp.diag(z) @ beta_v.T
    vbarX_v = vbar(X)

    def joint_pdf(q_v, R2_v):
        return jnp.exp((-bz / (2 * sigma2_v * gamma2(R2_v, q_v, vbarX_v)))) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    @jnp.vectorize
    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior
        _univariate_pdf = lambda R2_v: joint_pdf(q_v, R2_v)
        return jnp.sum(_univariate_pdf(grid) * surface)

    def cdf(pdf):
        weights = pdf(grid) * surface
        cdf = jnp.cumsum(weights)
        cdf /= cdf[-1]
        return cdf

    @partial(jax.jit, static_argnums=(0,))
    def invCDF(cdf, u):
        index = jax.lax.cond(
            jnp.any(cdf >= u),
            lambda _: jnp.argmax(cdf >= u),
            lambda _: jnp.argmax(cdf),  # If u is larger than any element in cdf, return the last index.
            operand=None
        )
        return grid.at[index].get()

    cdfq = cdf(univariate_pdf)

    @jax.jit
    def sampleqR():
        uv = jax.random.uniform(OP_key, shape=(2,))
        q = invCDF(cdfq, uv.at[0].get())

        cdfRconditiononq = cdf(lambda R2: joint_pdf(q, R2))
        R2 = invCDF(cdfRconditiononq, uv.at[1].get())
        return R2, q

    return sampleqR()  # function that will be looped over to generate samples of (q, R) given X z


@jax.jit
def z(OP_key, Y, X, R2_v, q_v, z_v):
    gamma2_v = gamma2(R2_v, q_v, vbar(X))

    def pdf_ratio(index, z_v):
        """
        Computhe the logratio up to proportionaly of P(Z_i = 1, Z_{-i}) / P(Z_i = 0, Z_{-i})
        """
        z_v = z_v.at[index].set(0)
        sz_v = sz(z_v)

        Xtilde_v0 = Xtilde(X, z_v)
        Wtilde_v0_for_det = Wtilde(Xtilde_v0, sz_v, gamma2_v)
        betahat_v0 = betahat(Wtilde_v0_for_det, Xtilde_v0, Y)
        _, logdet0 = jnp.linalg.slogdet(Wtilde_v0_for_det)

        z_v = z_v.at[index].set(1)
        sz_v += 1
        Xtilde_v1 = Xtilde(X, z_v)
        Wtilde_v1_for_det = Wtilde(Xtilde_v1, sz_v, gamma2_v)
        betahat_v1 = betahat(Wtilde_v1_for_det, Xtilde_v1, Y)
        _, logdet1 = jnp.linalg.slogdet(Wtilde_v1_for_det)

        Ysquared = Y.T @ Y

        log_ratio = jnp.log(q_v) - jnp.log(1 - q_v) - 1 / 2 * jnp.log(gamma2_v) - 1 / 2 * (logdet1 - logdet0) - \
                    T / 2 * (jnp.log(Ysquared - betahat_v1.T @ Wtilde_v1_for_det @ betahat_v1) - jnp.log(
            Ysquared - betahat_v0.T @ Wtilde_v0_for_det @ betahat_v0))
        log_ratio += -jnp.log(gamma2_v)  # correction due to constant size matrix
        ratio = jnp.exp(log_ratio)
        return ratio

    def pdf_exclusion(index, z):
        """
        P(z_i | z_{-i}) = 1 / (1+P(z_i | z_{-i})/P(1-z_i | z_{-i}))
        """
        ratio = pdf_ratio(index, z)
        zi = z.at[index].get()
        return jax.lax.cond(zi == 0, lambda _: jnp.exp(-jnp.log1p(ratio)), lambda _: jnp.exp(-jnp.log1p(1 / ratio)),
                            None)

    def gibbs(z):
        u = jax.random.uniform(OP_key, shape=(k,))

        def iter(z, i):
            p = pdf_exclusion(i, z)
            z = jax.lax.cond(u.at[i].get() > p, lambda _: z.at[i].set(1 - z.at[i].get()), lambda _: z,
                             None)  # not optimal
            return z, None

        z, _ = jax.lax.scan(iter, z, jnp.arange(k))
        return z

    return gibbs(z_v)


@jax.jit
def sigma2(OP_key, Y, X, R2_v, q_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, vbar(X))
    Xtilde_v = Xtilde(X, z)
    Wtilde_v_for_det = Wtilde(Xtilde_v, sz_v, gamma2_v)
    betahat_v = betahat(Wtilde_v_for_det, Xtilde_v, Y)
    form = T / 2
    param = (Y.T @ Y - betahat_v.T @ Wtilde_v_for_det @ betahat_v) / 2
    scale = 1 / param
    # Lorsqu'on regroupera, toute cette initialisation de variables _v ne sera évidemment à faire qu'une fois.
    return 1 / (jax.random.gamma(OP_key, a=form) * scale)  # Ytilde=Y


@jax.jit
def betatilde(OP_key, Y, X, R2_v, q_v, sigma2_v, z_v):
    sz_v = sz(z_v)
    gamma2_v = gamma2(R2_v, q_v, vbar(X))
    Xtilde_v = Xtilde(X, z_v)
    id = jnp.eye(k)
    Wtilde_v_p = Wtilde(Xtilde_v, sz_v, gamma2_v)
    invTerm = jnp.linalg.pinv(Wtilde_v_p)
    mean = invTerm @ Xtilde_v.T @ Y  # Pas de U*phi
    cov = invTerm * sigma2_v
    sample = mean + jnp.linalg.cholesky(cov) @ jax.random.normal(OP_key, shape=(k,))

    def reconstruct_beta_v():
        def iter(inps, i):
            j, _beta_v = inps
            _beta_v, i, j = jax.lax.cond(z_v.at[i].get() == True,
                                         lambda _: (_beta_v.at[i].set(sample.at[j].get()), i + 1, j + 1),
                                         lambda _i: (_beta_v, i + 1, j), None)
            return (j, _beta_v), None

        res, _ = jax.lax.scan(iter, (0, jnp.zeros(shape=(k,))), jnp.arange(k))
        _, beta_v = res
        return beta_v

    beta_v = reconstruct_beta_v()
    return beta_v
