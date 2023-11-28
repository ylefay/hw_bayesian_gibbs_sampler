import numpy as np
from scipy.stats import invgamma
from scipy.stats import multivariate_normal as mnormal
from scipy.special import gamma, gammaincc, comb
from simulate_data import generate_dataset
from scipy.integrate import quad, dblquad



k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1


def sz(z):
    return np.sum(z)


def vbar(X):
    return np.mean(np.var(X, axis=0))


def gamma2(R2_v, q_v, X):
    return (1 / (1 - R2_v) - 1) * 1 / (q_v * k * vbar(X))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z):
    return X[:, z == 1]


def sigma2_data(beta_v, X, Ry):  # le sigma_2 qui serviraà générer le jeu de données.
    return (1 / (Ry - 1)) * np.mean((beta_v @ X.T) ** 2, axis=0)


def betahat(Wtildeinv_v, Xtilde_v, Y):
    return Wtildeinv_v @ Xtilde_v.T @ Y





def R2q(X, z, sigma2_v, beta_v):
    def joint_pdf(q_v, R2_v, X, z, beta_v, sigma2_v):
        bz = beta_v @ np.diag(z) @ beta_v.T
        sz_v = sz(z)
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbar(X) * q_v * ((1 - R2_v) / R2_v) * bz)) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    grid_q = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in range(900, 1000)]
    grid_R2 = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in
                                                                                      range(900, 1000)]
    
    
    def f(R2,q):
        return joint_pdf(q,R2,X,z,beta_v,sigma2_v)
    norm=dblquad(f, 10**(-1), 1-10**(-1), 10**(-1), 1-10**(-1))[0]

    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior 
        def exp(R2):
            return joint_pdf(q_v,R2,X,z,beta_v,sigma2_v)
        return  quad(exp, 10**(-1), 1-10**(-1))[0]/norm

    def conditional_pdf(q_v,R2_v):
        # distribution of q conditional on R2, proportionate to the joint posterior
        bz = beta_v @ np.diag(z) @ beta_v.T
        sz_v = sz(z)
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbar(X) * q_v * ((1 - R2_v) / R2_v) * bz)) * R2_v ** (
            A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1) 
        

    # initial values for q and R2
    q_ = grid_q[np.random.random_integers(0,len(grid_q))]
    R_ = grid_R2[np.random.random_integers(0,len(grid_R2))]

    def invCDF(pdf, grid, u):
        weights = [pdf(i) for i in grid]
        normalize_constant = np.sum(weights)
        weights /= normalize_constant
        cdf = np.cumsum(weights)
        return grid[np.argmax(cdf < u)]

    def sampleqR():
        u=np.random.uniform(0,1)
        q_=invCDF(univariate_pdf,grid_q,u)
        def pdf_q(R_v):
            return conditional_pdf(q_,R_v)
        v=np.random.uniform(0,1)
        R_=invCDF(pdf_q,grid_R2,v)
        return (q_,R_)
        
    return sampleqR # function that will be looped over to generate samples of (q, R) given X z



def z(Y, X, R2_v, q_v):
    def pdf(z):
        sz_v = sz(z)
        gamma2_v = gamma2(R2_v, q_v, X)
        Xtilde_v = Xtilde(X, z)
        Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
        Wtildeinv_v = np.linalg.inv(Wtilde_v)
        betahat_v = betahat(Wtildeinv_v, Xtilde_v, Y)
        p = q_v ** sz_v * (1 - q_v) ** (k - sz_v) * (1 / gamma2_v) ** (sz_v / 2) * np.linalg.det(Wtildeinv_v) ** (1 / 2) \
            * (Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) ** (-T / 2)
        return p

    def pdf_exclusion(index, z):
        p = pdf(z)
        z[index] = 0
        p0 = pdf(z)
        z[index] = 1
        p1 = pdf(z)
        return p / (q_v * p1 + (1 - q_v) * p0)

    def iter_gibbs(z):
        for i in range(k):
            p = pdf_exclusion(i, z)
            u = np.random.uniform(0, 1)
            if u > p:
                z[i] = 0
            else:
                z[i] = 1
        return z

    def gibbs(init, iter):
        z = init
        for i in range(iter):
            z = iter_gibbs(z)
        return z

    return gibbs


def sigma2(Y, X, R2_v, q_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    Wtildeinv_v = np.linalg.inv(Wtilde_v)
    betahat_v = betahat(Wtildeinv_v, Xtilde_v, Y)
    # Lorsqu'on regroupera, toute cette initialisation de variables _v ne sera évidemment à faire qu'une fois.
    return invgamma(T / 2, (
            Y.T @ Y - betahat_v.T @ (Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v @ betahat_v) / 2))  # Ytilde=Y


def betatilde(Y, X, R2_v, q_v, sigma2_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z)
    id = np.eye(sz_v)
    mean = np.linalg.inv(id / gamma2_v + Xtilde_v.T @ Xtilde_v) @ Xtilde_v @ Y  # Pas de U*phi
    cov = np.linalg.inv(id / gamma2_v + Xtilde_v.T @ Xtilde_v) * sigma2_v
    return mnormal(mean, cov)


#%% 
