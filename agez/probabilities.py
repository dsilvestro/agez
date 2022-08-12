import numpy as np
import scipy.stats
np.set_printoptions(suppress=True, precision=3)

# probability functions
def prob_m(m, s, z, eta=0, epsilon=0):
    return np.sum(scipy.stats.norm.logpdf(m+eta, z, s+epsilon))

def prob_z_gamma(z, x, m=np.inf, p=0.99, a=5, b=3):
    if np.max(z) > m:
        prob = -np.inf
    else:
        delta = z - x # negative if z < x
        prob = np.sum(scipy.stats.gamma.logpdf(np.abs(delta[z<x]), a, scale=1/b) + np.log(1 - p))
        prob += np.sum(scipy.stats.gamma.logpdf(delta[z>x], a, scale=1/b) + np.log(p))
    return prob

def prob_z(z, x, m=1000, b=0.1):
    if np.max(z) > m:
        prob = -np.inf
    else:
        delta = z - x
        prob = np.sum(scipy.stats.cauchy.logpdf(delta, scale=b))
    return prob

def prob_z_CauUni(z, x, m=1000, b=0.1):
    "Half Cauchy (positive range), half uniform (negative range)"
    if np.max(z) > m:
        prob = -np.inf
    else:
        delta = z - x # negative if z < x
        prob = np.sum(scipy.stats.cauchy.logpdf(delta[delta > 0], scale=b[delta > 0]))
        prob += np.sum(np.log(1./(x[delta < 0] / 2)))
    return prob


def prob_x_process_prior(x, a=1, b=1):
    xdiff = np.diff(x)
    prior_x0 = scipy.stats.gamma.logpdf(x[-1], a=1, scale=100)
    return np.sum(scipy.stats.gamma.logpdf(xdiff, a=a, scale=1/b)) + prior_x0

def prob_x(x, z_min, a=1, b=1):
    xdiff = z_min - x
    return np.sum(scipy.stats.gamma.logpdf(xdiff, a=a, scale=1/b)) #+ prior_x0

def prob_r(r=None,alpha=1, beta=1):
    return np.sum(scipy.stats.beta.logpdf(r, alpha, beta))

def prob_I(I=None, p_error=0.01):
    return np.log(1 - p_error) * np.sum(I) + np.log(p_error) * len(I[I == 0])
