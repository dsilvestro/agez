import numpy as np
import copy
import scipy.stats
from scipy import ndimage
np.set_printoptions(suppress=True, precision=3)


def choose(lst, p):
    # faster version of np.choose()
    return lst[np.searchsorted(np.cumsum(p),np.random.random())]


def unique_unsorted(a_tmp):
    a = copy.deepcopy(a_tmp)
    indx = np.sort(np.unique(a, return_index=True)[1])
    u = a_tmp[indx]
    return u


def update_uniform(i, d=0.1, n=1, Mb=100, mb= -100):
    Ix = np.random.randint(0, i.shape,n) # faster than np.random.choice
    z = np.zeros(i.shape) + i
    z[Ix] = z[Ix] + np.random.uniform(-d, d, n)
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    return z


def update_uniform2D(i, d=0.1, n=1, Mb=100, mb= -100):
    Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = np.random.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    hastings = 0
    return z, (Ix, Iy), hastings


def multiplier_proposal_vector(q, d=1.05, f=0.25):
    S = np.shape(q)
    ff = np.random.binomial(1,f,S)
    u = np.random.uniform(0,1,S)
    l = 2 * np.log(d)
    m = np.exp(l * (u - .5))
    m[ff==0] = 1.
    new_q = q * m
    U=np.sum(np.log(m))
    return new_q,U


def multiplier_proposal(i, d=1.05):
    z = i + 0
    u = np.random.random()
    l = 2 * np.log(d)
    m = np.exp(l * (u - .5))
    z = z * m
    U = np.log(m)
    return z, U



def plot_ages():
    pass


def plot_dat(i, m=100):
    import matplotlib.pyplot as plt
    x = zrc_data._m[zrc_data._sampleIDnum==i]
    x = zrc_data._m[zrc_data._m < m]
    n, bins, patches = plt.hist((x), density=True, facecolor='g', alpha=0.75)
    plt.show()


def runMCMC(zrc_sampler,
            zrc_logger,
            n_iterations=10000,
            sampling_freq=100,
            print_freq=100):
    zrc_logger.init_header(samples=zrc_sampler._zrc_model.data._sampleID,
                           dating_methods=zrc_sampler._zrc_model.data._datingMethod)
    for i in range(n_iterations):
        zrc_sampler.mh_step()
        if i % sampling_freq == 0:
            zrc_logger.log_sample(zrc_sampler)
        if i % print_freq == 0:
            print(zrc_sampler.counter - 1,
                  np.round(zrc_sampler._post,2),
                  int(np.sum(zrc_sampler._zrc_model._I)),
                  np.round(np.max(zrc_sampler._zrc_model._x),2),
                  np.round(np.min(zrc_sampler._zrc_model._x),2))
            tmp = np.array(zrc_sampler._accepted_moves)
            print("Acc. prob.:", ndimage.mean(tmp[:, 1], labels=tmp[:, 0], index=np.unique(tmp[:, 0])))
                  # zrc_sampler._zrc_model._G_alpha, zrc_sampler._zrc_model._G_beta,
                  # zrc_sampler._zrc_model_prime._G_alpha, zrc_sampler._zrc_model_prime._G_beta)


