import numpy as np
import pandas as pd
import copy
from scipy import ndimage
import scipy.stats
np.set_printoptions(suppress=True, precision=3)


def unique_unsorted(a_tmp):
    a = copy.deepcopy(a_tmp)
    indx = np.sort(np.unique(a, return_index=True)[1])
    u = a_tmp[indx]
    return u


class ZirconData():
    def __init__(self,
                 sampleID: [],
                 zirconID: [],
                 stratPosition: [],
                 stratLevel: [],
                 datingMethod: [],
                 age_est: [],
                 age_std: [],
                 rescale_age: float = 0.01
                 ):
        self._sampleID = sampleID
        self._zirconID = [sampleID[i] + "_z%s_" % i + str(zirconID[i]) for i in range(len(sampleID))]
        self._stratLevel = np.array(stratLevel)
        self._stratPosition = np.array(stratPosition)
        self._datingMethod = np.array(datingMethod)
        self._n_dating_methods = len(np.unique(self._datingMethod))
        self._m = np.array(age_est, dtype=float) * rescale_age
        self._s = np.array(age_std, dtype=float) * rescale_age
        self._n_samples = len(np.unique(self._sampleID))
        self._n_zircons = len(self._sampleID)
        self._n_levels = len(np.unique(self._stratLevel))
        # numeric IDs
        self._zirconIDnum = np.arange(self._n_zircons)
        self._sampleIDnum = np.zeros(self._n_zircons).astype(int)
        j = 0
        for i in unique_unsorted(self._stratPosition):
            self._sampleIDnum[self._stratPosition == i] = j
            j += 1
        self._datingMethodIDnum = np.zeros(self._n_zircons).astype(int)
        j = 0
        for i in unique_unsorted(self._datingMethod):
            self._datingMethodIDnum[self._datingMethod == i] = j
            j += 1


class ZirconModel():
    def __init__(self, zrc_data):
        self.data = copy.deepcopy(zrc_data)
        self._indx_sampleIDs_unsorted = zrc_data._sampleIDnum
        self._n_samples = zrc_data._n_samples
        # indicators (0: wrong zircon, 1: correct)
        self._I = np.ones(len(zrc_data._m))
        # z: estimated true age of zircons (abs avoid negative)
        self._z = np.abs(np.random.normal(zrc_data._m, zrc_data._s * 0.0001, zrc_data._n_zircons))
        # d: initialize offset per sample in [0,1] so that x = min_z * d
        self._r = np.random.uniform(0, 1, self._n_samples)
        
        # age of most recent sample
        self._x0 = np.min(self.min_z()) * self._r[0]
        
        # init all ages are ~equal to x0
        self._x = np.ones(self._n_samples) * self._x0

        # init dating method bias
        self._eta = np.random.exponential(0.001, self.data._n_dating_methods)
        self._epsilon = np.random.exponential(0.001, self.data._n_dating_methods)


    # Latent parameters
    def update_x(self):
        for i in range(1, self._n_samples):
            self._x[i] = self._x[i - 1] + np.min(self.min_z()[i:] - self._x[i - 1]) * self._r[i]
    
    def min_z(self, return_all=False):
        if return_all:
            return ndimage.minimum(self._z[self._I == 1], #
                                   labels=self.data._sampleIDnum[self._I == 1],
                                   index=self.data._sampleIDnum)
        else:
            return ndimage.minimum(self._z[self._I == 1], #
                                   labels=self.data._sampleIDnum[self._I == 1],
                                   index=unique_unsorted(self.data._sampleIDnum))

    @property
    def get_x(self):
        self.update_x()
        return self._x
    


# probability functions
def prob_m_s(m, s, z=None, eta=None, epsilon=None):
    mu_i = m + eta
    sig_i = s + epsilon
    return np.sum(scipy.stats.norm.logpdf(z, mu_i, sig_i))

def prob_z(z=None, a=None, b=None, offsets=None):
    # offsets = self.min_z(return_all=True) - self.get_x[self.data._sampleIDnum[self._I == 1]]
    scipy.stats.gamma.logpdf(z, a, scale=b, loc=offsets)
    pass


def prob_r(r=None,alpha=5, beta=1):
    return np.sum(scipy.stats.beta.logpdf(r, alpha, beta))

def prob_I(I=None, p_error=0.01):
    return np.log(1 - p_error) * np.sum(I) + np.log(p_error) * len(I[I == 0])

# def prob_x0(self):
#     return scipy.stats.gamma.logpdf(np.min(self._z) - self._x0,1,scale=1)
# @prop
# def z(self):
#     return self._min_age_z * self._d

def update_uniform(i, d=0.1, n=1, Mb=100, mb= -100):
    Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = np.random.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, (Ix, Iy), hastings


class ZirconAgeSampler():
    def __init__(self, zrc_model):
        self._prm_prime = [copy.deepcopy(i) for i in [zrc_model._z, zrc_model._r, zrc_model._I]]
        self._likelihood = prob_z()
        self._prior = zrc_model.prob_I() + zrc_model.prob_r()
        self._update_f = [0.5, 0.8, 1]

    def mh_step(self):
        rr = np.random.random()
        if rr < self._update_f[0]:
            "update z"
            update = np.random.normal(0, 0.2, zrc_model.data._n_zircons) \
                     * np.random.binomial(1, size=zrc_model.data._n_zircons,p=0.05)
            self._prm_prime[0] = np.abs(self._prm_prime[0] + update)
        elif rr < self._update_f[1]:
            "update r"
            self._prm_prime[1] = UpdateUniform(self._prm_prime[1], d=0.05, Mb=1, mb=0)
        else:
            "update I"
            self._prm_prime[2] = np.abs(self._prm_prime[2] -
                                        np.random.binomial(1, size=zrc_model.data._n_zircons,p=0.05))








tb = pd.read_csv("../Tatacoa Geochron data data.csv")
# sort table by stratigraphic position: first item is the most recent (largest strat position)
tbl = tb.sort_values(by=['Stratigraphic position'], ascending=False)

zrc_data = ZirconData(sampleID=tbl['Sample Label'],
                      zirconID=tbl['Zircon number'],
                      stratLevel=tbl['Stratigraphic level'],
                      stratPosition=tbl['Stratigraphic position'],
                      datingMethod=tbl['Method'],
                      age_est=tbl['Age'],
                      age_std=tbl['Error 2s'])

zrc_model = ZirconModel(zrc_data)

zrc_data._sampleIDnum

print(len(zrc_model.min_z))
print(zrc_model.min_z)

zrc_model.get_x


zrc_sampler = ZirconAgeSampler(zrc_model)


# a = np.array([1,2,2,3,3,3,4,4,4,2,2,2])
# b = np.random.choice(range(5),a.shape)
# ndimage.minimum(a, labels=b, index=unique_unsorted(b))



















