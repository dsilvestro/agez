import numpy as np
import pandas as pd
import copy
from scipy import ndimage
import scipy.stats
np.set_printoptions(suppress=True, precision=3)



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





















