import numpy as np
import copy
from scipy import ndimage
import scipy.stats
np.set_printoptions(suppress=True, precision=3)
from agez import utilities
from agez import probabilities
small_number = 10e-10

class ZirconData:
    def __init__(self,
                 sampleID: [],
                 zirconID: [],
                 stratPosition: [],
                 stratLevel: [],
                 datingMethod: [],
                 age_est: [],
                 age_std: [],
                 rescale_age: float = 1,
                 rescale_s: float = 0.5
                 ):
        self._sampleID = np.array(sampleID)
        self._zirconID = [sampleID[i] + "_z%s_" % i + str(zirconID[i]) for i in range(len(sampleID))]
        self._stratLevel = np.array(stratLevel)
        self._stratPosition = np.array(stratPosition)
        self._datingMethod = np.array(datingMethod)
        self._n_dating_methods = len(np.unique(self._datingMethod))
        self._m = np.array(age_est, dtype=float) * rescale_age
        self._s = np.array(age_std, dtype=float) * rescale_s * rescale_age
        self._n_samples = len(np.unique(self._sampleID))
        self._n_zircons = len(self._sampleID)
        self._n_levels = len(np.unique(self._stratLevel))
        # numeric IDs
        self._zirconIDnum = np.arange(self._n_zircons)
        self._sampleIDnum = np.zeros(self._n_zircons).astype(int)
        j = 0
        for i in utilities.unique_unsorted(self._stratPosition):
            self._sampleIDnum[self._stratPosition == i] = j
            j += 1
        self._datingMethodIDnum = np.zeros(self._n_zircons).astype(int)
        j = 0
        for i in utilities.unique_unsorted(self._datingMethod):
            self._datingMethodIDnum[self._datingMethod == i] = j
            j += 1


class ZirconModel:
    def __init__(self, zrc_data):
        self.data = copy.deepcopy(zrc_data)
        self._n_samples = zrc_data._n_samples
        # init parameters
        self.init_Irz()
        # init dating method bias
        self._eta = np.zeros(self.data._n_dating_methods)
        self._epsilon = np.random.exponential(0.5, self.data._n_dating_methods)
        self._x_rate = 1
        self._x_mean = 1
        self._m_scale = np.ones(self._n_samples)
        self._b_alpha = 5
        self._b_beta = 1
        self._G_alpha = 1
        self._G_beta = 1
        self._truncatePrior = 0.1
        self.update_x_func = self.update_x
        self._unique_unsorted_sampleIDnum = utilities.unique_unsorted(self.data._sampleIDnum)
        self.estimate_truncation = 0
        # age of most recent sample
        self._x0 = np.min(self.min_z()) * self._r[0]

        # init all ages are ~equal to x0
        self._x = np.ones(self._n_samples) * self._x0

        self._max_age_boundary = np.max(self.data._m) * 1.1
        self._dating_methods = utilities.unique_unsorted(self.data._datingMethod)
        # fix epsilon value for max_age data:
        self._epsilon[self._dating_methods == 'fixed_age'] = 0.1

    def init_Irz(self):
        # indicators (0: wrong zircon, 1: correct)
        self._I = np.ones(len(self.data._m))
        # set indicator incompatible with fixed ages to 0
        self.min_ages = np.zeros(len(self.data._m))
        for i in range(len(self.data._m)):
            if self.data._datingMethod[i] == 'fixed_age':
                if self.data._m[i] > self.min_ages[i-1]:
                    self.min_ages[i:] =  self.data._m[i] + 0
            
            tmp_I = self._I[i:] + 0
            tmp_m = self.data._m[i:] + 0
            tmp_I[tmp_m < self.min_ages[i-1]] = 0 # label as incorrect
            self._I[i:] = tmp_I
        self._fixed_I = self._I + 0

        # z: estimated true age of zircons (abs avoid negative)
        rr = np.random.normal(0, self.data._s*0.1, self.data._n_zircons)
        self._z = np.abs(self.data._m + rr)
        # fix ages of unconformities and 'fixed_age' layers
        self._fixed_zr_ind = np.where(self.data._datingMethod == 'fixed_age')[0]
        self._z[self._fixed_zr_ind] = self.data._m[self._fixed_zr_ind]
        # fix other ages to be older than min ages
        ind = self._z < self.min_ages
        self._z[ind] = self.data._m[ind] + np.abs(rr[ind])

        # r: initialize offset per sample in [0,1] so that x = min_z * r
        dating_method_per_sample = np.zeros(self._n_samples).astype(str)
        for i in range(self._n_samples):
            tmp = self.data._datingMethod[self.data._sampleIDnum == i]
            dating_method_per_sample[i] = tmp[0]
        
        self._fixed_sample_ind = np.where(dating_method_per_sample == 'fixed_age')[0]
        self._r = np.random.uniform(0.9, 1, self._n_samples)
        self._r[self._fixed_sample_ind] = 1 - small_number
        
        
        
    # Latent parameters
    def update_x(self):
        tmp = self.min_z()
        self._x[0] = np.min(tmp) * self._r[0]
        for i in range(1, self._n_samples):
            # print(i, self._r[i] )
            self._x[i] = self._x[i - 1] + (np.min(tmp[i:] - self._x[i - 1]) * self._r[i]) #- small_number
            # print(i, self._r[i] ,  self._x[i - 1],  self._x[i], self._x[i - 1]- self._x[i])
    
    def update_x_rev(self):
        rev_order = np.arange(self._n_samples)[::-1]
        x_tmp = self._x * 0 + self._x[-1]
        x_tmp[-1] = self.min_z()[-1] - self._r[-1]
        for i in rev_order[1:]:
            # print(np.min(self._x[i:-1]), np.min(self.min_z()[i:-1]))
            x_tmp[i] = np.min([np.min(x_tmp[i:-1]), np.min(self.min_z()[i:-1])]) * self._r[i]
        self._x = x_tmp + 0

    def update_x_MAX(self):
        x_max = np.zeros(self._n_samples)
        for i in range(self._n_samples):
            x_max[i] = np.min(self.min_z()[i:])
        self._x = x_max + 0

    def min_z(self, return_all=False):
        z_temp = self._z + 0
        z_temp[self._I == 0] = np.max(self.data._m) # arbitrarily large
        if return_all:
            return ndimage.minimum(z_temp, #
                                   labels=self.data._sampleIDnum,
                                   index=self.data._sampleIDnum)
        else:
            return ndimage.minimum(z_temp, #
                                   labels=self.data._sampleIDnum,
                                   index=self._unique_unsorted_sampleIDnum)

    @property
    def get_x_max(self):
        x_max = np.zeros(self._n_samples)
        for i in range(self._n_samples):
            x_max[i] = np.min(self.min_z()[i:])
        
        return x_max

    
    @property
    def get_x(self):
        self.update_x_func()
        return self._x

    @property
    def calc_likelihood(self):
        return probabilities.prob_m(m=self.data._m,
                                    s=self.data._s,
                                    z=self._z,
                                    eta=self._eta[self.data._datingMethodIDnum],
                                    epsilon=self._epsilon[self.data._datingMethodIDnum])
    @property
    def calc_prior(self):
        if self.estimate_truncation:
            trunc = self._truncatePrior + self._G_alpha
        else:
            trunc = self._truncatePrior
        prior = probabilities.prob_z_CauUni(z=self._z,
                                     m=self._max_age_boundary,
                                     x=self.get_x[self.data._sampleIDnum],
                                     b=self._m_scale[self.data._sampleIDnum] + trunc)
 
        prior += probabilities.prob_I(self._I)

        prior += probabilities.prob_r(self._r, alpha=self._b_alpha, beta=self._b_beta)
        
        # prior += utilities.prob_x(self.get_x, self.min_z(),
        #                           a=self._x_mean, #*self._x_rate,
        #                           b=self._x_rate)
   
        # prior += utilities.prob_x_process_prior(self.get_x,
        #                                         a=self._x_mean*self._x_rate,
        #                                         b=self._x_rate)
        
        return prior


    @property
    def calc_hyperpriors(self):
        hp = scipy.stats.gamma.logpdf(self._b_alpha, a=1, scale=10)
        hp += scipy.stats.gamma.logpdf(self._b_beta, a=1, scale=10)
        hp += scipy.stats.gamma.logpdf(self._G_alpha, a=10, scale=0.5)
        hp += scipy.stats.gamma.logpdf(self._G_beta, a=10, scale=0.5)
        hp += np.sum(scipy.stats.cauchy.logpdf(self._m_scale, 0, self._G_beta)) # * self._n_samples
        #hp += np.sum(scipy.stats.gamma.logpdf(self._m_scale, self._G_alpha, scale=1/self._G_beta))
        hp += np.sum(scipy.stats.norm.logpdf(self._eta, scale=5))
        hp += np.sum(scipy.stats.gamma.logpdf(self._epsilon, a=1, scale=10))
        return hp
    
    @property
    def error_rate_per_sample(self):
        return 1 - ndimage.mean(self._I,
                                labels=self.data._sampleIDnum,
                                index=self._unique_unsorted_sampleIDnum)


class AgezSampler:
    def __init__(self, zrc_model, update_f=None):
        self._zrc_model = copy.deepcopy(zrc_model)
        self._zrc_model.update_x_func()
        self._zrc_model_prime = copy.deepcopy(zrc_model)
        if update_f is None:
            update_f = np.array([0.9, 0.4, 0.3, 0.02, 0.1])
        self._update_f = update_f/np.sum(update_f)
        self._likelihood = self._zrc_model.calc_likelihood
        self._prior = self._zrc_model.calc_prior + self._zrc_model.calc_hyperpriors
        self._post = self._likelihood + self._prior
        self.updates = []
        self.counter = 0
        self._accepted_moves = list()
        # update frequency of indicators |-> is datingMethod=='fixed_age' no error is allowed
        p_ind_update = 1 / self._zrc_model_prime.data._m * (zrc_model.data._datingMethod != 'fixed_age')
        # # for zircons incompatible with samples with incompatible 'fixed_age' I = 0 and can't be changed
        # p_ind_update = p_ind_update * self._zrc_model._fixed_I

        self.p_ind_update = p_ind_update / np.sum(p_ind_update)

    def reset_prime_state(self):
        if 0 in self.updates:
            self._zrc_model_prime._z = self._zrc_model._z + 0
        if 1 in self.updates:
            self._zrc_model_prime._r = self._zrc_model._r + 0
            self._zrc_model_prime._G_alpha = self._zrc_model._G_alpha
            self._zrc_model_prime._G_beta = self._zrc_model._G_beta
        if 3 in self.updates:
            self._zrc_model_prime._b_alpha = self._zrc_model._b_alpha + 0
            self._zrc_model_prime._b_beta = self._zrc_model._b_beta + 0
            self._zrc_model_prime._m_scale = self._zrc_model._m_scale + 0
        if 4 in self.updates:
            self._zrc_model_prime._eta = self._zrc_model._eta + 0
            self._zrc_model_prime._epsilon = self._zrc_model._epsilon + 0
        # I is (potentially) always updated
        self._zrc_model_prime._I = self._zrc_model._I + 0

    def reset_accepted_state(self):
        if 0 in self.updates:
            self._zrc_model._z = self._zrc_model_prime._z + 0
        if 1 in self.updates:
            self._zrc_model._r = self._zrc_model_prime._r + 0
            self._zrc_model._G_alpha = self._zrc_model_prime._G_alpha
            self._zrc_model._G_beta = self._zrc_model_prime._G_beta
        if 3 in self.updates:
            self._zrc_model._b_alpha = self._zrc_model_prime._b_alpha + 0
            self._zrc_model._b_beta = self._zrc_model_prime._b_beta + 0
            self._zrc_model._m_scale = self._zrc_model_prime._m_scale + 0
        if 4 in self.updates:
            self._zrc_model._eta = self._zrc_model_prime._eta + 0
            self._zrc_model._epsilon = self._zrc_model_prime._epsilon + 0
        # I is (potentially) always updated
        self._zrc_model._I = self._zrc_model_prime._I + 0

    def mh_step(self):
        hastings = 0
        rr = np.random.choice(np.arange(len(self._update_f)), p=self._update_f,size=1)
        if 0 in rr:
            "update z: NOTE that if data._s == 0, then the zircon age will not be updated"
            update = np.random.normal(0, self._zrc_model.data._s*0.5, self._zrc_model.data._n_zircons) \
                     * np.random.binomial(1, size=self._zrc_model.data._n_zircons,p=0.01)
            self._zrc_model_prime._z = np.abs(self._zrc_model_prime._z + update)

            # "fix other ages to be older than min ages by setting their indicator to 0"
            # ind = self._zrc_model_prime._z < self._zrc_model.min_ages
            # self._zrc_model_prime._I[ind] = 0
            # diff = np.abs(self._zrc_model.min_ages[ind] - self._zrc_model_prime._z[ind])
            # self._zrc_model_prime._z[ind] = self._zrc_model.min_ages[ind] + diff

            # self._zrc_model_prime.update_x()
            # print(self._zrc_model_prime.calc_likelihood, self._likelihood)
            # print(self._zrc_model_prime.calc_prior + self._zrc_model_prime.calc_hyperpriors, self._prior)
        if 1 in rr:
            "update r"
            self._zrc_model_prime._r = utilities.update_uniform(self._zrc_model_prime._r,
                                                                d=0.1, Mb=1, mb=0, n=5)
            self._zrc_model_prime._r[self._zrc_model._fixed_sample_ind] = 1-small_number # reset fixed_age samples
            "update G_rate"
            if self._zrc_model.estimate_truncation:
                self._zrc_model_prime._G_alpha, h = utilities.multiplier_proposal(self._zrc_model_prime._G_alpha,d=1.05)
                hastings += h
            self._zrc_model_prime._G_beta, h = utilities.multiplier_proposal(self._zrc_model_prime._G_beta,d=1.05)
            hastings += h

        if 2 in rr:
            "update I"
            indx = np.random.choice(range(self._zrc_model_prime.data._n_zircons), 1,  p=self.p_ind_update)
            self._zrc_model_prime._I[indx] = np.abs(1 - self._zrc_model_prime._I[indx])
            # self._zrc_model_prime._I = self._zrc_model_prime._I * self._zrc_model._fixed_I # reset incompatible samples

        if 3 in rr:
            "update b_rate"
            self._zrc_model_prime._b_alpha, h = utilities.multiplier_proposal(self._zrc_model_prime._b_alpha,d=1.1)
            hastings += h
            self._zrc_model_prime._b_beta, h = utilities.multiplier_proposal(self._zrc_model_prime._b_beta,d=1.1)
            hastings += h
            self._zrc_model_prime._m_scale, h = utilities.multiplier_proposal_vector(self._zrc_model_prime._m_scale,d=1.2)
            hastings += h

 
        if 4 in rr:
            "update method biases"
            #self._zrc_model_prime._eta = utilities.update_uniform(self._zrc_model_prime._eta,d=0.25)
            ind = self._zrc_model._dating_methods != 'fixed_age'
            tmp, h = utilities.multiplier_proposal_vector(self._zrc_model_prime._epsilon[ind],d=1.1)
            self._zrc_model_prime._epsilon[ind] = tmp
            hastings += h

        "fix ages that are younger than min constraints by setting their indicator to 0"
        ind = self._zrc_model_prime._z < self._zrc_model.min_ages
        self._zrc_model_prime._I[ind] = 0

        self.updates = rr

    
        # acceptance condition
        lik = self._zrc_model_prime.calc_likelihood
        prior = self._zrc_model_prime.calc_prior + self._zrc_model_prime.calc_hyperpriors
        post_prime = lik + prior

        if np.isfinite(post_prime):
            pass
        else:
            print(self._zrc_model_prime.calc_prior,
                  self._zrc_model_prime.calc_hyperpriors,
                  self._zrc_model.min_z()-self._zrc_model.get_x)
            print(self._zrc_model_prime.min_z()-self._zrc_model_prime.get_x)


        if post_prime - self._post + hastings >= np.log(np.random.random()):
            self._likelihood = lik
            self._prior = prior
            self._post = post_prime
            self.reset_accepted_state()
            self._accepted_moves.append([rr[0], 1])
        else:
            self.reset_prime_state()
            self._accepted_moves.append([rr[0], 0])
        self.updates = []
        self.counter += 1
        if len(self._accepted_moves) >= 100:
            self._accepted_moves = self._accepted_moves[-len(self._accepted_moves):]

# a = np.array([1,2,2,3,3,3,4,4,4,2,2,2])
# b = np.random.choice(range(5),a.shape)
# ndimage.minimum(a, labels=b, index=unique_unsorted(b))



















