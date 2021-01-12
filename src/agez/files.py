import csv
import os
import numpy as np
import pickle
np.set_printoptions(suppress=True, precision=3)
from agez import utilities
from agez import zrc_model

class postLogger():
    def __init__(self,
                 filename="ouput",
                 wdir="",
                 header=None):

        self.filename = filename
        self.wdir = wdir
        if header is None:
            self.header = ["it", "posterior", "likelihood", "prior"]

    def init_log_file(self):
        self.logfile = open(os.path.join(self.wdir, self.filename + ".log"), "w")
        self.wlog = csv.writer(self.logfile, delimiter='\t')
        self.log_line(self.header)

    def log_line(self, line):
        self.wlog.writerow(line)
        self.logfile.flush()

    def load_pkl(file_name):
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except:
            import pickle5
            with open(file_name, 'rb') as f:
                return pickle5.load(f)

    def save_pkl(self, obj):
        with open(os.path.join(self.wdir, self.filename + ".pkl"), 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class AgezLogger(postLogger):
    def init_header(self,samples=None, dating_methods=None):
        self.header = ["it", "posterior", "likelihood", "prior", "mean_z", "delta_z",
                       "b_alpha", "b_beta", "G_alpha", "G_beta", "I"]
        self.samples=samples
        self.dating_methods=dating_methods
        # super().__init__(self.header)

        if samples is not None:
            for i in utilities.unique_unsorted(samples):
                self.header.append("x_%s" % i)
            for i in utilities.unique_unsorted(samples):
                self.header.append("s_%s" % i) # Cauchy scales
            for i in utilities.unique_unsorted(samples):
                self.header.append("z_min_%s" % i)
            for i in utilities.unique_unsorted(samples):
                self.header.append("r_%s" % i)
            for i in utilities.unique_unsorted(samples):
                self.header.append("pI_%s" % i) # fraction of erroneous zircons

        if dating_methods is not None:
            for i in utilities.unique_unsorted(dating_methods):
                self.header.append("eta_%s" % i)
            for i in utilities.unique_unsorted(dating_methods):
                self.header.append("epsilon_%s" % i)

        self.init_log_file()

    def log_sample(self, zrc_sampler: zrc_model.AgezSampler):
        line = [zrc_sampler.counter, zrc_sampler._post,
                zrc_sampler._likelihood, zrc_sampler._prior,
                np.mean(zrc_sampler._zrc_model._z),
                np.mean(zrc_sampler._zrc_model._z-zrc_sampler._zrc_model.data._m),
                zrc_sampler._zrc_model._b_alpha,
                zrc_sampler._zrc_model._b_beta,
                zrc_sampler._zrc_model._G_alpha,
                zrc_sampler._zrc_model._G_beta,
                np.sum(1 - zrc_sampler._zrc_model._I)]
        if self.samples is not None:
            line = line + list(zrc_sampler._zrc_model.get_x)
            line = line + list(zrc_sampler._zrc_model._m_scale)
            line = line + list(zrc_sampler._zrc_model.min_z())
            line = line + list(zrc_sampler._zrc_model._r)
            line = line + list(zrc_sampler._zrc_model.error_rate_per_sample)
        if self.dating_methods is not None:
            line = line + list(zrc_sampler._zrc_model._eta) + list(zrc_sampler._zrc_model._epsilon)

        self.log_line(line)
        self.save_pkl(zrc_sampler)
































