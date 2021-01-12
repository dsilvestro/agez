import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import agez as zr

rseed = 5555 # np.random.randint(1000,9999)
np.random.seed(rseed)

n_iterations = 10000000
sampling_freq = 5000
print_freq = 5000
tb = pd.read_csv("../Tatacoa Geochron data data.csv")
# sort table by stratigraphic position: first item is the most recent (largest strat position)
tbl = tb.sort_values(by=['Stratigraphic position'], ascending=False)

zrc_data = zr.ZirconData(sampleID=tbl['Sample Label'],
                      zirconID=tbl['Zircon number'],
                      stratLevel=tbl['Stratigraphic level'],
                      stratPosition=tbl['Stratigraphic position'],
                      datingMethod=tbl['Method'],
                      age_est=tbl['Age'],
                      age_std=tbl['Error 2s'])

zrc_model = zr.ZirconModel(zrc_data)
zrc_sampler = zr.AgezSampler(zrc_model)

zrc_logger = zr.AgezLogger(filename="run%sT01" % rseed)

# run MCMC
zr.runMCMC(zrc_sampler,
           zrc_logger,
           n_iterations=n_iterations,
           sampling_freq=sampling_freq,
           print_freq=print_freq)

























