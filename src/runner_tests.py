import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)

from agez import zrc_model
from agez import files
rseed =1432
np.random.seed(rseed)

tb = pd.read_csv("../Tatacoa Geochron data data.csv")
# sort table by stratigraphic position: first item is the most recent (largest strat position)
tbl = tb.sort_values(by=['Stratigraphic position'], ascending=False)

zrc_data = zrc_model.ZirconData(sampleID=tbl['Sample Label'],
                                zirconID=tbl['Zircon number'],
                                stratLevel=tbl['Stratigraphic level'],
                                stratPosition=tbl['Stratigraphic position'],
                                datingMethod=tbl['Method'],
                                age_est=tbl['Age'],
                                age_std=tbl['Error 2s'])

zrc_model = zrc_model.ZirconModel(zrc_data)

zrc_model.get_x




def plot_dat_hist(i=-1, m=100):
    import matplotlib.pyplot as plt
    if i == -1:
        x = zrc_data._m[zrc_data._m < m]
    else:
        x = zrc_data._m[zrc_data._sampleIDnum == i]
        x = x[x < m]
    n, bins, patches = plt.hist((x), density=False, facecolor='g', alpha=0.75)
    plt.show()


plot_dat(i=12)


print(zrc_model.min_z())
print(zrc_model.get_x)

zrc_sampler = zrc_model.ZirconAgeSampler(zrc_model)
zrc_logger = files.AgezLogger(filename="run%s" % rseed)
zrc_logger.init_header(samples=zrc_data._sampleID, dating_methods=zrc_data._datingMethod)


# zrc_sampler.mh_step()

for i in range(10000000):
    zrc_sampler.mh_step()
    if i % 5000 == 0:
        print(zrc_sampler.counter-1, zrc_sampler._post, np.sum(zrc_sampler._zrc_model._I),
              np.max(zrc_sampler._zrc_model._x), np.min(zrc_sampler._zrc_model._x))
        # zrc_sampler._zrc_model.get_x
        zrc_logger.log_sample(zrc_sampler)

# for i in range(zrc_data._n_samples):
#     print(zrc_sampler._zrc_model._I[zrc_data._sampleIDnum == i])
#



print(zrc_sampler._zrc_model._b_rate)

print(zrc_sampler._zrc_model._x)
zrc_sampler._zrc_model.min_z()



# a = np.array([1,2,2,3,3,3,4,4,4,2,2,2])
# b = np.random.choice(range(5),a.shape)
# ndimage.minimum(a, labels=b, index=unique_unsorted(b))



# test I updater
zrc_model = zrc_model.ZirconModel(zrc_data)

s = np.random.choice(np.arange(zrc_model.data._n_samples))
s_indx = np.where(zrc_model.data._sampleIDnum == s)
print(zrc_model._I[s_indx])
Is = zrc_model._I[s_indx]
Ages_Is = zrc_model._z[s_indx]





add_zero = np.random.randint(2)
if add_zero:
    indx = np.max([0, len(np.where(Is == 0)[0])-1])
    print("select the oldest 0 (if any)", indx)
    Is[indx] = 0
else:
    indx = np.max([0, len(np.where(Is == 1)[0])-1])
    print("remove a zero (if any)", indx)
    Is[indx] = 1

print(Is)

zrc_model._I[s_indx] = Is

#
print(zrc_model._I[s_indx])



s = 9
s_indx = np.where(zrc_model.data._sampleIDnum == s)
zrc_model._I[s_indx]
zrc_model._z[s_indx]
zrc_model.min_z()[s]

i_temp = zrc_model._I[s_indx]
i_temp[6] = 0

zrc_model._I[s_indx] = i_temp



s = 4
s_indx = np.where(zrc_model.data._sampleIDnum == s)
zrc_sampler._zrc_model._z[s_indx]
zrc_sampler._zrc_model.data._m[s_indx]

zrc_sampler._zrc_model.min_z()[s]

zrc_sampler._zrc_model.data._m[s_indx]




import numpy as np
import agez as zr
z = np.array([9, 12, 13, 18, 100.])
x = np.ones(len(z)) * 8
b = 2
print(zr.prob_z_CauUni(z, x, b=b))

print(zr.prob_z(z, x, b=b))









