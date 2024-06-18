import vars
import models
import dataInterp
import viz
import numpy as np
import scipy as sp
import seaborn as sns

import sys

pars = np.zeros((vars.n, 3))

for i in range(3):
    pars[:, i] = sp.stats.lognorm.rvs(s=vars.pars[i+3], scale=vars.pars[i], size=vars.n)

sizes = models.gdr(pars)

progs = dataInterp.findProgs(sizes)

KM = dataInterp.makeKM(progs)

viz.KMplot(KM)