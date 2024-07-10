import lmfit
import numpy as np
import pandas as pd
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

n = 200
lhs_n = 1000
maxSize = 10000
minSize = 0.001
days = 1096


#Kill it when it goes below e-9
params = lmfit.Parameters()
params.add('g', value=0.02, min=0.01, max=0.06, vary=True)
params.add('gDistr', value=0.15, min=0, max=0.5, vary=False)
params.add('d', value=0.15, min=0.03, max=0.25, vary=True)
params.add('dDistr', value=0.15, min=0, max=0.5, vary=False)
params.add('r', value=0.04, min=0, max=0.05, vary=True)
params.add('rDistr', value=0.15, min=0, max=0.5, vary=False)

medDayScan = 60
dayNoise = 30
scanNoise = 0.01
numScans = 19

#Creating an array of ideal sizes
df = pd.read_csv("sizes.csv", header=None)
idealSizes = df.to_numpy()

df = pd.read_csv("scanDates.csv", header=None)
idealScanDates = df.to_numpy()

df = pd.read_csv("km.csv", header=None)
idealKM = df.to_numpy()
idealKM = np.squeeze(np.asarray(idealKM))
