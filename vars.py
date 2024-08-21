import lmfit
import numpy as np
import pandas as pd
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

n = 300
lhs_km = 5000
lhs_size = 1000
topLHS = 10
maxSize = 10000
minSize = 0.001
days = 1096
epsilon = 0.2


#Kill it when it goes below e-9
params = lmfit.Parameters()
params.add('g', value=0.03, min=0.01, max=0.06, vary=True)
params.add('gDistr', value=0.15, min=0, max=0.5, vary=False)
params.add('d', value=0.15, min=0.03, max=0.25, vary=True)
params.add('dDistr', value=0.15, min=0, max=0.5, vary=False)
params.add('r', value=0.04, min=0.00001, max=0.05, vary=True)
params.add('rDistr', value=0.15, min=0, max=0.5, vary=False)

parNames = []
parNames.append((0, 'g'))
parNames.append((1, 'gDistr'))
parNames.append((2, 'd'))
parNames.append((3, 'dDistr'))
parNames.append((4, 'r'))
parNames.append((5, 'rDistr'))


medDayScan = 60
dayNoise = 30
numScans = 19

#Creating an array of ideal sizes
df = pd.read_csv("sizes.csv", header=None)
idealSizes = df.to_numpy()
idealSizes = np.where(np.isnan(idealSizes), maxSize + 1, idealSizes)

df = pd.read_csv("scanDates.csv", header=None)
idealScanDates = df.to_numpy()

df = pd.read_csv("kma.csv", header=None)
idealKM = df.to_numpy()
idealKM = np.squeeze(np.asarray(idealKM))
