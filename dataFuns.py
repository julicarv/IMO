import vars
import numpy as np
import scipy as sp
import models

def parsToKM(params):
    sizes = models.gdr(params, True)
    sizes = transform(sizes)
    progs = findProgs(sizes)
    return makeKM(progs)

def findProgs(sizes):
    output = np.empty(vars.n, dtype=int)
    for i in range(vars.n):
        minI = np.argmin(sizes[i])
        min = sizes[i, minI]
        for j in range(vars.days - minI):
            if(sizes[i, minI + j] > min*1.72):
                output[i] = j + minI
                break
    np.sort(output)
    return output

def makeKM(progs):
    progs = np.sort(progs)
    uProgs, freqs = np.unique(progs, return_counts=True)
    freqs = freqs.astype(int)
    uProgs = uProgs.astype(int)
    freqList = list(zip(uProgs, freqs))

    counter = 0
    KM = np.ones(vars.days)
    for i in range(len(freqList) - 1):
        counter += freqList[i][1]
        for j in range(freqList[i+1][0] - freqList[i][0]):
            KM[freqList[i][0] + j] -= counter/vars.n
    for i in range(vars.days - freqList[-1][0]):
        KM[-1 - i] = 0
    return KM

#Makes the sizes a step function with noise, still var.days long
def transform(sizes, scanMean=vars.medDayScan, scanStd=vars.scanNoise, dayStd=vars.dayNoise):
    for i in range(vars.n):
        k = round(2*vars.days/scanMean)
        unroundedSD = np.random.uniform(low=-15+scanMean, high=15+scanMean, size=k)
        scanDists = np.empty((k,), dtype=int)
        for j in range(len(unroundedSD)):
            scanDists[j] = round(unroundedSD[j])
        scanDays = np.empty(k, dtype=int)
        scanDays[0] = 0
        for j in range(k-1):
            scanDays[j+1] = scanDays[j] + scanDists[j]
        for j in range(k):
            if(scanDays[j] >= vars.days):
                break
            sizes[i, scanDays[j]] *= sp.stats.truncnorm(-2, 2, scale=scanStd).rvs()
            for k in range (min(scanDays[j+1]-scanDays[j]-1, vars.days-scanDays[j]-1)):
                sizes[i, scanDays[j]+k+1] = sizes[i, scanDays[j]]
    return sizes