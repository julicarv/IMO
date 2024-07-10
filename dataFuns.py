from random import randint
import vars
import numpy as np
import scipy as sp
import models
import sys

def parsToKM(distParams):
    params = np.empty((vars.n, 3))
    params[:, 0] = sp.stats.lognorm.rvs(distParams.get('gDistr'), scale=distParams.get('g'), size=vars.n)
    params[:, 1] = sp.stats.lognorm.rvs(distParams.get('dDistr'), scale=distParams.get('d'), size=vars.n)
    params[:, 2] = sp.stats.lognorm.rvs(distParams.get('rDistr'), scale=distParams.get('r'), size=vars.n)

    scanDates = makeScanDates()
    sizes = models.gdr(params, scanDates)
    progs = findProgs(sizes, scanDates)
    return makeKM(progs)

def findProgs(sizes, scanDates):
    output = np.empty(vars.n, dtype=int)
    for i in range(vars.n):
        minI = np.argmin(sizes[i])
        min = sizes[i, minI]
        for j in range(vars.numScans - minI):
            if(sizes[i, minI + j] > min*1.72):
                output[i] = scanDates[i, j + minI]
                break
    np.sort(output)
    return output

def makeKM(progs):
    progs = np.sort(progs)

    uProgs, freqs = np.unique(progs, return_counts=True)
    freqList = list(zip(uProgs.astype(int), freqs.astype(int)))

    counter = 0
    KM = np.ones(vars.days)
    for i in range(len(freqList) - 1):
        counter += freqList[i][1]
        for j in range(freqList[i+1][0] - freqList[i][0]):
            try:
                KM[freqList[i][0] + j] -= counter/vars.n
            except IndexError:
                print(uProgs)
                sys.exit()
    for i in range(vars.days - freqList[-1][0]):
        KM[-1 - i] = 0
    return KM

#Generates scan dates for the KM curve
def makeScanDates():
    output = np.zeros((vars.n, vars.numScans))
    for i in range(vars.n):
        for j in range(vars.numScans - 2):
            output[i, j+1] = vars.medDayScan*(j+1) + randint(-30, 30)
        output[i, vars.numScans - 1] = vars.medDayScan*(vars.numScans-1) + randint(-min(30, vars.days-1-vars.medDayScan*(vars.numScans-1)), min(30, vars.days-1-vars.medDayScan*(vars.numScans-1)))
    return output


#Makes the sizes a step function with noise, still var.days long. No longer useful due to making the sizes matrix smaller
#def transform(sizes, scanMean=vars.medDayScan, scanStd=vars.scanNoise, dayStd=vars.dayNoise):
#   for i in range(vars.n):
#       k = round(2*vars.days/scanMean)
#       unroundedSD = np.random.uniform(low=-15+scanMean, high=15+scanMean, size=k)
#       scanDists = np.empty((k,), dtype=int)
#       for j in range(len(unroundedSD)):
#           scanDists[j] = round(unroundedSD[j])
#       scanDays = np.empty(k, dtype=int)
#       scanDays[0] = 0
#       for j in range(k-1):
#           scanDays[j+1] = scanDays[j] + scanDists[j]
#       for j in range(k):
#           if(scanDays[j] >= vars.days):
#               break
#           sizes[i, scanDays[j]] *= sp.stats.truncnorm(-2, 2, scale=scanStd).rvs()
#           for k in range (min(scanDays[j+1]-scanDays[j]-1, vars.days-scanDays[j]-1)):
#               sizes[i, scanDays[j]+k+1] = sizes[i, scanDays[j]]
#   return sizes
