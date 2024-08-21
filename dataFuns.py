from random import randint
import vars
import numpy as np
import scipy as sp
import models
import sys
from matplotlib import pyplot as plt

def parsToKM(distParams):
    params = np.empty((vars.n, 3))
    params[:, 0] = sp.stats.lognorm.rvs(distParams.get('gDistr'), scale=distParams.get('g'), size=vars.n)
    params[:, 1] = sp.stats.lognorm.rvs(distParams.get('dDistr'), scale=distParams.get('d'), size=vars.n)
    params[:, 2] = sp.stats.lognorm.rvs(distParams.get('rDistr'), scale=distParams.get('r'), size=vars.n)

    scanDates = makeScanDates()
    sizes = models.gdrMany(params, scanDates)
    progs = findProgs(sizes, scanDates)
    return makeKM(progs)

def findProgs(sizes, scanDates):
    output = (vars.days+1)*np.ones(vars.n, dtype=int)
    for i in range(vars.n):
        minI = np.argmin(sizes[i])
        minS = sizes[i][minI]
        for j in range(vars.numScans - minI):
            if(sizes[i][minI + j] > minS*1.72):
                output[i] = scanDates[i][j + minI]
                break
    np.sort(output)
    return output

def makeKM(progs):
    progs = np.sort(progs)
    uProgs, freqs = np.unique(progs, return_counts=True)
    freqList = list(zip(uProgs.astype(int), freqs.astype(int)))
    
    if(uProgs[-1] > vars.days + 1):
        print(uProgs)
        sys.exit()

    counter = 0
    KM = np.ones(vars.days)
    for i in range(len(freqList) - 1):
        counter += freqList[i][1]
        for j in range(freqList[i+1][0] - freqList[i][0]):
            try:
                KM[freqList[i][0] + j] -= counter/vars.n
            except IndexError:
                break
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

def histogram(data, bounds):
    fig, axes = plt.subplots(1, len(data[0]), figsize=(15, 5))

    for i in range(len(data[0])):
        axes[i].hist(data[:, i], bins=10, edgecolor='black', range=(bounds[i][0], bounds[i][1]))
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def scatterplot3(data, bounds):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])    
    ax.set_xlabel('g')
    ax.set_ylabel('d')
    ax.set_xbound(bounds[0][0], bounds[0][1])
    ax.set_ybound(bounds[1][0], bounds[1][1])
    plt.show()

def scatterplot(data, bounds, names, vars):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, vars[0]], data[:, vars[1]])   
    ax.set_xlabel(names[vars[0]])
    ax.set_ylabel(names[vars[1]])
    ax.set_xbound(bounds[vars[0]][0], bounds[vars[0]][1])
    ax.set_ybound(bounds[vars[1]][0], bounds[vars[1]][1])
    plt.show()

def viz(data, bounds, names):
    histogram(data, bounds)
    scatterplot3(data, bounds)
    #scatterplot(data, bounds, names, [0, 1])
    #scatterplot(data, bounds, names, [0, 2])
    #scatterplot(data, bounds, names, [1, 2])

def makeVizData(params):
    vizData = np.zeros((vars.topLHS, len([p for p in vars.params.values() if p.vary])))
    for i in range(vars.topLHS):
        k = 0
        for  p_name in (params[i]):
            if params[i][p_name].vary:
                vizData[i, k] = params[i][p_name].value
                k += 1
    varying_params = [p for p in vars.params.values() if p.vary]
    bounds = [(p.min, p.max) for p in varying_params]
    return vizData, bounds
