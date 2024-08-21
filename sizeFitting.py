import vars
import fittingFuns
import matplotlib.pyplot as plt
import lmfit
import fittingFuns
import numpy as np
import models
import dataFuns
import scipy as sp

n = len(vars.idealSizes)

#This code largely deals with the sizes in their log form
for i in range(n):
    #Verify later that removing 0.001s is acceptable
    if(vars.idealSizes[i][-1] == vars.minSize):
        continue
    ideal = np.log10(vars.idealSizes[i])
    scanDates = vars.idealScanDates[i]
    
    
    bounds = [fittingFuns.gSizeBounds(ideal, scanDates, vars.params['g'].max*2), (vars.params['d'].min/2, vars.params['d'].max*2), (vars.params['r'].min/2, vars.params['r'].max*2)]
    
    points = fittingFuns.boundsToPoints(bounds)
    hull = sp.spatial.ConvexHull(points)
    
    lhsparams = fittingFuns.inHull(hull, vars.lhs_size)
    bestLHSparams, minDiffs = fittingFuns.bestLHS(fittingFuns.residualSize, lhsparams, scanDates, np.power(10, ideal))
    print(minDiffs)

    bestparams = [bestLHSparams[0][p].value for p in bestLHSparams[0]]
    
    bestLHSparams = fittingFuns.parsToNums(bestLHSparams)
    print(bestLHSparams)
    #for i in range(vars.topLHS):
    #    print([bestLHSparams[i][p].value for p in bestLHSparams[i]])
    #print()
    
    hull = sp.spatial.ConvexHull(bestLHSparams)
    lhsparams = fittingFuns.inHull(hull, vars.lhs_size)
    
    bestLHSparams, minDiffs = fittingFuns.bestLHS(fittingFuns.residualSize, lhsparams, scanDates, np.power(10, ideal))
    print(minDiffs)

    bestparams = [bestLHSparams[0][p].value for p in bestLHSparams[0]]
    for i in range(vars.topLHS):
        print([bestLHSparams[i][p].value for p in bestLHSparams[i]])
    print()
    
    
        
        
        
    bestfit = np.log10(models.gdr(bestparams, scanDates))
    vizData, null = dataFuns.makeVizData(bestLHSparams)
    dataFuns.viz(vizData, bounds, ['g', 'd', 'r'])

    plt.figure(figsize=(10, 6))
    plt.plot(scanDates, ideal, color=(0, 0, 1), label='ideal')
    plt.plot(scanDates, bestfit, color=(1, 0, 0), label='best fit')
    plt.xlabel('days')
    plt.ylabel('pfs')
    plt.title(f"Size {str(i + 1)}")
    plt.legend()
    plt.show()  