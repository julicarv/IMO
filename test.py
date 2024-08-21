import vars
import dataFuns
import fittingFuns
import matplotlib.pyplot as plt
import lmfit
import fittingFuns
import numpy as np
import sys

#LHS to find a decent starting fit
#bounds = [(vars.params[p].min, vars.params[p].max) for p in vars.params]
#lhs_params = 

lhs_params = fittingFuns.lhsKM()

bestLHSparams, minDiffs = fittingFuns.bestLHS(fittingFuns.residualKM, lhs_params)

vizData, bounds = dataFuns.makeVizData(bestLHSparams)
print(minDiffs)
print(vizData)
dataFuns.viz(vizData, bounds, ['g', 'd', 'r'])

sys.exit()

#New width on each side is smaller
for i in range(vars.topLHS):
    for p_name in bestLHSparams[i]:
        bestLHSparams[i][p_name].max = 1/5 * vars.params[p_name].max + 4/5 * bestLHSparams[i][p_name].value
        bestLHSparams[i][p_name].min = 1/5 * vars.params[p_name].min + 4/5 * bestLHSparams[i][p_name].value


optimizer_kws = {'method':'least_squares', 'xtol':1e-12, 'ftol':1e-10, 'max_nfev':200, 'verbose':2}
bestParams = []
for i in range(vars.topLHS):
    fit_obj = lmfit.minimize(fittingFuns.residualKM, bestLHSparams[i], **optimizer_kws)
    bestParams.append(fit_obj.params)


#New part of the code
bestKM = [dataFuns.parsToKM(bestParams[i]) for i in range(vars.topLHS)]
kmDiffs = [sum((bestKM[i] - vars.idealKM[i]) ** 2 for i in range(vars.days)) for i in range(vars.topLHS)]

bestKM = bestKM[np.argmin(kmDiffs)]
print(np.min(kmDiffs))

#New part of the code
plt.figure(figsize=(10, 6))

plt.plot(range(vars.days), vars.idealKM, color=(0, 0, 1), label='ideal')
#for i, km in enumerate(bestKM):
#    plt.plot(range(vars.days), km, color=((3*i/(vars.topLHS)%1), i/(vars.topLHS), 0), label='fit ' + str(i))
plt.plot(range(vars.days), bestKM, color=(1, 0, 0), label='best fit')
plt.plot(range(vars.days), dataFuns.parsToKM(bestLHSparams[0]), color=(0, 1, 0), label='best lhs')

plt.xlabel('days')
plt.ylabel('pfs')
plt.legend()

plt.show()
