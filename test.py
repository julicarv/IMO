import vars
import dataFuns
import fittingFuns
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import lmfit
import fittingFuns

#LHS to find a decent starting fit
lhs_params = fittingFuns.lhs(vars.lhs_n)

minDiff = 999999
bestLHSparams = vars.params
for param_set in lhs_params:
    if fittingFuns.residual(param_set, vars.n) < minDiff:
        bestLHSparams = param_set
        minDiff = fittingFuns.residual(param_set, vars.n)

#New width on each side is 1/5 what it was before
for p_name in bestLHSparams:
    bestLHSparams[p_name].max = 1/3 * vars.params[p_name].max + 2/3 * vars.params[p_name].value
    bestLHSparams[p_name].min = 1/3 * vars.params[p_name].min + 2/3 * vars.params[p_name].value
print(bestLHSparams)


optimiser_kws = {'method':'least_squares', 'xtol':1e-10, 'ftol':1e-10, 'max_nfev':200, 'verbose':2}
fit_obj = lmfit.minimize(fittingFuns.residual, bestLHSparams, args=(vars.n,), **optimiser_kws)
bestParams = fit_obj.params


#New part of the code
bestKMparams = np.empty([vars.n, 3])

bestKMparams[:, 0] = sp.stats.lognorm.rvs(bestParams.get('gDistr'), scale=np.exp(bestParams.get('g')), size=vars.n)
bestKMparams[:, 1] = sp.stats.lognorm.rvs(bestParams.get('dDistr'), scale=np.exp(bestParams.get('d')), size=vars.n)
bestKMparams[:, 2] = sp.stats.lognorm.rvs(bestParams.get('rDistr'), scale=np.exp(bestParams.get('r')), size=vars.n)

bestKM = dataFuns.parsToKM(bestKMparams)
#lhsKM = dataFuns.parsToKM(bestLHSparams)

print(bestParams)


#New part of the code
plt.figure(figsize=(10, 6))

plt.plot(range(vars.days), bestKM, color='blue', label='fit')
plt.plot(range(vars.days), vars.ideal, color='red', label='ideal')
#plt.plot(range(vars.days), lhsKM, color='green', label='lhs')

plt.xlabel('days')
plt.ylabel('pfs')
plt.legend()

plt.show()