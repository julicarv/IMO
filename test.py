import vars
import dataFuns
import fittingFuns
import matplotlib.pyplot as plt
import lmfit
import fittingFuns

#LHS to find a decent starting fit
minDiff = 999999
lhs_params = fittingFuns.lhsKM(vars.lhs_n)

bestLHSparams = vars.params
for param_set in lhs_params:
    if fittingFuns.residualKM(param_set) < minDiff:
        bestLHSparams = param_set
        minDiff = fittingFuns.residualKM(param_set)

#New width on each side is smaller
for p_name in bestLHSparams:
    bestLHSparams[p_name].max = 1/5 * vars.params[p_name].max + 4/5 * vars.params[p_name].value
    bestLHSparams[p_name].min = 1/5 * vars.params[p_name].min + 4/5 * vars.params[p_name].value
print(bestLHSparams)


optimiser_kws = {'method':'least_squares', 'xtol':1e-12, 'ftol':1e-10, 'max_nfev':200, 'verbose':2}
fit_obj = lmfit.minimize(fittingFuns.residualKM, bestLHSparams, **optimiser_kws)
bestParams = fit_obj.params


#New part of the code
bestKM = dataFuns.parsToKM(bestParams)

#Need to change bestLHSparams format before doing this line
#lhsKM = dataFuns.parsToKM(bestLHSparams)

print(bestParams)


#New part of the code
plt.figure(figsize=(10, 6))

plt.plot(range(vars.days), bestKM, color='blue', label='fit')
plt.plot(range(vars.days), vars.idealKM, color='red', label='ideal')
#plt.plot(range(vars.days), lhsKM, color='green', label='lhs')

plt.xlabel('days')
plt.ylabel('pfs')
plt.legend()

plt.show()
