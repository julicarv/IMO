import vars
import numpy as np

def gdr(params, KM=False):
    sizes = np.empty((vars.n, vars.days))
    for i in range(vars.n):
        g = params[i, 0]
        d = params[i, 1]
        r = params[i, 2]
        big = False
        for j in range(vars.days):
            sizes[i, j] = np.exp(g*j - d/r * (1 - np.exp(-r*j)))
            if(KM and sizes[i, j] > 4):
                while(j < vars.days - 1):
                    j += 1
                    sizes[i, j] = 5
                break
    return sizes