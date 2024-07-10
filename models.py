import vars
import numpy as np

#Returns a matrix with the sizes of all of the people
def gdr(params, scanDates, n=vars.n):
    sizes = np.empty((n, len(scanDates[0])))
    for i in range(n):
        g = params[i, 0]
        d = params[i, 1]
        r = params[i, 2]
        
        for j in range(vars.numScans):
            temp = max(vars.minSize, np.exp(g*scanDates[i, j] - d/r * (1 - np.exp(-r*scanDates[i, j]))))
            if (temp > vars.maxSize):
                sizes[i, j:vars.numScans] = vars.maxSize + 1
                break
            else:
                sizes[i, j] = temp
    return sizes
