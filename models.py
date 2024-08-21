import vars
import numpy as np

#Returns a matrix with the sizes of all of the people
def gdr(params, scanDates):
    sizes = np.empty((vars.numScans,))
    g = params[0]
    d = params[1]
    r = params[2]
    
    for i in range(vars.numScans):
        temp = max(vars.minSize, np.exp(g*scanDates[i] - d/r * (1 - np.exp(-r*scanDates[i]))))
        if (temp > vars.maxSize):
            sizes[i:vars.numScans] = vars.maxSize + 1
            break
        else:
            sizes[i] = temp
            
    return sizes

def gdrMany(params, scanDates, n=vars.n):
    return [gdr(params[i], scanDates[i]) for i in range(n)]
