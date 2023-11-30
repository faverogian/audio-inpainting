import numpy as np

def rmse(M1, M2):
    squared_diff = (M1-M2) ** 2 #matrix difference squared elementwise
    msd = np.mean(squared_diff) #mean squared difference
    rmse = np.sqrt(msd) #root mean squared error
    return rmse

def mae(M1, M2): #mean average error
    return np.mean(np.abs(M1 - M2))