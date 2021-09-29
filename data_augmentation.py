import numpy as np
from scipy.interpolate import CubicSpline

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        x_curve = []
        for i in range(X.shape[1]):
            x_curve.append(CubicSpline(xx[:,i], yy[:,i])(x_range))
        return np.array(x_curve).transpose()

def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def DA_TimeWarp(X, sigma=0.2):
    def DistortTimesteps(X, sigma=0.2):
        tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = [(X.shape[0]-1)/tt_cum[-1,i] for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
        return tt_cum
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
    return X_new

def DA(X, y, iterations = 10):
    features = X
    labels = y
    for iter in range(iterations):
        DA_batch = []
        DA_batch_label = []
        for i in range(X.shape[0]):
            DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.001)))
            DA_batch_label.append(y[i])
#             if y[i] == 0 and random.uniform(0,1) < 0.22311:
#                 DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.002)))
#                 DA_batch_label.append(y[i])
        DA_batch = np.array(DA_batch)
        DA_batch_label = np.array(DA_batch_label)
        features = np.vstack((features,DA_batch))
        labels = np.concatenate((labels, DA_batch_label))
    
    return features, labels

def DA_wc(X, y, iterations = 10):
    features = X
    labels = y
    for iter in range(iterations):
        DA_batch = []
        DA_batch_label = []
        for i in range(X.shape[0]):
            DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.001)))
            DA_batch_label.append(y[i])
#             if y[i] == 0 and random.uniform(0,1) < 0.22311:
#                 DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.002)))
#                 DA_batch_label.append(y[i])
        DA_batch = np.array(DA_batch)
        DA_batch_label = np.array(DA_batch_label)
        features = np.vstack((features,DA_batch))
        labels = np.vstack((labels, DA_batch_label))
    
    return features, labels

def DA_3cla(X, y, iterations = 10):
    features = X
    labels = y
    for iter in range(iterations):
        DA_batch = []
        DA_batch_label = []
        for i in range(X.shape[0]):
            DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.001)))
            DA_batch_label.append(y[i])
            if y[i] == 2:
                for j in range(4):
                    DA_batch.append(DA_TimeWarp(DA_MagWarp(features[i], 0.001)))
                    DA_batch_label.append(y[i])
        DA_batch = np.array(DA_batch)
        DA_batch_label = np.array(DA_batch_label)
        features = np.vstack((features,DA_batch))
        labels = np.concatenate((labels, DA_batch_label))
    
    return features, labels