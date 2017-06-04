# Copyright 2017 Wagner <wagnerweb2010@gmail.com>
#
import numpy as np
from scipy import optimize


class neuralnetwork(object):
    """Neural network of 3 layers"""
    def __init__(self, INPUTLAYERSIZE=2, H1LAYERSIZE=3,
                 H2LAYERSIZE=3, OUTPUTLAYERSIZE=1,
                 ERR=0.0001):
        # Constants
        self.INPUTLAYERSIZE = INPUTLAYERSIZE
        self.H1LAYERSIZE = H1LAYERSIZE
        self.H2LAYERSIZE = H2LAYERSIZE
        self.OUTPUTLAYERSIZE = OUTPUTLAYERSIZE
        self.ERR = 0.0001
        # Weights
        self.weight1 = np.random.rand(INPUTLAYERSIZE, H1LAYERSIZE)
        self.weight2 = np.random.rand(H1LAYERSIZE, H2LAYERSIZE)
        self.weight3 = np.random.rand(H2LAYERSIZE, OUTPUTLAYERSIZE)

    def foward(self, data, extended=False):
        """Fowards a input "data"
        Params:  self.foward(data, extended=False)
        If extended = True, returns outs, s3, a2, s2, a1, s1
        if extended = False, returns outs"""
        # First layer
        z1 = np.dot(data, self.weight1)
        a1 = self.ativation(z1)
        # Second layer
        z2 = np.dot(a1, self.weight2)
        a2 = self.ativation(z2)
        # Third layer
        z3 = np.dot(a2, self.weight3)
        outs = self.ativation(z3)

        if extended is True:
            return outs, z3, a2, z2, a1, z1  # Returns
        else:
            return outs

    def ativation(self, x):
        """Sigmoid of x"""
        return 1 / (1 + np.exp(-x))

    def derivativ(self, x):
        """Derivative sigmoid of x"""
        atvx = self.ativation(x)
        return atvx * (1 - atvx)

    def cost(self, trainData, tgtData):
        """Returns the error caused by the atual weights
        This code has overfitting prevention"""
        z3 = self.foward(trainData)
        return (sum((z3 - tgtData) ** 2)
                / (2 * trainData.shape[0])
                + ((self.ERR / 2)
                   * ((sum(self.weight1**2)
                       + sum(self.weight2**2)
                       + sum(self.weight3**2)))))

    def primedcost(self, trainData, tgtData):
        """Calculates primed cost of Dcost/dW3, Dcost/dW2, Dcost/dW1"""
        outs, z3, a2, z2, a1, z1 = self.foward(trainData, extended=True)
        error = (outs - tgtData)
        NSHAPE = trainData.shape[0]

        # Calculating dCostdW3
        sigma3 = np.multiply(error, self.derivativ(z3))
        dCostdW3 = np.dot(a2.T, sigma3) / NSHAPE + self.ERR*self.weight3
        # Calculating dCostdW2
        sigma2 = np.dot(sigma3, self.weight3.T)
        sigma2 = sigma2 * self.derivativ(z2)
        dCostdW2 = np.dot(a1.T, sigma2) / NSHAPE + self.ERR*self.weight2
        # Calculating dCostdW1
        sigma1 = np.dot(sigma2, self.weight2.T)
        sigma1 = sigma1 * self.derivativ(z1)
        dCostdW1 = np.dot(trainData.T, sigma1) / NSHAPE + self.ERR*self.weight1

        return dCostdW3, dCostdW2, dCostdW1

    def organizeline(self, A, B, C):
        return np.concatenate((A.ravel(), B.ravel(), C.ravel()))

    def getweight(self):
        # Get W1 and W2 unrolled into vector:
        return self.organizeline(self.weight1, self.weight2, self.weight3)

    def setweight(self, params):
        # Set W1, W2 and W3 using single paramater vector.
        # W1
        W1START = 0
        W1END = self.INPUTLAYERSIZE * self.H1LAYERSIZE
        self.weight1 = np.reshape(params[W1START:W1END],
                                  (self.INPUTLAYERSIZE, self.H1LAYERSIZE))
        # W2
        W2START = W1END
        W2END = W2START + (self.H1LAYERSIZE * self.H2LAYERSIZE)
        self.weight2 = np.reshape(params[W2START:W2END],
                                  (self.H1LAYERSIZE, self.H2LAYERSIZE))
        # W3
        W3START = W2END
        W3END = W3START + (self.H2LAYERSIZE * self.OUTPUTLAYERSIZE)
        self.weight3 = np.reshape(params[W3START:W3END],
                                  (self.H2LAYERSIZE, self.OUTPUTLAYERSIZE))

    def getgrad(self, traindata, tgtdata):
        dCostdW3, dCostdW2, dCostdW1 = self.primedcost(traindata, tgtdata)
        return self.organizeline(dCostdW1, dCostdW2, dCostdW3)


class trainer(object):
    def __init__(self, NN):
        self.NN = NN

    def getCostGrad(self, weights, X, Y):
        self.NN.setweight(weights)
        cost = self.NN.cost(X, Y)
        grad = self.NN.getgrad(X, Y)

        return cost, grad

    def train(self, trainData, tgtdata):
        # Initial value for paramters
        params0 = self.NN.getweight()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.getCostGrad, params0,
                                 jac=True, method='BFGS',
                                 args=(trainData, tgtdata), options=options)

        self.NN.setweight(_res.x)
        self.optresults = _res


if __name__ == '__main__':

    X = np.array([[2, 3],
                  [4, 5],
                  [8, 8],
                  [3, 1],
                  [0, 5]])

    Y = np.array([[1],
                  [3],
                  [4],
                  [0],
                  [5]])

    X = X/8
    Y = Y/10

    a = neuralnetwork()
    cost1, cost2, cost3 = a.primedcost(X, Y)
    print(cost1, cost2, cost3)
