from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det

class LMSTrainer(BaseEstimator):
    def __init__(self, analitic=False):
        self.analitic = analitic
        self._trained = False
        self.w_hat = None

    def __hTeta(self, x):

        res = self.teta0 + self.teta1 * x

        return res

    def __costFunction(self, X_train, y_train=None):
        pass

    def fit(self, X_train, y_train=None):

        if self.analitic:

            X = np.insert(X_train, 0, 1, 1)
            temp = np.dot(X.T, X)
             # (X^T * X)^-1 * X^T * y
            if np.linalg.det(temp) == 0:
                w_hat = np.dot(np.dot(np.linalg.pinv(temp), X.T), y_train)
            else:
                w_hat = np.dot(np.dot(np.linalg.inv(temp), X.T), y_train)
            
            self.w_hat = w_hat
            self.coef = self.w_hat[1:]
            self.intercept = self.w_hat[0]
            plt.plot(self.coef)
            plt.show()

        else:
            pass

        self._trained = True
        return self
        
    def predict(self, X_test, y_test=None):
        
        if not self._trained:
            raise RuntimeError("You must train classifer before predicting data!")
            
        return 1

X_train = [[1,2],[1,2]]
y_train = [1,1]
X_test  = [[1,2]]
y_test  = [1]
trainer = LMSTrainer(analitic=True)
predictor = trainer.fit(X_train=X_train, y_train=y_train)
