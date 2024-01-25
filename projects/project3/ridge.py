import numpy as np

class RidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha
        self.W = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Add column of ones for bias term
        X = np.c_[np.ones(n_samples), X]
        
        # Calculate weights analytically
        I = np.identity(n_features + 1)
        reg_term = self.alpha * I
        reg_term[0,0] = 0 # Do not regularize bias term
        
        self.W = np.linalg.pinv(X.T.dot(X) + reg_term).dot(X.T).dot(y)
        
    def predict(self, X):
        # Add column of ones for bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        return X.dot(self.W)
    
    def MSE(self, y_pred, y_test):
        MSE = np.mean((y_pred - y_test) ** 2)
        return MSE