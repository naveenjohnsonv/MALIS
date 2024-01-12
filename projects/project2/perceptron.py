import numpy as np

class Perceptron:
    
    def __init__(self,learning_rate,epochs):
        """
        Constructor for the Perceptron class
        Args:
            w (numpy array): Weights vector
            b (float): Bias term
            learning_rate (float): Learning rate for the weight update rule 
            epochs (int): Number of passes over the training dataset
        """
        self.w = None
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def train(self, X, y):
        """
        Trains the perceptron model
        Args:
            X (numpy array): Array of training samples 
            y (numpy array): Array of labels (0 or 1) for samples
        """
        n_samples, n_features = X.shape
        
        # Initialize weights  
        self.w = np.zeros(n_features)
        
        # Train model 
        for _ in range(self.epochs):
            for i in range(n_samples):
                # Perceptron update rule
                update = self.learning_rate * (y[i] - self.predict(X[i]))
                self.w += update * X[i]
                self.b += update
        
    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X
        Args:
            X (numpy array): Array of samples 
        Returns:
            numpy array: Predicted class labels (0 or 1) for X
        """
        
        # Predict labels 0 or 1
        y_pred = np.where(X @ self.w + self.b <= 0, 0, 1)  
        return y_pred