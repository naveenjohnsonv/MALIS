import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics import accuracy_score
class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        self.X = []
        self.y = []
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''        
        self.X = X
        self.y = y

    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        distances = self.minkowski_dist(X_new, p)
        # Get the indices of the k nearest neighbors for each point in X_new.
        k_nearest_neighbors = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        # Get the labels of the k nearest neighbors.
        k_nearest_labels = self.y[k_nearest_neighbors]
        # Predict the most common class among the k nearest neighbors.
        y_hat = np.array([np.argmax(np.bincount(labels)) for labels in k_nearest_labels]) # ChatGPT was used to debug an error enountered here
        return y_hat

    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        dst = distance_matrix(X_new, self.X, p=p)
        return dst

# Load the training data
df_train = pd.read_csv('training.csv')
X_train = df_train[['X1', 'X2']].values
y_train = df_train['y'].values.astype(int)

# Load the validation data
df_val = pd.read_csv('validation.csv')
X_val = df_val[['X1', 'X2']].values
y_val = df_val['y'].values.astype(int)

# Do Hyper-parameter tuning for a set of chosen k and p values
best_k, best_p, best_accuracy = 0, 0, 0
k_values = list(range(1, 51))
p_values = list(range(1, 6))

for k in k_values:
    for p in p_values:
        model = KNN(k)
        model.train(X_train, y_train)
        y_pred = model.predict(X_val, p)
        accuracy = accuracy_score(y_val, y_pred)
        if accuracy > best_accuracy:
            best_k, best_p, best_accuracy = k, p, accuracy

# Print the best k and p
print(f'Best k: {best_k}')
print(f'Best p: {best_p}')
print(f'Best Accuracy: {best_accuracy}')