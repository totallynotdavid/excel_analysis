from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import numpy as np
import pandas as pd

def entrenar_regresor_mlp(X_train, Y_train):
    """
    Train the MLP Regressor model with the given training data.
    
    Parameters:
    - X_train (array-like): Features for training.
    - Y_train (array-like): Target values for training.
    
    Returns:
    - modelo_red_neuronal (MLPRegressor): Trained model.
    """
    modelo_red_neuronal = MLPRegressor(activation='logistic', hidden_layer_sizes=(200), max_iter=1000, solver='adam')
    modelo_red_neuronal.fit(X_train, Y_train)
    return modelo_red_neuronal

def get_optimal_threshold(Y_test, y_pred):
    """
    Compute the optimal threshold for classification based on ROC curve.
    
    Parameters:
    - Y_test (array-like): True target values.
    - y_pred (array-like): Predicted values.
    
    Returns:
    - optimal_threshold (float): Computed optimal threshold value.
    """
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred) # fpr = Tasa de falsos positivos que salen positivos, tpr = Tasa de positivos que salen positivos
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1-fpr, index=i), 'tf': pd.Series(tpr - (1-fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    optimal_threshold = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds'].values[0]
    return optimal_threshold

def compute_predicted_return(model, X_test):
    """
    Compute the sum of predicted returns using the model.
    
    Parameters:
    - model (MLPRegressor): Trained MLP model.
    - X_test (array-like): Features for prediction.
    
    Returns:
    - predicted_return (float): Total sum of predicted returns.
    """
    return np.sum(model.predict(X_test))