import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(predictions, labels):
    """
    Calculate evaluation metrics for classification results.
    
    Args:
        predictions: List/array of predicted labels
        labels: List/array of true labels
        
    Returns:
        Dictionary containing accuracy and F1 score
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": float(accuracy),
        "f1": float(f1)
    } 