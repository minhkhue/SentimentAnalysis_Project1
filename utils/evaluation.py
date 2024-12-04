import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from underthesea import word_tokenize, pos_tag, sent_tokenize
from utils import TextProcessing as tpr
import regex
import string
import os
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")
    
# Ghi model sau khi đã train
import pickle
def Save_Object(obj,filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    return

def Load_Object(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Hàm tính các metrics score
def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }