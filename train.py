# ERMPLOYEE SALARY CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 

#IMPORTING MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model 
import matplotlib.pyplot as plt

# IMPORTING DATA
data = pd.read_csv("data/salary.csv")