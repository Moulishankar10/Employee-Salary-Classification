# EMPLOYEE SALARY CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 

#IMPORTING MODULES
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)
