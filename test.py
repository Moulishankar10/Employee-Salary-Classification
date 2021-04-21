# EMPLOYEE SALARY CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 

#IMPORTING MODULES
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# INPUT DATA
print("\n\n Enter the required details of an employee to explore the employee's salary range. \n\n")

sl = int(input("\nSatisfaction level of the Employee's performance (%) : "))
le = int(input("\nRecent Evaluation of the Employee (%) : "))
nop = int(input("\nNumber of projects completed successfully : "))
amh = int(input("\nAverage working hours per month : "))
yac = int(input("\nNumber of Years working in the company : "))

# PROCESSING INPUT DATA
details = [sl,le,nop,amh,yac]