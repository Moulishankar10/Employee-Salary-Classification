# EMPLOYEE SALARY CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 

#IMPORTING MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import save_model 
import matplotlib.pyplot as plt

# IMPORTING DATA
data = pd.read_csv("data/salary.csv")

# PREPROCESSING DATA
x = data.iloc[:,[0,1,2,3,4]].values
y = data.iloc[:,5].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = None, stratify = y)

# BUILDING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(24, input_dim = 5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# TRAINING THE DATA
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 15, batch_size = 6, validation_data = (x_val, y_val))
print("\n\n ----- Model is trained successfully ! ----- \n\n")

# VISUALISING THE MODEL LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
plt.show()

# VISUALISING THE MODEL ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('MODEL ACCURACY')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='top right')
plt.show()

# ACCURACY OF THE MODEL
score = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAccuracy of the model : {round(score[1]*100,2)}%")

# SAVING THE TRAINED MODEL
PATH = './model'
save_model(model,PATH)
print(f"\n\n ---- Successfully stored the trained model at {PATH} ---- \n\n")
