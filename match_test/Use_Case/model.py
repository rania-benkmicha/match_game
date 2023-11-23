
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, Activation
from tensorflow.keras import layers


class regression_length_model():
    def __init__(self):
        
        self.model=RidgeCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1])
    def fit(self,X_train,y_train):
        self.model.fit(X_train, y_train)
    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return int(y_pred[0])
    def evaluate(self,X_test,y_test):
        y_pred = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)#R_2 score
        rmse_score=mean_squared_error(y_test,y_pred)
        return (accuracy,rmse_score)
        
class lstm_generator():
    def __init__(self,sequence_length,n_x):
        self.model= Sequential()
        self.model.add(LSTM(256, input_shape=(sequence_length, n_x), return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(256))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(n_x, activation='softmax'))
    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train, epochs=20, batch_size=32)
    def predict(self,seed):
        prediction = self.model.predict(np.expand_dims(seed, 0))
        return prediction

class regression_norm_values_model():
    def __init__(self):
      
        self.model=RandomForestRegressor(n_estimators=100)
    def fit(self,X_train,y_train):
        self.model.fit(X_train, y_train)
    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred[0]
    def evaluate(self,X_test,y_test):
        y_pred = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)#R_2 score
        rmse_score=mean_squared_error(y_test,y_pred)
        return (accuracy,rmse_score)   

     