
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import json
import pandas as pd

class Extracton():
    def __init__(self,path='/home/rbk/Desktop/footbar_test/Use_Case_Footbar',file_name='match_1.json'):
        self.path=path
        self.file_name=file_name
        self.extracted_data=None
    def extract_data(self):
        file_match = open('{}/{}'.format(self.path,self.file_name))
        data = json.load(file_match)
        self.extracted_data=pd.DataFrame(data)

class Transform():
    def __init__(self,dataframe,target='label'):
        self.df=dataframe
        self.target_name=target
        self.target=None
    def list_action_data(self):
        data=self.df.copy()
        self.outlier_treatment(data)
        all_actions=list(data[self.target_name])
        return all_actions


    def create_playstyle_sequences_for_lstm(self, playstyle_actions, playstyle_threshold=0.3, min_sequence_length=5):
        actions=self.list_action_data()
        sequences = []
        current_sequence = []
        attacking_actions_count = 0
        for action in actions:
           current_sequence.append(action)
           if action in playstyle_actions:
               attacking_actions_count += 1
           if  attacking_actions_count / len(current_sequence) >= playstyle_threshold and  len(current_sequence) >= min_sequence_length:
            # Check if the current sequence has reached the attacking threshold           
                   sequences.append(current_sequence.copy())
                   current_sequence = []
                   attacking_actions_count = 0
    # If there are remaining actions,
        if current_sequence and len(current_sequence) >= min_sequence_length:
            sequences.append(current_sequence)
        return  sequences


    def one_hot_encoding_lstm(self,playstyle_actions):
        sequences=self.create_playstyle_sequences_for_lstm( playstyle_actions, playstyle_threshold=0.3, min_sequence_length=5)
        X_list = []
        set_action=['walk','run','dribble','rest','pass','tackle','shot','cross']
        n_x=len(set_action)
        for element in sequences:     
        # convert to one-hot-encoding
            T_x = len(element)    
            X_ohe = np.zeros((T_x, n_x))
            for t in range(T_x):        
                X_ohe[t, set_action.index(element[t])] = 1
            # add to the list
            X_list.append(X_ohe)
        return X_list
    
       
    def create_lstm_data(self,playstyle_actions):
        X_list=self.one_hot_encoding_lstm(playstyle_actions)
        X_train_list = []
        y_train_list = []
        sequence_length=4
        for example in X_list:     
          for i in range(example.shape[0] - sequence_length):       
             X_train_list.append(example[i:i+sequence_length])
             y_train_list.append(example[i+sequence_length])
        X_train = np.asarray(X_train_list)
        y_train = np.asarray(y_train_list)
        return(X_train,y_train)

    def outlier_treatment(self,dataframe):
        dataframe =dataframe.drop(dataframe[dataframe[self.target_name] == 'no action'].index)
    
    def has_missing_values(self,lst):
        ''' categorical features'''
        return any(x is None or np.isnan(x) for x in lst)

    def interpolate_time_series(self,lst):
        ''' numerical features'''
        series = pd.Series(lst)
        interpolated_series = series.interpolate(method='linear')
        return interpolated_series.tolist()
    def missing_values(self,dataframe):
        missing_values = dataframe['norm'].apply(self.has_missing_values)
        if missing_values.any()==True:                     
            dataframe['norm'] = dataframe['norm'].apply(self.interpolate_time_series)
        dataframe=dataframe.dropna(subset=['label'])

    def label_encoding(self,dataframe):
       label_encoder= LabelEncoder()
       label_encoder.fit(dataframe[self.target_name])
       dataframe[self.target_name] = label_encoder.transform(dataframe[self.target_name])
       return label_encoder
        
    def regressor_data_construction(self):
        ''' construction data for second model that predict length of the gait based on label/previous label, temporal index''' 
        df_reg=self.df.copy()
        self.missing_values(df_reg)
        self.outlier_treatment(df_reg)
        
        df_reg['sequence_length'] = df_reg['norm'].apply(len)
        df_reg.drop(columns='norm',inplace=True)
        label_encoder=self.label_encoding(df_reg)
        df_reg['precedent_label'] = df_reg[self.target_name].shift()
        df_reg=df_reg.iloc[1:]
        df_reg['precedent_label'] = df_reg['precedent_label'].astype(int)
        df_reg['sample_temporal_index'] = df_reg.index
        y=df_reg['sequence_length']
        x=df_reg.drop(columns='sequence_length')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
        return (X_train, X_test, y_train, y_test,label_encoder)
        
    def create_for_normvalues_data(self):
        ''' construction data for the third model that predicts the norm values based on
         label/previous label/norm length and temporal position in notml list'''
        df_norm=self.df.copy()
        self.missing_values(df_norm)
        self.outlier_treatment(df_norm)
        df_norm['sequence_length'] = df_norm['norm'].apply(len)
        label_encoder=self.label_encoding(df_norm)
        df_norm['precedent_label'] = df_norm[self.target_name].shift()
        df_norm = df_norm.iloc[1:]
        df_norm['precedent_label'] = df_norm['precedent_label'].astype(int)
        df_norm['timestep_in_list']=[list(range(element)) for element in df_norm['sequence_length']]
        df_norm= df_norm.explode(['norm','timestep_in_list'])
        y=df_norm['norm']
        x=df_norm.drop(columns='norm')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5,random_state=42)
        return (X_train, X_test, y_train, y_test,label_encoder)
        


class Load():
    def __init__(self):
        self.data_loaded=None
    def save_data(self):
        self.df.to_csv('/home/rbk/Desktop/footbar test/Use_Case_Footbar/loaded_data.csv', index=False)