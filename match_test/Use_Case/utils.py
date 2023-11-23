import pandas as pd
import numpy as np

def norm_value_prediction(third_model,norm_length,actual_action,previous_action,label_encoder_third):
    norm_list=[]
    for i in range(norm_length):
       #i is considered as timestep in norm list
       norm_list.append(third_model.predict([[label_encoder_third.transform([actual_action])[0],norm_length,label_encoder_third.transform([previous_action])[0],i]]))
    return norm_list    


def generation_sequence(lstm_model,second_model,third_model,seed,n_x,sequence_length,set_action,match_duration,label_encoder_third,label_encoder_second):
    total_number_of_temporal_acquisition=match_duration*60//0.01

    pattern=np.zeros((sequence_length, n_x))
    for t in range(sequence_length):
        pattern[t, set_action.index(seed[t])] = 1#one hot encoding for seed
    match_list = []
    
    #dict_counter={'walk':0,'run':0,'dribble':0,'rest':0,'pass':0,'tackle':0,'shot':0,'cross':0}

    total_gait_element=0
    temporal_action_index=-1
    while total_gait_element <total_number_of_temporal_acquisition:
        temporal_action_index+=1
        penality=-0.4#normal game
        #penalty_array=[-0.6,-0.6,0,-0.4,0,0,+0.2,0]
        column_sums = np.sum(pattern, axis=0)
        #print("number",np.multiply(penalty_array, column_sums))
           
        prediction = lstm_model.predict(pattern)
        #prediction[0]+= np.multiply(penalty_array, column_sums)  
        prediction[0]+=penality*column_sums     
        label = np.argmax(prediction)#retourne l'indice de la valeur maximale
        generated_action=set_action[label]

        previous_action_one_hot_encoding=pattern[-1]
        action_index = np.argmax(previous_action_one_hot_encoding)
        previous_action = set_action[action_index]#getting previous action

        #length norm prediction
        generated_length=second_model.predict([[label_encoder_second.transform([generated_action])[0],label_encoder_second.transform([previous_action])[0],temporal_action_index]])
        
        total_gait_element+=generated_length
        if total_gait_element>total_number_of_temporal_acquisition:
            generated_length=total_number_of_temporal_acquisition-total_gait_element#fixed length for last genrated action
            norm_list=norm_value_prediction(third_model,int(generated_length),generated_action,previous_action,label_encoder_third)
            break
        else:
            norm_list=norm_value_prediction(third_model,generated_length,generated_action,previous_action,label_encoder_third)
   
        print("generated action",generated_action)
        print("previosua action",previous_action)
        print("predictionprobability",prediction)
        print("length",generated_length)
        print("norm list",norm_list)

        pattern = np.append(pattern[1:],
                            np.expand_dims(np.eye(n_x)[label], 0),
                            axis=0)
        gait_dict={}
        gait_dict['norm']=norm_list
        gait_dict['label']=generated_action
        match_list.append(gait_dict)
    print(match_list)   
    return match_list




    
    


    