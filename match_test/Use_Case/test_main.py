from model import *
from etl import *
from utils import *
from result import *
import argparse




def main(match_length,n_x,set_action,playstyle_actions=["shot", "sprint","pass"],seed=['run','shot','shot','cross'],path='./data_match',file_name='match_1.json',game_number=1):
    sequence_length=4
    game_data=Extracton(path,file_name)
    game_data.extract_data()
    #print('here',game_data.extracted_data)
    transformed_game=Transform(dataframe=game_data.extracted_data)
    X_train_second_model, X_test_second_model, y_train_second_model, y_test_second_model,label_encoder_second_model=transformed_game.regressor_data_construction()
    #print(X_train_second_model,y_train_second_model)
    X_train_third_model, X_test_third_model, y_train_third_model, y_test_third_model,label_encoder_third_model=transformed_game.create_for_normvalues_data()
    #print(X_train_third_model,y_train_third_model)
    second_model=regression_length_model()
    third_model=regression_norm_values_model()
    second_model.fit(X_train_second_model.values,y_train_second_model)
    third_model.fit(X_train_third_model.values,y_train_third_model)
    #print(third_model.predict(X_test_third_model))
    X_train_lstm_model,y_train_lstm_model=transformed_game.create_lstm_data(playstyle_actions)
    #print(y_train_lstm_model)
    generator_model=lstm_generator(sequence_length,n_x)
    generator_model.compile()
    generator_model.fit(X_train_lstm_model,y_train_lstm_model)
    match_list=generation_sequence(generator_model,second_model,third_model,seed,n_x,sequence_length,set_action,match_length,label_encoder_third_model,label_encoder_second_model)
    file_path = 'result/match_game_{}_result.json'.format(game_number)

# Save the list of the simulated game  to a JSON file
    save_match(file_path,match_list)        
    print('finish generating match game')



parser = argparse.ArgumentParser(description='SCINet on ETT dataset')
parser.add_argument('--file_name', type=str, default='match_1.json', help='input data file name')
parser.add_argument('--path', type=str, default='./data_match', help='path to result folder')
parser.add_argument('--seed', type=list, default=['run','shot','shot','cross'], help='seed to generate match')
parser.add_argument('--set_action', type=list, default=['walk','run','dribble','rest','pass','tackle','shot','cross'], help='set of match actions')
parser.add_argument('--match_length', type=int, default=5, help='length of match')
parser.add_argument('--number_of_games', type=int, default=1, help='number of games')
parser.add_argument('--playstyle_actions', type=list, default=["shot", "sprint","pass"], help='specific type of game')
args = parser.parse_args()

n_x=len(args.set_action)

for i in range(args.number_of_games):
    main(args.match_length,n_x,args.set_action,args.playstyle_actions,args.seed,args.path,args.file_name,i+1)
    print(n_x)