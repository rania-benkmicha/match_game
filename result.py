import json

def save_match(file_path,match_list):
    with open(file_path, 'w') as json_file:
         json.dump(match_list, json_file)