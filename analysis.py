#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def testing_with_all_files(TESTING_FILE, data_dict, pred):
    similarity_dict_inner = {}
    for file_name in data_dict.keys():
        print("File: ", file_name, end="\t")
        if TESTING_FILE == file_name:
            print("POSITIVE", end="\t")
        else:
            print("NEGATIVE", end="\t")
        # similarity, avg_, max_, min_ = avg_compute_similarity(data_dict[file_name], [pred])
        similarity = avg_compute_similarity(data_dict[file_name], preprocess(pred), model="transformer")
        
        print(similarity)
        similarity_dict_inner[file_name] = float(similarity)

   
    return similarity_dict_inner
"""
Created on Sat Sep  7 17:50:14 2024

@author: tanzeel
"""
import pandas as pd
from utils import files_to_dict, avg_compute_similarity, plot_confusion_matrix, preprocess

hopper_data = pd.read_csv('seed_pred_reinforce.csv')
# access_data = pd.read_csv('seed_pred_forget_access.csv')
# hopper_data = pd.read_csv("Results/hopper/seed_pred_base.csv")

data_dict = {"carbon.txt":[],"clock.txt":[],"soccer.txt":[],"human.txt":[],"airplane.txt":[]}

keys = list(data_dict.keys())
key_index = 0
for index, val in enumerate(list(hopper_data["pred"])):
    data_dict[keys[key_index]].append(val)
    if (index+1)%20==0:
        key_index += 1
    
    
similarity_dict = {}
content = files_to_dict()     
for testing_file, preds in data_dict.items():
    print("Testing File: ",testing_file)
    similarity_dict[testing_file] = {}
    for pred in preds:
        similarity_dict[testing_file] = testing_with_all_files(testing_file, content, pred)
    
    
    #similarity_dict[testing_file][content_file] = similarity_dict[content_file]
    print("Score: ",similarity_dict[testing_file])
    

plot_confusion_matrix(similarity_dict, "reinforce.png")



