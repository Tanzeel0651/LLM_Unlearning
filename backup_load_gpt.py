

import torch
import numpy as np
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
from utils import files_to_dict, generate_seed_list, \
    avg_compute_similarity, line_chart, preprocess, plot_confusion_matrix

model_path = "model/fine_tuned_gpt2/"
# model_path = "model/fine_tuned_gpt2_forget_set/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the generation
# TESTING_FILE = "clock.txt"

def testing_with_all_files(TESTING_FILE):
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

# analysis = "similarity_index.json"
# data_dict = files_to_dict()
# runs = 3
# similarity_dict = {}
# for testing_file in os.listdir('dataset/'):
#     for _ in range(runs):
#         seed = generate_seed_list(testing_file, seed_content=data_dict, size=1)[0]
#         print("Testing File: ", testing_file)
#         print("seed: ", seed)
#         pred = ""
#         count = 0
#         while len(pred.split()) < 20 or count>30:
#             pred = generate_text(seed)
#             seed = generate_seed_list(testing_file, seed_content=data_dict, size=1)[0]
#             print(f"New Seed: {seed}, count: {count}")
#             count += 1
        
#         print("pred: ", pred)
        
#         if similarity_dict.get(testing_file):
#             score_inner = testing_with_all_files(testing_file)
#             for key in similarity_dict[testing_file].keys():
#                 similarity_dict[testing_file][key] += score_inner[key]
#         else:
#             similarity_dict[testing_file] = testing_with_all_files(testing_file)
            

#     for key,value in similarity_dict[testing_file].items():
#         similarity_dict[testing_file][key] = value/runs
        
class seed_list():
    def __init__(self, seed_content):
        self.seed_list = []
        self.length = 20
        self.seed_content = seed_content
    
    def generate_single_seed(self, testing_file):
        seed = generate_seed_list(testing_file, seed_content=self.seed_content, size=1)[0]
        self.seed_list.append(seed)
        return seed
    
    def replace_seed(self, old_seed, testing_file):
        seed = generate_seed_list(testing_file, seed_content=self.seed_content, size=1)[0]
        self.seed_list[self.seed_list.index(old_seed)] = seed
        return seed
    
    def save_seed(self):
        pd.DataFrame(self.seed_list).to_csv("seed_list.csv", header=False, index=False)
        
    
    


analysis = "similarity_index.json"
data_dict = files_to_dict()
runs = 3
seed_list = seed_list(seed_content=data_dict)
similarity_dict = {}

for testing_file in os.listdir('dataset/'):
    seed = seed_list.generate(testing_file, size=1)
    # seed = generate_seed_list(testing_file, seed_content=data_dict, size=1)[0]
    for _ in range(runs):
        print("Testing File: ", testing_file)
        print("seed: ", seed)
        pred = ""
        count = 0
        pred = generate_text(seed)
        while len(pred.split()) < 20 or count>30:
            seed = seed_list.replace_seed(testing_file, seed_content=data_dict, size=1)[0]
            pred = generate_text(seed)
            print(f"New Seed: {seed}, count: {count}")
            count += 1
        
        print("pred: ", pred)
        
        if similarity_dict.get(testing_file):
            score_inner = testing_with_all_files(testing_file)
            for key in similarity_dict[testing_file].keys():
                similarity_dict[testing_file][key] += score_inner[key]
        else:
            similarity_dict[testing_file] = testing_with_all_files(testing_file)
            

    for key,value in similarity_dict[testing_file].items():
        similarity_dict[testing_file][key] = value/runs
           

# if os.path.isfile(analysis):
#     try:
#         similarity_dict = json.load(open(analysis, 'r'))
#     except:
#         similarity_dict = {}

# analysis_file = open(analysis, 'w')
  

   
print(similarity_dict)
# plot_confusion_matrix(similarity_dict)
# json.dump(similarity_dict, analysis_file)
# line_chart(similarity_dict, TESTING_FILE)

