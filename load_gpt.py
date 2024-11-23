

import torch
import numpy as np
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
from utils import files_to_dict, generate_seed_list, \
    avg_compute_similarity, line_chart, preprocess, plot_confusion_matrix

# configs
SEED_SAVE_FILE = "seed_pred_base.csv"
CONFUSION_FILE = "target.png"
RUNS = 20
model_path = "model/fine_tuned_gpt2_target/"    
#model_path = "model/fine_tuned_gpt2_vanilla/"
# model_path = "model/fine_tuned_gpt2_forget/"
# model_path = "model/fine_tuned_gpt2_forget_percentile/"
# model_path = "model/fine_tuned_gpt2_forget_vanilla/"
# forget = [True if 'forget' in model_path else False][0]
forget = True

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

        
class Seeder():
    def __init__(self, seed_content):
        self.seed_dict = {}
        self.seed_content = seed_content
        self.data_dict = {}
        self.load_seed()
        
    
    def generate_single_seed(self, testing_file):
        seed = generate_seed_list(testing_file, seed_content=self.seed_content, size=1)[0]
        if not self.seed_dict.get(testing_file):
            self.seed_dict[testing_file] = [seed]
        else:
            self.seed_dict[testing_file].append(seed)
        return seed
    
    def replace_seed(self, old_seed, testing_file):
        seed = generate_seed_list(testing_file, seed_content=self.seed_content, size=1)[0]
        self.seed_dict[testing_file][self.seed_dict[testing_file].index(old_seed)] = seed
        return seed
    
    def save_seed(self, pred_list, file_name):
        file_names = []
        seed_list = []
        for file, seeds in self.seed_dict.items():
            for seed in seeds:
                file_names.append(file)
                seed_list.append(seed)
        pd.DataFrame({"file": file_names, "seed": seed_list, "pred":pred_list}).to_csv(file_name, header=True, index=False)
        
    def load_seed(self):
        print("Loading Seed from seed_pred_base.csv")
        seed_df = pd.read_csv("seed_pred_base.csv")
        file_name = seed_df["file"]
        seeds = seed_df["seed"]
        
        for idx, seed in enumerate(seeds):
            if file_name[idx] not in self.seed_dict:
                self.seed_dict[file_name[idx]] = [seed]
            else:
                self.seed_dict[file_name[idx]].append(seed)
        
    def get_item(self, file, idx):
        if not self.seed_dict:
            self.load_seed()
        seed_file = self.seed_dict.get(file)
        if seed_file and len(seed_file) > idx:
             return seed_file[idx]
        return None
    


analysis = "similarity_index.json"
data_dict = files_to_dict()
runs = RUNS
seeder = Seeder(seed_content=data_dict)
similarity_dict = {}
pred_list = []

#fcount = 0
for testing_file in seeder.seed_dict.keys():
    # seed = generate_seed_list(testing_file, seed_content=data_dict, size=1)[0]
    for i in range(runs):
        if forget and seeder.get_item(testing_file,i):
            seed = seeder.get_item(testing_file, i)
            #fcount += 1
        else:
            seed = seeder.generate_single_seed(testing_file)
        print("Testing File: ", testing_file)
        print("seed: ", seed)
        pred = ""
        count = 0
        pred = generate_text(seed)
        if not forget:
            while len(pred.split()) < 20 or count>30:
                seed = seeder.replace_seed(seed, testing_file)
                pred = generate_text(seed)
                print(f"New Seed: {seed}, count: {count}")
                count += 1
        
        print("pred: ", pred)
        pred_list.append(pred)
        
        if similarity_dict.get(testing_file):
            score_inner = testing_with_all_files(testing_file, data_dict, pred)
            for key in similarity_dict[testing_file].keys():
                similarity_dict[testing_file][key] += score_inner[key]
        else:
            similarity_dict[testing_file] = testing_with_all_files(testing_file, data_dict, pred)
            

    for key,value in similarity_dict[testing_file].items(): 
        similarity_dict[testing_file][key] = value/runs
    
    #if testing_file=="virus.txt": import pdb;pdb.set_trace()
           

# if os.path.isfile(analysis):
#     try:
#         similarity_dict = json.load(open(analysis, 'r'))
#     except:
#         similarity_dict = {}

# analysis_file = open(analysis, 'w')
seeder.save_seed(pred_list=pred_list, file_name=SEED_SAVE_FILE)

   
print(similarity_dict)
plot_confusion_matrix(similarity_dict, file_name=CONFUSION_FILE)
# json.dump(similarity_dict, analysis_file)
# line_chart(similarity_dict, TESTING_FILE)
    
