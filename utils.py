import random
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# Download necessary NLTK data
nltk.download("wordnet")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import os


def preprocess(text):
    # Remove specified punctuation and stem/lemmatize
    exclude = ['_', '-', '(', ')', '[', ']', '{', '}', '.', ',', ';', ':', '"']
    result = ''.join([lemmatizer.lemmatize(stemmer.stem(i)) for i in text if not i.isdigit() and i not in exclude]).strip()
    if result:
        return result

def files_to_dict(datadir="dataset/"):
    seed_content = {}
    for file in os.listdir("dataset/"):
        content = []
        if file.endswith("txt"):
            file_ = open(datadir+file, 'r')
            content.extend(file_.read().split("."))
            seed_content[file] = [preprocess(line) for line in content if preprocess(line)]
            print("Seed File Added: ", file)
            file_.close()
    return seed_content


def remove_stopwords(sentence):
    stopwords_ = set(stopwords.words('english'))
    return [word for word in sentence.split() if word not in stopwords_]


def generate_seed_list(file_name, seed_content=None, string_length=4, size=10):
  substrings = []
  if not seed_content:
      seed_content = files_to_dict()
  seed_content = seed_content[file_name]
  # import pdb;pdb.set_trace()
  seed_content = ' '.join(seed_content).split()
  while len(substrings) < size:
    rand_index = random.randint(0, len(seed_content)-(string_length+1))
    substrings.append(" ".join(seed_content[rand_index:rand_index+string_length]))
  return substrings

# Function to generate sentence embeddings
######## Bert Model

def refrence_model(model_name = "bert-base-uncased"):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return bert_tokenizer, bert_model

def get_sentence_embedding(sentence, max_sequence_len=50):
    bert_tokenizer, bert_model = refrence_model()
    inputs = bert_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=max_sequence_len)
    outputs = bert_model(**inputs)
    # Get the embeddings of the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embeddings

# Function to compute similarity index between two sentences
def compute_similarity(sentence1:str, sentence2:str):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    import pdb;pdb.set_trace()
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def compute_similarity_from_embedding(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


####### Transformers
def transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_transformer_similarity(text1, text2, model=SentenceTransformer('all-MiniLM-L6-v2')):
    # Compute embeddings
    #model = transformer_model()
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    return similarity_score.max()

##########
###############

def avg_compute_similarity(sen_list1:list, sen_list2:list, model="bert"):
    similarity = []
    # for sen1 in sen_list1:
    #     for sen2 in sen_list2:
    #         if model=='bert':
    #             similarity.append(compute_similarity(sen1, sen2))
    #         elif model == 'transformer':
    #             similarity.append(compute_transformer_similarity(sen1, sen2))
    model = model.lower()
    if model=='bert':
        return compute_similarity(sen_list1, sen_list2).cpu().numpy()
    elif model == 'transformer':
        return compute_transformer_similarity(sen_list1, sen_list2).cpu().numpy()
    
    # return similarity, sum(similarity)/len(similarity), max(similarity), min(similarity)

def plot_confusion_matrix(similarity_scores_dict: dict, file_name=None):
    all_keys = set(similarity_scores_dict.keys())
    for sub_dict in similarity_scores_dict.values():
        all_keys.update(sub_dict.keys())
    
    # Sort the keys for a consistent ordering
    sorted_keys = sorted(all_keys)

    reordered_dict = {key: {sub_key: similarity_scores_dict.get(key, {}).get(sub_key, 0) for sub_key in sorted_keys} for key in sorted_keys}

    df = pd.DataFrame(reordered_dict)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Similarity Score'})
    
    # Add title and labels
    plt.title("Similarity Confusion Matrix", fontsize=16)
    plt.ylabel("Predicted Document", fontsize=12)
    plt.xlabel("Actual Document", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # Show the plot
    plt.tight_layout()
    # plt.show()
    if file_name:
        plt.savefig(file_name)
        print("Confusion Plot saved: ",file_name)
        

def line_chart(similarity_scores: dict, actual_label):
    # similarity_scores: {Airplane:[1,2,3...], soccer:[],....}
    labels = list(similarity_scores.keys())
    plt.figure()
    l_h = []
    
    for label in labels:
        x = [i for i in range(1, len(similarity_scores[label])+1)]
        y = similarity_scores[label]
        if label==actual_label:
            h, = plt.plot(x, y, markerfacecolor="green", linewidth=3, label="Actual "+label)
        else:
            h, = plt.plot(x, y, markerfacecolor="red", linewidth=3, label=label)
            
        l_h.append(h)
            
    plt.legend(handles=l_h)
    plt.show()           







