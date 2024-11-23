

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import os
import re

content = []
for file in os.listdir("dataset/"):
    if file.split(".")[-1] == "txt":
        file = open('dataset/'+file, 'r')
        content.extend(file.read().split("."))
        
print("Length of content: ", len(content))

def preprocess(text):
    result = re.sub(r'[\.\?\!\,\:\;\"]', '', text)
    return result

content = [preprocess(text) for text in content]

input_text = []
output_text = []

for text in content:
    input_text.append("<sos> "+text)
    output_text.append(text+" <eos>")
    
all_text = input_text+output_text

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(all_text)
word_count = len(tokenizer.word_index) + 1
print("Tokenizer word count: ",word_count)

input_text = tokenizer.texts_to_sequences(input_text)
output_text = tokenizer.texts_to_sequences(output_text)


max_length = max([len(x.split()) for x in all_text])

print("Max length of sentence: ", max_length)

input_text = pad_sequences(input_text, maxlen=max_length, padding='post')
output_text = pad_sequences(output_text, maxlen=max_length, padding='post')

print("Input Data Size: ", np.array(input_text).shape)
print("Output Data Size: ", np.array(output_text).shape)

one_hot_targets = np.zeros((len(input_text), max_length, word_count))
for i, target_sequence in enumerate(output_text):
  for t, word in enumerate(target_sequence):
    if word > 0:
      one_hot_targets[i, t, word] = 1


LATENT_DIM = 25
EMBEDDING_DIM = 50


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('/home/tanzeel/Documents/glove_vector/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare embedding matrix
print('Filling pre-trained embeddings...')
embedding_matrix = np.zeros((word_count, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(
  word_count,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  # trainable=False
)


input_ = Input(shape=(max_length,))
initial_h = Input(shape=(LATENT_DIM, ))
initial_c = Input(shape=(LATENT_DIM, ))
# x = Embedding(word_count, EMBEDDING_DIM)(input_)
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
x,_,_ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(word_count, activation='softmax')
output = dense(x)

model = Model([input_, initial_h, initial_c], output)
model.compile(
    loss = "categorical_crossentropy",
    optimizer="adam",
    metrics=["Accuracy"])


print("Training Model....")
z = np.zeros((len(input_text), LATENT_DIM))
r = model.fit(
    [input_text, z, z],
    one_hot_targets,
    batch_size = 32,
    epochs=500)



# make a sampling model
input2 = Input(shape=(2,)) # we'll only input one word at a time
x = embedding_layer(input2)
x, h, c = lstm(x, initial_state=[initial_h, initial_c]) # now we need states to feed back in
output2 = dense(x)  
sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])


# reverse word2idx dictionary to get back words
# during prediction
word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}
random_word = lambda : word2idx[list(word2idx.keys())[np.random.randint(0, len(word2idx))]]

def sample_line():
  # initial inputs
  np_input = np.array([[ word2idx['<sos>'], random_word()]])
  h = np.zeros((1, LATENT_DIM))
  c = np.zeros((1, LATENT_DIM))

  # so we know when to quit
  eos = word2idx['<eos>']

  # store the output here
  output_sentence = []

  for _ in range(100):
    o, h, c = sampling_model.predict([np_input, h, c])

    # print("o.shape:", o.shape, o[0,0,:10])
    # idx = np.argmax(o[0,0])
    
    probs = o[0,0]
    if np.argmax(probs) == 0:
      print("wtf")
    probs[0] = 0
    probs /= probs.sum()
    idx = np.random.choice(len(probs), p=probs)
    if idx == eos:
      break
  
    # accuulate output
    output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

    # make the next input into model
    np_input[0,0] = idx

  return ' '.join(output_sentence)

# generate a 4 line poem
while True:
  for _ in range(4):
    print(sample_line())

  ans = input("---generate another? [Y/n]---")
  if ans and ans[0].lower().startswith('n'):
    break












