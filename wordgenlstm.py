import numpy as np
import keras
import random
import time
import os
import operator
import sys
import string
from collections import Counter
import keras as K
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Embedding, CuDNNLSTM, Dense, Dropout
from sklearn.metrics.pairwise import cosine_similarity

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
# print("GPU: ", get_gpu_name())
# print("CUDA Version: ", get_cuda_version())
# print("CuDNN Version: ", get_cudnn_version())

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
words = [word for word in stripped if word.isalpha()]

min_occurences = 5
counts = Counter(words)
words = [word for word in words if counts[word] >= min_occurences]
freq_map = dict(sorted(Counter(words).items(), key=lambda x: x[1], reverse=True))
index_map = dict([(word, i) for i, word in enumerate(freq_map.keys())])
reverse_index_map = dict([(v, k) for k, v in index_map.items()])
num_words = len(list(set(words)))
text_size = len(words)

seq_len = 25
sentences = []
next_words= []
for i in range(0,text_size-seq_len, 1): # was going by 3
    sentences.append(words[i:i+seq_len])
    next_words.append(words[i+seq_len])

X_train = np.zeros((len(sentences), seq_len), dtype=np.int64)
y_train = np.zeros((len(sentences), num_words), dtype=np.int64)

for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        X_train[i, j] = index_map[word]
    y_train[i, index_map[next_words[i]]] = 1

embedding_vector_length = 256
batch_size = 64

model = Sequential()
model.add(Embedding(num_words, embedding_vector_length, input_length=seq_len))
model.add(CuDNNLSTM(512))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_words, activation='softmax'))
model.compile(optimizer=keras.optimizers.rmsprop(), loss='categorical_crossentropy')

print("Constructing Model with: TEXT SIZE: {}, NUM DISTINCT WORDS: {}, SEQUENCE LENGTH: {}."\
    .format(text_size, num_words, seq_len))

for it in range(500):

    print('=' * 70)
    print('\nITERATION: {}'.format(it))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    startIndex = np.random.randint(0,text_size-seq_len-1)
    sentence = words[startIndex:startIndex+seq_len]

    X_eval = np.zeros((1, seq_len), dtype=np.int64)
    for j, word in enumerate(sentence):
        X_eval[0, j] = index_map[word]

    prediction = model.predict(X_eval, batch_size=batch_size,  verbose=0)
    next_word = reverse_index_map[prediction.argmax()]
    print('\nSENTENCE: {}'.format(' '.join(sentence)))
    print('\nNEXT WORD: {}'.format(next_word))
