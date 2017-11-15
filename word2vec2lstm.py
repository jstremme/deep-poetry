import numpy as np
import keras
import random
import collections
import time
import os
import operator
import sys
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(data_dir, **params):

    tokenize = lambda x: simple_preprocess(x)

    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenize(line.encode('ascii', 'ignore'))

    sentences = SentenceGenerator(data_dir)
    model = Word2Vec(sentences, **params)
    words2vecs = dict([(k, model.wv[k]) for k, v in model.wv.vocab.items()])

    return words2vecs, sentences

data_path = 'stacked_poems'
words2vecs, sentence_sequences = create_embeddings(data_path, size=300, min_count=1, window=7, sg=1, iter=25)
text = [str(el) for sublist in sentence_sequences for el in sublist]
text_size = len(list(sentence_sequences))
unique_words = len(words2vecs.keys())

vec_len = 300
seq_len = 15
sentences = []
next_words= []
for i in range(0,text_size-seq_len, 1): # was going by 3
    sentences.append(text[i:i+seq_len])
    next_words.append(text[i+seq_len])

X_train = np.zeros((len(sentences), seq_len, vec_len))
y_train = np.zeros((len(sentences), vec_len))
print(X_train.shape)
print(y_train.shape)

for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        X_train[i, j] = words2vecs[word]
    y_train[i] = words2vecs[next_words[i]]

batch_size = 128
    
model = Sequential()
model.add(LSTM(150, input_shape=(seq_len, vec_len), return_sequences=True, implementation=1))
model.add(Dropout(0.25))
model.add(LSTM(150, input_shape=(seq_len, vec_len), implementation=1))
model.add(Dense(vec_len, activation='softmax'))
model.compile(optimizer=keras.optimizers.rmsprop(lr=0.007), loss='categorical_crossentropy')

for it in range(50):

    print('=' * 70)
    print('iteration: {}'.format(it))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    startIndex = np.random.randint(0,text_size-seq_len-1)
    generated = ' '
    sentence = text[startIndex:startIndex+seq_len]
    generated = generated.join(sentence)
    print("generating with seed: \n\n{}".format(generated))
    
    for w in range(0, 20):

        X_eval = np.zeros((1, seq_len, vec_len))
        for j, word in enumerate(sentence):
            X_eval[0, j] = words2vecs[word]
        
        prediction = model.predict(X_eval, batch_size=batch_size,  verbose=0)
        similarities_dict = {}
        for word, vec in words2vecs.items():
            similarity_array = cosine_similarity(prediction, np.array(vec).reshape(1, -1))
            similarity = similarity_array.tolist()[0][0]
            similarities_dict[word] = similarity

        most_similar_tuple = sorted(similarities_dict.items(), key=operator.itemgetter(1))[-1]
        print(most_similar_tuple) 
        next_word = most_similar_tuple[0]

        generated += next_word
        sentence = sentence[1:]
        sentence.append(next_word)

    print(sentence)
