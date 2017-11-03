import numpy as np
import keras
import random
import collections
import time
import os
import operator
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
    vocab = dict([(k.encode('ascii', 'ignore'), model.wv[k].tolist()) for k, v in model.wv.vocab.items()])

    return vocab, sentences

def get_vocab_dicts(vocab):

    words2vecs = vocab
    vecs2words = {}
    for k, v in vocab.items():
        vecs2words[str(v)] = k # vector saved as a string in reverse_dictionary

    return words2vecs, vecs2words

data_path = 'stacked_poems'
vocab, sentence_sequences = create_embeddings(data_path, size=100, min_count=1, window=5, sg=1, iter=25)
dictionary, reverse_dictionary = get_vocab_dicts(vocab)

text = [el for sublist in sentence_sequences for el in sublist]
text_size = len(list(sentence_sequences))
unique_words = len(vocab.keys())

vec_len = 100
seq_len = 15
sentences = []
next_words= []
for i in range(0,text_size-seq_len, 3):
    sentences.append(text[i:i+seq_len])
    next_words.append(text[i+seq_len])

X_train = np.zeros((len(sentences), seq_len, vec_len))
y_train = np.zeros((len(sentences), vec_len))
print(X_train.shape)
print(y_train.shape)
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        X_train[i, j] = dictionary[word]
    y_train[i] = dictionary[next_words[i]]

batch_size = 128
    
model = Sequential()
model.add(LSTM(150, input_shape=(seq_len, vec_len), return_sequences=True, implementation=1))
model.add(Dropout(0.25))
model.add(LSTM(150, input_shape=(seq_len, vec_len), implementation=1))
model.add(Dropout(0.5))
model.add(Dense(vec_len, activation='softmax'))
model.compile(optimizer=keras.optimizers.rmsprop(lr=0.007), loss='categorical_crossentropy')

for it in range(10):

    print('=' * 70)
    print('iteration: {}'.format(it))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    startIndex = np.random.randint(0,text_size-seq_len-1)
    generated = ' '
    sentence = text[startIndex:startIndex+seq_len]
    generated = generated.join(sentence)
    print("generating with seed: \n\n{}".format(generated))
    
    X_eval = np.zeros((1, seq_len, vec_len))
    for j, word in enumerate(sentence):
        X_eval[0, j] = dictionary[word]
    
    prediction = model.predict(X_eval, batch_size=1, verbose=0)[0]
    similarities_dict = {}
    for word, vec in dictionary.items():
        similarity_array = cosine_similarity(np.array(prediction).reshape(1, -1), np.array(vec).reshape(1, -1))
        similarity = similarity_array.tolist()[0][0]
        similarities_dict[word] = similarity

    most_similar_tuple = sorted(similarities_dict.items(), key=operator.itemgetter(1))[-1]
    print(most_similar_tuple) # could take a random word from the top n most similar instead of always taking the best
    next_word = most_similar_tuple[0]
