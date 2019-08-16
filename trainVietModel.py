import os
import tensorflow as tf
import numpy as np
import argparse
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint

def checkCorpus(string):
    currentDir = os.listdir()
    if (string in currentDir and os.path.isfile(string)):
        return string
    else:
        # print("No folder named %s" % string)
        return -1

parser = argparse.ArgumentParser()
parser.add_argument('-corpus', dest='corpus', type=checkCorpus, default='VNESEcorpus.txt')
parser.add_argument('-epochs', dest='epochs', type=int, default=500)
parser.add_argument('-seq_length', dest='seq_length', type=int, default=15)
args = parser.parse_args()

corpusFile = args.corpus
corpusSequenceFile = corpusFile[:-4] + '_' + 'char_sequences.txt'
seq_length = args.seq_length
epochs = args.epochs

# load doc into memory
def load_data(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_data(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

if (not os.path.exists(corpusSequenceFile)):
    # load text
    raw_text = load_data(corpusFile)
    # print(raw_text)

    # clean
    tokens = raw_text.split()
    raw_text = ' '.join(tokens)

    # organize into sequences of characters
    length = seq_length
    sequences = list()
    for i in range(length, len(raw_text)):
    	# select sequence of tokens
    	seq = raw_text[i-length:i+1]
    	# store
    	sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    save_data(sequences, corpusSequenceFile)

# load
raw_data = load_data(corpusSequenceFile)
lines = raw_data.split('\n')

chars = sorted(list(set(raw_data)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
# model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# continue checkpoint
checkpoint = ModelCheckpoint('savedEpochs/model-epoch-{epoch:03d}.h5', period=5)
listEpochs = [x for x in os.listdir('savedEpochs/') if x[:12] == 'model-epoch-' and x[-3:] == '.h5']

if (len(listEpochs) > 0):
    lastEpoch = max([int(x[12:-3]) for x in listEpochs])
    lastEpochFile = 'savedEpochs/model-epoch-%03d.h5' % lastEpoch
    # load weights
    model.load_weights(lastEpochFile)
    print("CONTINUE TRAINING FROM %03d EPOCH......" % lastEpoch)
else:
    lastEpoch = 0

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, y, epochs=epochs, initial_epoch = lastEpoch, validation_split=0.1, callbacks=[checkpoint])

# save the model to file
model.save('viet-lang-model.h5')

# save the mapping
pickle.dump(mapping, open('viet-lang-mapping.pkl', 'wb'))
