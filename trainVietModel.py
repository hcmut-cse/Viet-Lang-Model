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
parser.add_argument('-part_size', dest='part_size', type=int, default=50000)
parser.add_argument('-checkpoint_period', dest='checkpoint_period', type=int, default=5)
args = parser.parse_args()

corpusFile = args.corpus
corpusSequenceFile = corpusFile[:-4] + '_' + 'char_sequences.txt'
seq_length = args.seq_length
epochs = args.epochs
part_size = args.part_size
period = args.checkpoint_period

# load doc into memory
def load_data(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_data(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding='utf8')
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

    # Delete data to save memory
    del(sequences)
    del(tokens)
    del(raw_text)

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

# Delete data to save memory
del(lines)
del(raw_data)
del(mapping)
del(chars)

sequences = np.array(sequences)
X_train, y_train = sequences[:,:-1].copy(), sequences[:,-1].copy()

del(sequences)

input_shape = (seq_length, vocab_size)
current_part = 0
max_part = int(len(X_train) / part_size) + 1
lastEpoch = 0
if (os.path.exists('savedEpochs/current_part.txt')):
    with open('savedEpochs/current_part.txt', 'r', encoding='utf8') as f:
        current_part = int(f.read())

# define model
model = Sequential()
# model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if (os.path.exists('savedEpochs/part_%d' % current_part)):
    listEpochs = [x for x in os.listdir('savedEpochs/part_%d' % current_part) if x[:12] == 'model-epoch-' and x[-3:] == '.h5']

    if (len(listEpochs) > 0):
        lastEpoch = max([int(x[12:-3]) for x in listEpochs])
        lastEpochFile = 'savedEpochs/part_%d/model-epoch-%03d.h5' % (current_part, lastEpoch)
        # load weights
        model.load_weights(lastEpochFile)
        print("CONTINUE TRAINING FROM PART %d EPOCH %03d......" % (current_part, lastEpoch))
    else:
        lastEpoch = 0

def constrain(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

for i in range(current_part, max_part):
    with open('savedEpochs/current_part.txt', 'w', encoding='utf8') as f:
        f.write(str(i))

    print("====================================================================")
    print("=                       TRAINING PART %03d                         =" % i)
    print("====================================================================")

    if (not os.path.exists('savedEpochs/part_%d' % i)):
        os.mkdir('savedEpochs/part_%d' % i)

    if (i > current_part):
        lastEpoch = 0

    start_point = i * part_size
    end_point = (i + 1) * part_size
    end_point = constrain(end_point, 0, len(X_train))

    X = np.array([to_categorical(x, num_classes=vocab_size) for x in X_train[start_point:end_point]])
    y = to_categorical(y_train[start_point:end_point], num_classes=vocab_size)

    # continue checkpoint
    checkpoint = ModelCheckpoint('savedEpochs/part_%d/model-epoch-{epoch:03d}.h5' % i, period=period)

    # fit model
    model.fit(X, y, epochs=epochs, initial_epoch = lastEpoch, validation_split=0.1, callbacks=[checkpoint])


# save the model to file
model.save('viet-lang-model.h5')

# save the mapping
pickle.dump(mapping, open('viet-lang-mapping.pkl', 'wb'))
