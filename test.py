from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[1], encoded.shape[2])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('viet-lang-model.h5')
# load the mapping
mapping = load(open('viet-lang-mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 15, 'cổng thông ', 20))
# test mid-line
print(generate_seq(model, mapping, 15, 'bảo hiểm xã ', 20))
# test not in original
print(generate_seq(model, mapping, 15, 'hành vi vi ', 20))