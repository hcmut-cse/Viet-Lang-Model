from pickle import load
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import argparse
import underthesea
import numpy as np
import re

# Set GPU allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

SEQ_LENGTH = 30
CORRECT_THRESHOLD = 1e-3

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    newString = re.sub("[^a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=2:
            long_words.append(i)
    return (" ".join(long_words)).strip()

def encode_string(mapping, seq_length, in_text):
	# encode the characters as integers
	encoded = [mapping[char] for char in in_text]

	# truncate sequences to a fixed length
	encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

	# one hot encode
	# encoded = to_categorical(encoded, num_classes=len(mapping))
	# encoded = encoded.reshape(1, encoded.shape[1], encoded.shape[2])

	return encoded

def decode_string(mapping, in_text):
	out_text = ""
	for i in range(len(in_text)):
		for char, index in mapping.items():
			if index == in_text[i]:
				out_text += char
				break
	return out_text

def insert(source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

def replace(source_str, insert_str, start_pos):
	source_list = list(source_str)
	if (start_pos > len(source_list)):
		return source_str
	for i in range(len(insert_str)):
		source_list[start_pos + i] = insert_str[i]
	return ''.join(source_list)

# load the model
model = load_model('viet-lang-model.h5')

# load the mapping
mapping = load(open('viet-lang-mapping.pkl', 'rb'))

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# Define function to get
print("Number of layers: %d" % len(model.layers))
# get_1st_lstm_output = K.function([model.layers[0].input], [model.layers[0].output])
# get_2nd_lstm_output = K.function([model.layers[1].input], [model.layers[1].output])
# get_proba_output = K.function([model.layers[2].input], [model.layers[3].output])

# generate a sequence of characters with a language model
def correct_mistake(model, mapping, seq_length, seed_text):
	# in_text = text_cleaner(seed_text)
	# in_text = encode_string(mapping, seq_length, in_text)

	# char = model.predict_classes(in_text)
	# print(char)
	# # in_text = in_text + [char]
	# char = decode_string(mapping ,char)
	# print(char)
	# return ''

	

	in_text = text_cleaner(seed_text)
	# in_text = encode_string(mapping, seq_length, in_text)
	
	# out_text = in_text[0]
	# out_text_predict = ''
	# for i in range(len(in_text)):
	# 	out_text_predict += in_text[i]
	# 	out_text_predict_encode = encode_string(mapping, seq_length , out_text_predict)
	# 	if( i > seq_length):
	# 		out_text_predict = out_text_predict[1:]
	# 	proba_list_char = model.predict_proba(out_text_predict_encode)
	# 	next_char = model.predict_classes(out_text_predict_encode)
		
		# if((i+1 <= len(in_text)-1) and int(next_char) != mapping[in_text[i+1]]):
		# 	print(proba_list_char[0][mapping[in_text[i+1]]])
		# 	if(proba_list_char[0][mapping[in_text[i+1]]] > 1e-10):
		# 		out_text += in_text[i]
		# 	else:
		# 		out_text += decode_string(mapping , next_char)
		# else:
		# 	out_text+=in_text[i]
	# print(out_text)
	# return ''	

	out_text = in_text[0:4]
	out_text_predict = in_text[0:4]
	flag = True
	i = 4
	# for i in range(12,len(in_text)):
	
	while flag:
		out_text_predict_encode = encode_string(mapping , seq_length , out_text_predict)
		
			
		
		proba_list_char = model.predict_proba(out_text_predict_encode)
		next_char = model.predict_classes(out_text_predict_encode)
		# print(proba_list_char[0][mapping[in_text[i]]])
		# print(proba_list_char[0][mapping[decode_string(mapping,next_char)]])
		if((i+1 <= len(in_text)-1) and int(next_char) != mapping[in_text[i]]):
			if(proba_list_char[0][mapping[in_text[i]]] > CORRECT_THRESHOLD ):
				out_text += in_text[i]
			else:
				out_text += decode_string(mapping , next_char)
				if(decode_string(mapping, next_char) == ' ' and in_text[i] != ' '):
					temp = in_text[i:]
					in_text = in_text[0:i]
					in_text = in_text + ' ' + temp
		else:
			out_text+=in_text[i]
		
		

		if(i < len(in_text)-1):
			out_text_predict += in_text[i]
			i = i + 1
		else:
			flag = False
		
		if(i > seq_length):
			out_text_predict = out_text_predict[1:]


	print(out_text)
	
	return ''







	


	
	
	

	
		


				



	

	# in_text = text_cleaner(seed_text)
	# # print(seed_text)
    # out_text = ""

    # in_text_encoded = encode_string(in_text)
    # for i in range(len(in_text)):
        


		
	







	# text_pos = list(range(len(in_text)))
	# for end_pos in text_pos:
	# 	if (end_pos == 0):
	# 		continue

	# 	encoded = encode_string(mapping, seq_length, in_text[:end_pos])
	# 	# print(encoded.shape)

	# 	# Define get hidden layer output function
	# 	# lstm_1st = get_1st_lstm_output([encoded])
	# 	# print(lstm_1st)
	# 	# print(lstm_1st[0].shape)
	# 	# lstm_2nd = get_2nd_lstm_output(lstm_1st)
	# 	# print(lstm_2nd)
	# 	# print(lstm_2nd[0].shape)
	# 	# proba_output = get_proba_output(lstm_2nd)[0][0].tolist()
	# 	# print(proba_output)
	# 	# print(proba_output[0].shape)

	# 	proba_output = model.predict_proba(encoded)[0].tolist()

	# 	next_char_encoded = mapping[in_text[end_pos]]
	# 	next_char_proba = proba_output[next_char_encoded]
	# 	# next_char_decoded = in_text[end_pos]

	# 	predict_char_proba = max(proba_output)
	# 	predict_char_encoded = proba_output.index(predict_char_proba)
	# 	predict_char_decoded = decode_string(mapping, [predict_char_encoded])

	# 	# print("==================================")
	# 	# print("\"%s\" proba (Original): %f" % (in_text[end_pos], next_char_proba))
	# 	# print("\"%s\" proba (Predicted): %f" % (predict_char_decoded, predict_char_proba))
	# 	# print("==================================")

	# 	# if (predict_char_decoded == ' ' and in_text[end_pos] != ' ' and predict_char_proba > CORRECT_THRESHOLD and end_pos > 1 and next_char_proba < CORRECT_THRESHOLD):
	# 	# 	if (len(seed_text) - end_pos < 3):
	# 	# 		continue
	# 	# 	in_text = insert(in_text, ' ', end_pos)
	# 	# 	text_pos.append(len(text_pos))
	# 	# elif (predict_char_proba > next_char_proba and predict_char_proba > CORRECT_THRESHOLD and next_char_proba < CORRECT_THRESHOLD):
	# 	# 	in_text = replace(in_text, predict_char_decoded, end_pos)

	#return in_text

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--f', dest='file', type=str, help="Input file for correcting.", default="")
	args = parser.parse_args()

	file = args.file
	if (file):
		with open(file, 'r', encoding='utf8') as f:
			text_data = f.read()

		# text_data = text_data.replace('\n', ' ')
		# token_words = underthesea.word_tokenize(text_data)
		#
		# print("Processing...")
		# for i, words in enumerate(token_words):
		# 	corrected_word = correct_mistake(model, mapping, 15, words)
		# 	token_words[i] = corrected_word
		# 	print(words + '\t===>>>\t' + corrected_word)
		# 	# if (i%50 == 0):
		# 	# 	print("...")
		#
		# with open(file[:-4] + '_corrected' + file[-4:], 'w', encoding='utf8') as f:
		# 	for word in token_words:
		# 		f.write(word + ' ')

		text_corrected = correct_mistake(model, mapping, SEQ_LENGTH, text_data)
		with open(file[:-4] + '_corrected' + file[-4:], 'w', encoding='utf8') as f:
			f.write(text_corrected)

		print("Finished correcting and saved.")
	else:
		while True:
			inputText = input('Input: ')
			if (inputText == 'Exit'): break
			# while inputText[-1:] != ' ':
			inputText = correct_mistake(model, mapping, SEQ_LENGTH, inputText)
			print("Output: " + inputText)
