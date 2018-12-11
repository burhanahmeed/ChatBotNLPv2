import re
import string
import numpy as np
from csv import DictReader
from nltk.tokenize import word_tokenize
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from keras.models import model_from_json
from keras.preprocessing import text, sequence
from keras import backend as K
from sklearn.model_selection import train_test_split

import tensorflow as tf

import json

class preprocessing:
	# def txtpreprocessing(self, dataset):
	# 	stemmer = StemmerFactory().create_stemmer()
	# 	stopwords = StopWordRemoverFactory().create_stop_word_remover()
	# 	for row in dataset:
	# 		row['message'] = row.get('message').casefold()
	# 		row['message'] = re.sub(r"[0-9]", "", row.get('message'))
	# 		row['message'] = re.sub('['+string.punctuation+']', "", row.get('message'))
	# 		row['message_stopwords'] = stopwords.remove(row['message'])
	# 		row['message_stemmed'] = stemmer.stem(row['message_stopwords'])
	# 		row['message_tokenized'] = word_tokenize(row['message_stemmed'])


	def predict(self, **inputtxt):
		np.random.seed(2017)
		tf.set_random_seed(2017)		
		dataset = []

		with open('static/data_percakapan.json') as json_file:
			dataset = json.load(json_file)

		labels_set = []
		for i in dataset:
			labels_set.append(i['category'])

		json_file = open('static/modelChatbot.json', 'r')
		loaded_model_json = json_file.read()
		# json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("static/modelChatbot.h5")

		encoder = LabelEncoder()
		encoder.fit(labels_set)		
		text_labels = encoder.classes_
		# print(text_labels)
		max_words = 100
		tokenize = text.Tokenizer(num_words=max_words, char_level=False)		
		txt_step1 = inputtxt['inputtxt'].lower()
		txt_step2 = re.sub(r"[ ](?=[ ])|[^-_,A-Za-z0-9 ]+", "", txt_step1)
		tokenize.fit_on_texts([txt_step2])
		tok_res = tokenize.texts_to_matrix([txt_step2])
		prediction = loaded_model.predict(np.array(tok_res))
		prediksi = text_labels[np.argmax(prediction)]

		print(prediksi)

		acc_dataset = []
		for x in dataset:
		    if x['category'] == prediksi:
		#         dataset_msg.append(x['message'])
		#         print(get_sim("apakah harga samsung naik", x['message']), x['response'])
		        acc_dataset.append(list([self.get_sim(inputtxt['inputtxt'], x['message']), x['response']]))

		print(txt_step2)
		response_result = max(node for node in acc_dataset)
		K.clear_session()
		return response_result[1]



	# jaccard similarity
	def get_sim (self, str1, str2):
	    a = set(str1.split())
	    b = set(str2.split())
	    c = a.intersection(b)
	    return float(len(c)) / (len(a) + len(b) - len(c))