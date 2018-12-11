import re
import string
import numpy as np
from csv import DictReader
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import train_test_split

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
		dataset = []
		# with open('static/PercakapanV6.csv', 'r') as file:
		#     reader = DictReader(file, delimiter=';')
		#     for row in reader:
		# #        print('row in dict')
		# #        print(row)
		#         if row['tag_final'] == '' and row['message'] !='start' and row['message'] !='end' :
		#             dataset[len(dataset)-1]['response'] = row['message']
		#         elif row['message'] !='start' and row['message'] !='end':
		#             dataset.append(
		#                     {
		#                         'message': row['message'],
		#                         'category' : row['tag_final']
		#                     }
		#                 )
		# print(dataset)
		# self.txtpreprocessing(dataset)
		# shuffle(dataset)
		# print(dataset)

		with open('static/data_percakapan.json') as json_file:
			dataset = json.load(json_file)

		datatrain_index = round(len(dataset)*8/10)
		datatrain = dataset[:datatrain_index]
		datatest  = dataset[datatrain_index:]

		datatrain, datatest = train_test_split(dataset, test_size=0.2)
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform([row['message'] for row in datatrain])
		X_train_counts.shape

		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		X_train_tfidf.shape

		# print(X_train_tfidf)

		text_clf_svm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', svm.SVC(C=1, gamma=1.0)),])

		text_clf_svm.fit([row['message'] for row in datatrain], [row['category'] for row in datatrain])
		prediksi = text_clf_svm.predict([inputtxt['inputtxt']])
		print(prediksi)
		acc_dataset = []
		for x in dataset:
		    if x['category'] == prediksi:
		#         dataset_msg.append(x['message'])
		#         print(get_sim("apakah harga samsung naik", x['message']), x['response'])
		        acc_dataset.append(list([self.get_sim(inputtxt['inputtxt'], x['message']), x['response']]))

		print(inputtxt['inputtxt'])
		response_result = max(node for node in acc_dataset)
		return response_result[1]



	# jaccard similarity
	def get_sim (self, str1, str2):
	    a = set(str1.split())
	    b = set(str2.split())
	    c = a.intersection(b)
	    return float(len(c)) / (len(a) + len(b) - len(c))