# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import joblib
import pickle
from joblib import load

#import package untuk phishing detection
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

#import package untuk safebot
from keras.models import load_model
import random
import string
import tensorflow as tf
import json
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load data json
# data = json.loads(open('safebot.json').read())

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# Text Pre-Processing function untuk API phishing detection
def text_preprocessing_process(text):
    tokens = tokenizer.tokenize(text)
    tokens_stemmed = [stemmer.stem(token) for token in tokens]
    processed_text = ' '.join(tokens_stemmed)
    return processed_text

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# Routing for API phishing
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    prediction_input = ""
    
    if request.method == 'POST':
        prediction_input = request.form['data']
        
        # Text Pre-Processing
        prediction_input = text_preprocessing_process(prediction_input)
        
        # Vectorization
        feature = cv.transform([prediction_input])
        
        # Prediction (Web Phishing or Web Aman)
        prediction = model.predict(feature)
        
        if prediction == 0:
            hasil_prediksi = "Web Phishing"
        else:
            hasil_prediksi = "Web Aman"
        
        # Return the prediction result in JSON format
        return jsonify({
            "data": hasil_prediksi,
        })
    
# Routing for API safebot
# @app.route("/api/response", methods=['POST'])
# def apiResponse():
#     prediction_input = ""
#     text_p= []
    
#     if request.method == 'POST':
#         prediction_input = request.form['data']
        
#         # Menghapus punktuasi dan konversi ke huruf kecil
#         prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
#         prediction_input = ''.join(prediction_input)
#         text_p.append(prediction_input)
        
#         # Tokenisasi dan Padding
#         prediction_input = tokenizer.texts_to_sequences(text_p)
#         prediction_input = np.array(prediction_input).reshape(-1)
#         prediction_input = pad_sequences([prediction_input],input_shape)

#         # Mendapatkan hasil keluaran pada model 
#         output = model.predict(prediction_input)
#         output = output.argmax()
        
#         # Menemukan respon sesuai data tag 
#         response_tag = le.inverse_transform([output])[0]
#         response = random.choice(responses[response_tag])
        
#         # Return the prediction result in JSON format
#         return jsonify({
#             "data": response,
#         })

# =[Main]========================================

if __name__ == '__main__':
        
    #Setup Phishing
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	stemmer = SnowballStemmer("english")
   	
	cv = CountVectorizer(vocabulary=pickle.load(open('feature.pickle', 'rb')))
	
	# Load model phishing yang telah ditraining
	model = load('model_phishing_lr.model')



    #Setup Safebot

    # Load model phishing yang telah ditraining
	# model = load_model('model_chatbot_lstm.model') 

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)