from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
# import pandas as pd
import googletrans
import os


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model.h5')
vars = 'vars.pkl'
translator = googletrans.Translator()

def generate_sentiment(model, vars, translator, input_text):
    def redummies(x):
        if np.argmax(x) == 0:
            return 'Negative'
        elif np.argmax(x) == 1:
            return 'Neutral'
        else:
            return 'Positive'


    with open('vars.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        MAX_LEN = pickle.load(f)

    input_text = translator.translate(input_text, src="id", dest="en").text
    tokenized_input_text = np.asarray(tokenizer.encode(input_text, add_special_tokens=True)).reshape((1, -1))
    padded_input_text = pad_sequences(tokenized_input_text, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction_result = model.predict(padded_input_text).flatten()
    prediction_result_label = redummies(prediction_result)

    return prediction_result_label


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form:
            input_text = request.form['input_text']
            sentiment = generate_sentiment(model, vars, translator, input_text)

            return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')


# @app.route('/display/<filename>')
# def send_uploaded_image(filename=''):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
