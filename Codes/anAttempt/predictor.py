import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils.np_utils as ku
import keras.utils as ku 
import pandas as pd
import tensorflow as tf

# loading tokenized data 
import pickle

with open("/Users/Claire/Documents/NUS/BZA/Y4S2/BT4222/anAttempt/tokenizer_LSTM.pkl","rb") as f:
    tokenizer_1 = pickle.load(file=f)
    new_model = tf.keras.models.load_model("/Users/Claire/Documents/NUS/BZA/Y4S2/BT4222/anAttempt/model_final")

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer_1.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted,axis=1)       

        # predict_x=model.predict(X_test) 
        # classes_x=np.argmax(predict_x,axis=1)

        output_word = ""
        for word,index in tokenizer_1.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


lstm_ft = tf.keras.models.load_model("/Users/Claire/Documents/NUS/BZA/Y4S2/BT4222/anAttempt/lstm-fasttext")


def predict(sentence):
  word_tokenizer = Tokenizer(num_words=7577,
                           lower=True,
                           split=" ",
                           char_level=False) 

  word_tokenizer.fit_on_texts(sentence)
  sentence_seq = word_tokenizer.texts_to_sequences(sentence)

  sentence_processed = pad_sequences(sequences= sentence_seq, maxlen=15, padding='post', truncating='post')
  result = lstm_ft.predict(sentence_processed)
  #print(result)

  if result[0][0] > result[0][1]:
    return "This post is neutral."
  elif result[0][1] > result[0][0] and result[0][1] >= 0.8:
    return "You may have written a hate speech. Are you sure you want to post it?"



def main():
    st.title("Sentence Prediction")
    
    html_temp = """
        <div style ="background-color:blue;padding:13px">
        <h1 style ="color:black;text-align:center;">Hate Speech Deterrence App </h1>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html = True)
 

    context = st.text_input("contextual words", "Type Here")
    prediction_length = int(st.number_input("Type Here"))
    result = ""
    prediction = ''
    publish = ""

    if st.button("Predict"):
        result = generate_text(context, prediction_length,new_model, 28)
        prediction = predict([result])
        publish = result + ": " + prediction
    st.success('The output is {}'.format(publish))

if __name__ =='__main__':
    main()

