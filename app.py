import streamlit as st
import random
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from googletrans import Translator
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import spacy
import joblib
nltk.download('stopwords')
python -m spacy download en_core_web_sm

less_informative_words = pd.read_csv('final_words_to_remove_updated.csv')['words_to_remove'].tolist()

stop_words = set(stopwords.words('english'))

# Loading the heavy objects like model, vectorizer, etc. globally to avoid reloading on each request
label_encoder = joblib.load('LabelEncoder.pkl')
scaler = joblib.load('MinMaxScaler.pkl')
vectorize = joblib.load('TfidfVectorizer.pkl')
model = joblib.load('OVO_LR_model.pkl')

class TextProcessor:
    def __init__(self):
        self.translator = Translator()
        self.nlp = spacy.load('en_core_web_sm')

    def text_cleaning_steps_1(self, short_text, long_text):
        text = short_text + ' ' + long_text
        text = text.lower()

        try:
            translated = self.translator.translate(text, src='de', dest='en')
            text = translated.text
        except Exception as e:
            text
        
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' '),
            (r'\S+\.\S+@gmail\.com', ' '),
            (r'\[\s*cid:[^\]]+\]', ' '),
            (r'\d+', ' '),
            (r"[^\w\s']", ' ')
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        text = text.replace('x000d', ' ').replace('\n', ' ').replace('â€', ' ')
        text = text.encode("ascii", "ignore").decode()
        text = ' '.join(text.split()).strip()
        
        return text

    def lemmatize_text(self, text):
        return [token.lemma_.lower() for token in self.nlp(text) if token.text.strip()]

    def remove_stopwords(self, lemmas):
        return [token for token in lemmas if token not in stop_words]

    def removing_noise(self, tokens):
        return [token for token in tokens if len(token) > 1 and token not in less_informative_words]

    def additional_cleaning(self, tokens):
        tokens = ['abended' if token == 'evening' else token for token in tokens]
        return ' '.join(set(tokens))

    def process_text(self, short_text, long_text):
        cleaned_text = self.text_cleaning_steps_1(short_text, long_text)
        lemmas = self.lemmatize_text(cleaned_text)
        tokens = self.remove_stopwords(lemmas)
        tokens = self.removing_noise(tokens)
        final_text = self.additional_cleaning(tokens)
        print('The text has been cleaned')
        return final_text


def predict_label(text_length, cleaned_text, scaler, vectorize, model, label_encoder):
    test_description_dense = np.hstack((
        vectorize.transform([cleaned_text]).toarray(),
        scaler.transform(np.array([text_length]).reshape(-1, 1))
    ))
    pred = model.predict(test_description_dense)
    return label_encoder.inverse_transform(pred)


def generate_ticket_number():
    return f"IN{random.randint(100000, 999990)}"

def main():
    if "ticket_number" not in st.session_state:
        st.session_state.ticket_number = generate_ticket_number()

    st.title("Ticketing Tool - Assignment Group Prediction")
    st.subheader(f"Ticket Number: {st.session_state.ticket_number}")

    with st.form(key='ticket_form'):
        employee_id = st.text_input("**Employee ID (6-digit number)**")
        location = st.text_input("**Location (Country)**")
        short_description = st.text_input("**Short Description**")
        long_description = st.text_area("**Long Description**")
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        if not employee_id.isdigit() or len(employee_id) != 6:
            st.error("Please enter a valid 6-digit numerical Employee ID.")
        elif not employee_id or not location or not short_description or not long_description:
            st.error("Please fill all the required fields.")
        else:
            processor = TextProcessor()
            cleaned_text = processor.process_text(short_description, long_description)
            text_length = len(cleaned_text)
            predicted_label = predict_label(text_length, cleaned_text, scaler, vectorize, model, label_encoder)[0]
            
            data = {
                "Ticket Number": [st.session_state.ticket_number],
                "Employee ID": [employee_id],
                "Location": [location],
                "Short Description": [short_description],
                "Long Description": [long_description],
                "Predicted Assignment Group": [predicted_label]
            }
            df = pd.DataFrame(data)
            st.success(f"Predicted Assignment group for ticket {st.session_state.ticket_number}: {predicted_label}")
            st.table(df)


if __name__ == "__main__":
    main()
