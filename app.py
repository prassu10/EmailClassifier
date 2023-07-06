import os
import string
import codecs
import random



import numpy as np
import pandas as pd
import streamlit as st



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk import NaiveBayesClassifier, classify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer



nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download("punkt")
nltk.download("stopwords")



# Spam Detection
def get_emails_list(file_dir, tag, proportion=1):
    files = os.listdir(file_dir)
    files_length = int(len(files)*proportion)
    files = files[:files_length]
    tag_list = []
    for a_file in files:
        if not a_file.startswith("."):
            with codecs.open(os.path.join(file_dir, a_file), "r", encoding="ISO-8859-1", errors="ignore") as f:
                email = f.read()
        tag_list.append((email, tag))
    return tag_list


# getting emails
EMAIL_DIR = os.path.join("archive/")
SPAM_DIR = os.path.join(EMAIL_DIR, "spam")
HAM_DIR = os.path.join(EMAIL_DIR, "ham")

stemmer = LancasterStemmer()

spam_list = get_emails_list(SPAM_DIR, "spam", 1)
ham_list = get_emails_list(HAM_DIR, "ham", 1)
email_list = spam_list + ham_list

email_df = pd.DataFrame(email_list)

email_df.rename(columns = {0:'message'}, inplace = True)
email_df.rename(columns = {1:'category'}, inplace = True)

email_df.loc[email_df['category'] == 'spam', 'category',] = 0
email_df.loc[email_df['category'] == 'ham', 'category',] = 1

X = email_df['message']
Y = email_df['category']

# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Training the Model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluating the trained model
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


def spam_detection(input_mail):
    input_mail=[input_mail]
    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)
    # making prediction
    prediction = model.predict(input_data_features)
    print(int(prediction),end=' ')
    if (prediction[0]==1):
        return 1
    else:
        return 0



# Tone Detection
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        return -1
    elif score['neg'] < score['pos']:
        return 1
    else:
        return 0

def tone_detection(input_mail):
    text = input_mail
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")
    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)
    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(': ')
            if word in lemma_words:
                emotion_list.append(emotion)
    if not emotion_list:
        return 0
    else:
        return sentiment_analyse(cleaned_text)



# Action Detection
def action_analysis(action_list):
    if 1 in action_list and 0 in action_list and -1 in action_list:
        return 5
    elif 1 in action_list and 0 in action_list:
        return 2
    elif 1 in action_list and -1 in action_list:
        return 3
    elif 0 in action_list and -1 in action_list:
        return 4
    elif 1 in action_list:
        return 1
    elif 0 in action_list:
        return 0
    elif -1 in action_list:
        return -1

def action_detection(input_mail):
    text = input_mail
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")
    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        final_words.append(word)
    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
    action_list = []
    with open('action.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, action = clear_line.split(': ')
            word, action = word.lower(), action.lower()
            if word in lemma_words:
                if action == 'reply':
                    action_list.append(1)
                elif action == 'forward':
                    action_list.append(-1)
                elif action == 'neutral':
                    action_list.append(0)
    if not action_list:
        return 0
    else:
        return action_analysis(action_list)



ps = PorterStemmer()

with open("style.css") as source_design:
    st.markdown(f"<style>{source_design.read()}</style>", unsafe_allow_html=True)

st.title("Email Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    result=spam_detection(input_sms)
    if result == 1:
        st.subheader("Not Spam Mail")
    elif result == 0:
        st.subheader("Spam Mail")
    else:
        st.subheader("Nat Spam Mail")


    result = tone_detection(input_sms)
    if result == 1:
        st.subheader("Positive Sentiment")
    elif result == 0:
        st.subheader("Neutral Sentiment")
    else:
        st.subheader("Negative Sentiment")
        

    result = action_detection(input_sms)
    if result == 1:
        st.subheader("Reply Mail")
    elif result == 0:
        st.subheader("Neutral Mail")
    elif result == -1:
        st.subheader("Forward Mail")
    elif result == 5:
        st.subheader("Neutral Mail, Reply Mail, Forward Mail")
    elif result == 4:
        st.subheader("Neutral Mail, Forward Mail")
    elif result == 3:
        st.subheader("Forward Mail, Reply Mail")
    elif result == 2:
        st.subheader("Neutral Mail, Reply Mail")
