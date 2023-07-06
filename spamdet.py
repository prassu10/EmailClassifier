import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk import NaiveBayesClassifier, classify

import string
import codecs
import random

nltk.download("punkt")
nltk.download("stopwords")

EMAIL_DIR = os.path.join("archive/")
SPAM_DIR = os.path.join(EMAIL_DIR, "spam")
HAM_DIR = os.path.join(EMAIL_DIR, "ham")

stemmer = LancasterStemmer()

def spam_detection(input_mail):
    input_mail=[input_mail]
    
    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)
    
    # making prediction
    prediction = model.predict(input_data_features)
    
    if (prediction[0]==1):
        return 1
    else:
        return 0


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

def trainspam():
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

    # transform the text data to feature vectors that can be used as input to the Logistic regression
    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    # convert Y_train and Y_test values as integers
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    # Training the Model

    # Logistic Regression
    model = LogisticRegression()

    # training the Logistic Regression model with the training data
    model.fit(X_train_features, Y_train)


    # Evaluating the trained model

    # prediction on training data
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    print('Accuracy on training data : ', accuracy_on_training_data)

    # prediction on test data
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

    print('Accuracy on test data : ', accuracy_on_test_data)


# Building a Predictive System

test_mail_list = ["Participate in our new lottery!", 
                  "See the minutes from the last meeting attached", 
                  "Investors are coming to our office on Monday", 
                  "Try out this new medicine",
                  """
                     Subject: confidential folder to safely pass information to arthur andersen
                     we have become increasingly concerned about confidential information ( dpr / position info , curves , validations / stress tests , etc ) being passed to arthur andersen for audit purposes over the web to their arthur andersen email addresses . ( necessary now they no longer have access to enron ' s internal email system )
                     please use the folder described below when passing any info ( that you would have concerns about if it was picked up by a third party ) via the shared drive that has been set up for this specific purpose .
                     note : aa should also use the shared drive to pass info back if there are questions , or the data needs updating . we should also consider the sensitivity of audit findings and special presentations if they are being distributed electronically .
                     please pass this note to others in your groups who have the need to pass info back and forth .
                     details on how to access for those who will use this method to pass info :
                     a secured folder has been set up on the " o " drive under corporate called arthur _ andersen ( o : \ corporate \ arthur _ anderson ) . please post all confidential files in this folder rather than emailing the files to their company email address . if you need access to this folder , submit an erequest through the it central site : http : / / itcentral . enron . com / data / services / securityrequests / . arthur andersen will be able to retrieve these files for review with their terminal server access at the three allen center location .
                     please contact vanessa schulte if you have any problems or questions
                     beth apollo
                  """,
                  """
                     Subject: yukos oil
                     dear friend ,
                     i am mr olsom berghart a personal treasurer to mikhail khodorkovsky the richest man in russia and owner of the following companies : chairman ceo : yukos oil ( russian most largest oil company ) chairman ceo : menatep sbp bank ( a well reputable financial institution with its branches all over the world )
                     source of funds :
                     i have a profiling amount in an excess of us $ 100 , 500 , 000 which i seek your partnership in accommodating for me . you will be rewarded with 4 % of the total sum for your partnership . can you be my partner on this ?
                     introduction of my self
                     as a personal consultant to him , authority was handed over to me in transferring money of an american oil merchant for his last oil deal with my boss mikhail khodorkovsky . already the funds have left the shore of russia to a european private bank where
                     the final crediting is expected to be carried out . while i was on the process , my boss got arrested for his involvement in politics by financing the leading and opposing political parties ( the union of right forces , led by boris nemtsov , and yabloko , a liberal / social democratic party led by gregor yavlinsky ) which poses treat to president vladimir putin second tenure as russian president . you can catch more of the story on the following website :
                     your role :
                     all i need from you is to stand as the beneficiary of the above quoted sum and i will re - profile the funds with your name , which will enable the european bank transfer the sum to you . i have decided to use this sum to relocate to your country and never to
                     be connected to any of mikhail khodorkovsky conglomerates . the transaction has to be concluded before my boss is out from jail . as soon as i confirm your readiness to conclude the transaction with me , i will provide you with the details .
                     thank you very much
                     regards
                     olsom berghart ( mr )
                     mail sent from webmail service at php - nuke powered site
                     - http : / / yoursite . com
                  """
                 ]

for input_mail in test_mail_list:
    result = spam_detection(input_mail)

    if (result==1):
      print('Ham mail')
    else:
      print('Spam mail')
