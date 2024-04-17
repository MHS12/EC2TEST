import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# NLP libraries to clean the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
# Vectorization technique TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# For Splitting the dataset
from sklearn.model_selection import train_test_split
# Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from django.conf import settings
# Accuracy measuring library
from sklearn.metrics import accuracy_score
import neattext.functions as nxf
from nltk.tokenize import word_tokenize

path = settings.MEDIA_ROOT + "//" + 'tweets_combined.csv'
data = pd.read_csv(path)

def Cleaning_Data():
    path = settings.MEDIA_ROOT + "//" + 'tweets_combined.csv'
    data = pd.read_csv(path)
    data['clean_data'] = data['tweet'].apply(nxf.remove_userhandles)
    data['clean_data'] = data['clean_data'].apply(nxf.remove_hashtags)
    data['clean_data'] = data['clean_data'].apply(nxf.remove_emojis)
    data['clean_data'] = data['clean_data'].apply(nxf.remove_multiple_spaces)
    data['clean_data'] = data['clean_data'].apply(nxf.remove_special_characters)
    data['clean_data'] = data['clean_data'].apply(stop_words_cleaning)
    data=data[['sno','tweet', 'clean_data', 'target']]
    return data


def stop_words_cleaning(text):
  text=text.lower()
  word_tokens = word_tokenize(text)
  stop_words = stopwords.words('english')
  fi_s=[w for w in word_tokens if not w in stopwords.words('english') ]
  text = ' '.join(fi_s)
  return text


data=Cleaning_Data()
X= data['clean_data']
Y = data['target']

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


def process_SVM():
    # 2. Support Vector Machine(SVM) - SVM works relatively well when there is a clear margin of separation between classes.
    svm_model = SVC(kernel='linear')
    # Fitting training set to the model
    svm_model.fit(xv_train, y_train)
    # Predicting the test set results based on the model
    svm_y_pred = svm_model.predict(xv_test)
    # Calculate the accuracy score of this model
    score = accuracy_score(y_test, svm_y_pred) 
    svm_report = classification_report(y_test, svm_y_pred, output_dict=True)
    print('Accuracy of SVM model is ', score)
    return score, svm_report





def process_naiveBayes():
    # 3. Naive Bayes
    nb_model = GaussianNB()
    # Fitting training set to the model
    nb_model.fit(xv_train.toarray(), y_train)
    # Predicting the test set results based on the model
    nb_y_pred = nb_model.predict(xv_test.toarray())
    # Calculate the accuracy score of this model
    nb_acc = accuracy_score(y_test, nb_y_pred)
    nb_report = classification_report(y_test, nb_y_pred, output_dict=True)
    print('Accuracy of Naive Bayes model is ', nb_acc)
    return nb_acc, nb_report


def process_dtc():
        # 3. dtc
    dtc_model = DecisionTreeClassifier()
    # Fitting training set to the model
    dtc_model.fit(xv_train.toarray(), y_train)
    # Predicting the test set results based on the model
    dtc_y_pred = dtc_model.predict(xv_test.toarray())
    # Calculate the accuracy score of this model
    dtc_acc = accuracy_score(y_test, dtc_y_pred)
    dtc_report = classification_report(y_test, dtc_y_pred, output_dict=True)
    print('Accuracy of DTC model is ', dtc_acc)
    return dtc_acc, dtc_report









def fake_news_det(news):
    svm_model = SVC(kernel='linear')
    svm_model.fit(xv_train, y_train)
    input_data = {"text": [news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(stop_words_cleaning)
    print(new_def_test)
    new_x_test = new_def_test["text"]
    print(new_x_test)
    vectorized_input_data = vectorization.transform(new_x_test)
    prediction = svm_model.predict(vectorized_input_data)

    if prediction == 1:
       return "depressive"
    else:
        return "non_depressive"