import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print(stopwords.words('english'))

news_dataset = pd.read_csv('train.csv/train.csv')
# print(news_dataset.isnull().sum())

news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
# print(news_dataset['content'])

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
# print(X)
# print(Y)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
# print(news_dataset['content'])

X = news_dataset['content'].values
Y = news_dataset['label'].values
# print(X)
# print(Y)
# print(Y.shape)

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
# print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print('Accuracy score of the training data : ', training_data_accuracy)

X_new = X_test[4]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')