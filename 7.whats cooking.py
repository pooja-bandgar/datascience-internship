import pandas as pd

df = pd.read_json("C:/Users/admin/Downloads/CASESTUDY-WhatsCooking-main/CASESTUDY-WhatsCooking-main/train.json")
print(df)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

nb = MultinomialNB()
dtc = DecisionTreeClassifier()


dc = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

# print(df["cuisine"].unique())
x = df['ingredients']
y = df['cuisine'].apply(dc.index)

df['all_ingredients'] = df['ingredients'].map(';'.join)


cv = CountVectorizer()
x = cv.fit_transform(df['all_ingredients'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

dtc.fit(x_train, y_train)
y_dtc = dtc.predict(x_test)

nb.fit(x_train, y_train)
y_nb = nb.predict(x_test)



print("Decision Tree:", accuracy_score(y_test, y_dtc))
print("Naive Bayes:", accuracy_score(y_test, y_nb))