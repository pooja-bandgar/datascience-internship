import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

df = pd.read_csv("C:/Users/SHRAWANI/Downloads/archive (1)/Titanic-Dataset.csv")
print(df)

le=LabelEncoder()
le.fit(df["Sex"])
df["Sex"]=le.transform(df["Sex"])
print(df["Sex"])

x=df.drop("Name",axis=1)
x=x.drop("Cabin",axis=1)
x=x.drop("Embarked",axis=1)
x=x.drop("Survived",axis=1)
x=x.drop("Ticket",axis=1)
x=x.drop("PassengerId",axis=1)
y=df["Survived"]
print(x)
print(y)

print(df.isnull().sum())
x["Age"].fillna((x["Age"].median()),inplace=True)
print(x.isnull().sum())

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)
print(Counter(y))

logr = LogisticRegression()
pca = PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10,test_size=0.1)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
print(accuracy_score(y_test,y_pred))