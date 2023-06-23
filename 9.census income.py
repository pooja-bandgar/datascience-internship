import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("C:/Users/SHRAWANI/Downloads/Census Income/adult.csv")
print(df)

df['workclass'] = df['workclass'].replace(["?"], [" "])
df['occupation'] = df['occupation'].replace(["?"], [" "])
df['native.country'] = df['native.country'].replace(["?"], [" "])

print(df["workclass"])
print(df["occupation"])

le = LabelEncoder()
le.fit(df['workclass'])
df['workclass'] = le.transform(df['workclass'])

le.fit(df['education'])
df['education'] = le.transform(df['education'])

le.fit(df['relationship'])
df['relationship'] = le.transform(df['relationship'])

le.fit(df['marital.status'])
df['marital.status'] = le.transform(df['marital.status'])

le.fit(df['native.country'])
df['native.country'] = le.transform(df['native.country'])

le.fit(df['sex'])
df['sex'] = le.transform(df['sex'])

le.fit(df['race'])
df['race'] = le.transform(df['race'])

le.fit(df['occupation'])
df['occupation'] = le.transform(df['occupation'])
le.fit(df['income'])
df['income'] = le.transform(df['income'])
print(df)

df["occupation"].fillna((df["occupation"].median()),inplace=True)
df["native.country"].fillna((df["native.country"].median()),inplace=True)
df["workclass"].fillna((df["workclass"].median()),inplace=True)
print(df)
x=df.drop("fnlwgt",axis=1)
x=x.drop("education.num",axis=1)
x=x.drop("income",axis=1)
y=df["income"]

mod = ExtraTreesClassifier()
mod.fit(x,y)
print(mod.feature_importances_)
feat_importance=pd.Series(mod.feature_importances_,index=x.columns)
feat_importance.nlargest(13).plot(kind="barh")
plt.show()

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)
print(Counter(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)