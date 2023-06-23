import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv("C:/Users/SHRAWANI/Downloads/Wine Quality/winequalityN.csv")
print(df)
le = LabelEncoder()
le.fit(df['type'])
df['type'] = le.transform(df['type'])

print(df.isnull().sum())
x=df.drop("quality",axis=1)
y=df["quality"]
print(x)
print(y)
x["fixed acidity"].fillna((x["fixed acidity"].median()),inplace=True)
x["volatile acidity"].fillna((x["volatile acidity"].median()),inplace=True)
x["citric acid"].fillna((x["citric acid"].median()),inplace=True)
x["residual sugar"].fillna((x["residual sugar"].median()),inplace=True)
x["chlorides"].fillna((x["chlorides"].median()),inplace=True)
x["pH"].fillna((x["pH"].median()),inplace=True)
x["sulphates"].fillna((x["sulphates"].median()),inplace=True)
print(x.isnull().sum())

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)
print(Counter(y))

mod = ExtraTreesClassifier()
mod.fit(x,y)
print(mod.feature_importances_)
feat_importance=pd.Series(mod.feature_importances_,index=x.columns)
feat_importance.nlargest(12).plot(kind="barh")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
