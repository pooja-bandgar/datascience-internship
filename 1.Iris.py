import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("C:/Users/SHRAWANI/Downloads/archive (4)/IRIS.csv")
print(df)

X = df.drop('species', axis = 1)
Y = df['species']
print(X)
print(Y)

bestfeatures =SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis = 1)
featuresScores.columns = ['Specs','Score']
print(featuresScores)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index = X.columns)
feat_importance.nlargest(4).plot(kind = 'barh')
plt.show()
print(df.isnull().sum)
print(df.notnull().sum())

df['sepal_length'].fillna((df['sepal_length'].mean()), inplace=True)
print(df.isnull().sum())
df['sepal_length'].fillna((df['sepal_length'].max()), inplace=True)
print(df.isnull().sum())

print(Counter(Y))
ros = RandomOverSampler(random_state = 0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))

logr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(accuracy_score(y_test,y_pred))


print(df['sepal_length'])
Q1=df['sepal_length'].quantile(0.25)
Q3=df['sepal_length'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
print(upper)
print(lower)
out1 = df[df['sepal_length']<lower].values
out2 = df[df['sepal_length']>upper].values
df['sepal_length'].replace(out1,lower,inplace = True)
df['sepal_length'].replace(out2,upper,inplace = True)
print(df['sepal_length'])