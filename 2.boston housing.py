import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("C:/Users/SHRAWANI/Downloads/archive/housing.csv")
print(df)
df.describe()
print(df.isnull().sum())

x = df.drop("MEDV", axis=1)
y = df["MEDV"]

mod = ExtraTreesRegressor()
mod.fit(x, y)
print(mod.feature_importances_)
feat_importance = pd.Series(mod.feature_importances_, index=x.columns)
feat_importance.nlargest(13).plot(kind="barh")
plt.show()

z_scores = (df - df.mean()) / df.std()
outliers = (z_scores > 3) | (z_scores < -3)

plt.figure(figsize=(12, 6))
sb.boxplot(data=df, orient='h')
plt.title("Box Plot of Data Columns")
plt.xlabel("Value")
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

df_no_outliers = df.copy()
for column in df.columns:
    is_outlier = (df[column] < lower_threshold[column]) | (df[column] > upper_threshold[column])
    df_no_outliers[column] = df[column].mask(is_outlier)

print(df_no_outliers)
plt.figure(figsize=(12, 6))
sb.boxplot(data=df_no_outliers, orient='h')
plt.xlabel("Value")
plt.show()

df['MEDV_category'] = pd.cut(df['MEDV'], bins=[0, 20, 50, 100], labels=['Low', 'Medium', 'High'])
x = df.drop(["MEDV", "MEDV_category"], axis=1)
y = df['MEDV_category']
logr = LogisticRegression()
pca = PCA(n_components=2)
x_transformed = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, random_state=10, test_size=0.1)
logr.fit(x_train, y_train)
y_pred = logr.predict(x_test)
print(accuracy_score(y_test, y_pred))

