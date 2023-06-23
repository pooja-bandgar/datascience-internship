import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

dg = pd.read_csv("C:/Users/SHRAWANI/Downloads/Black Friday Sales/test.csv")
df = pd.read_csv("C:/Users/SHRAWANI/Downloads/Black Friday Sales/train.csv")

print(df)
print(dg)

print(df.isnull().sum())

df["Product_Category_2"] = df["Product_Category_2"].fillna(0)
df["Product_Category_3"] = df["Product_Category_3"].fillna(0)
print(df.isnull().sum())

le = LabelEncoder()
le.fit(df['Product_ID'])
df['Product_ID'] = le.transform(df['Product_ID'])
print(df)

le.fit(df['City_Category'])
df['City_Category'] = le.transform(df['City_Category'])
print(df)
print(df["City_Category"])

le.fit(df['Age'])
df['Age'] = le.transform(df['Age'])
print(df["Age"])

df['Stay_In_Current_City_Years'] = le.fit_transform(df['Stay_In_Current_City_Years'])
df['Gender'] = le.fit_transform(df['Gender'])

rus = RandomUnderSampler(random_state=42)
x_resampled, y_resampled = rus.fit_resample(df.drop(["Purchase", "User_ID"], axis=1), df["Purchase"])

x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
