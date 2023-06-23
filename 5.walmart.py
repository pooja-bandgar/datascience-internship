import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

logr = LinearRegression()

df=pd.read_csv("C:/Users/SHRAWANI/Downloads/CaseStudy-Walmart-main/CaseStudy-Walmart-main/train.csv")
print(df)
df['Date'] = pd.to_datetime(df['Date'])
df['days']=df['Date'].dt.day
df['month']=df['Date'].dt.month
df['year']=df['Date'].dt.year
df['WeekOfYear'] = df.Date.dt.isocalendar().week
df.drop('Date',axis=1,inplace=True)

le = LabelEncoder()
df['IsHoliday'] = le.fit_transform(df['IsHoliday'])


x=df.drop('Weekly_Sales',axis=1)
y=df['Weekly_Sales']

pca = PCA(n_components=3)
fit = pca.fit(x)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.2)
logr.fit(X_Train, Y_Train)
print(logr.score(X_Test,Y_Test))


