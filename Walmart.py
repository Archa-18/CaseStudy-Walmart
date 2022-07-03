import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

lr = LinearRegression()

data = pd.read_csv('train.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['days'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['WeekOfYear'] = data.Date.dt.isocalendar().week
data.drop('Date', axis=1, inplace=True)

le = LabelEncoder()
data['IsHoliday'] = le.fit_transform(data['IsHoliday'])

x = data.drop('Weekly_Sales', axis=1)
y = data['Weekly_Sales']

#Feature Extraction
pca = PCA(n_components=3)
fit = pca.fit(x)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.3)

lr.fit(X_Train, Y_Train)

print(lr.score(X_Test,Y_Test))

#Model Score 0.03063294371509706