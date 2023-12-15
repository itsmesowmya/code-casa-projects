import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv(r"C:\Users\91988\Downloads\Stock price prediction\NFLX.csv")
df=df.dropna()
print(df)
df.plot(x="Date",y="Close", label="Netflix Closing Price", xlabel="Date", ylabel="Closing Price")
plt.xticks(rotation=45)
plt.show()
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(10,5))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.distplot(df[col])
plt.show()
plt.subplots(figsize=(10,5))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.boxplot(df[col])
plt.show()
plt.show()
model=RandomForestRegressor()
X=df[['Open','High','Low','Volume']]
X=X[:(len(df)-1)]
Y=df['Close']
Y=Y[:(len(df)-1)]
model.fit(X,Y)
prediction=model.predict(X)
print('Accuracy Score: ',model.score(X,Y))
new_data=df[['Open','High','Low','Volume']].tail(1)
predict=model.predict(new_data)
print("The model predicts Netflix stock price of last date 04-02-2022 to be:",predict)
print("Actual Netflix stock price value on 04-02-2022 is:",df[['Close']].tail(1).values[0][0])
