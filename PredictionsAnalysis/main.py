import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Tesla.csv')

#print(df.describe)


#Exploratory Data Analysis - analyzing the data using visual techniques:
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.xlabel('Time')
#plt.show()


plt.figure(figsize=(15,5))
plt.plot(df['High'], color='blue', label='High')
plt.plot(df['Low'], color='red', label='Low')
plt.title('Tesla Low and High Prices over time.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.xlabel('Time')
#plt.show()
#observe how to lower and upper bound of Tesla stocks suddenly increased 



#checking for any null values...
print(df.isnull().sum())
#...there were none


features = ['Open','High','Low','Close','Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
#plt.show()
#observed two peaks in the open, high, low and close data,
#indicating the data varies significantly in 2 regions.
#Volume data is left skewed.




"""
Companies prepare quarterly (every three months) results
and publishes them for analyses on company's performance.
This is important to consider...
"""
splitted_date = df['Date'].str.split('-', expand=True)
df['day'] = splitted_date[2].astype('int')
df['month'] = splitted_date[1].astype('int')
df['year'] = splitted_date[0].astype('int')

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)



#For some observations...

data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open','High','Low','Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
#plt.show()


print(df.groupby('is_quarter_end').mean())
#observed differences in price and volume between the end of a quarter
#and non quarter end months.


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
# target = 1 if the next 'close' value is higher than the current 'close' value else 0
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
#print(df.head(10))


# Used to check whether the target is balanced
plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
#plt.show()



# Since I have added in features to the dataset, I have to check that
# there are no highly correlated features which do not contribute
# to the algorithm

plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
#plt.show()
# no correlation between added features - good to go!

#splitting and normalising the data
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
#print(X_train.shape, X_valid.shape)
print(f'HERE: {X_train}')

models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print(f'predict_proba = {models[i].predict_proba(X_train)[:,1]}')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
# observe that XGBClassifier has the highest performance but
# is overfitting (difference between the training and the validation accuracy is too high)
# Logistic Regression seems good, so use models[0].
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
# confusion matrix shows that the model is not very accurate :(