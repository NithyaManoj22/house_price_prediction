import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn import ensemble

path = "C:/Users/User/Desktop/hpp.csv"
dataset = pd.read_csv(path)
print(dataset.describe())

dataset['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,10))
sns.jointplot(x=dataset.lat.values, y=dataset.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine


plt.scatter(dataset.bedrooms,dataset.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

dataset.floors.value_counts().plot(kind='bar')
plt.scatter(dataset.floors,dataset.price)
plt.xlabel("Floors")
plt.ylabel("Price")
plt.show()


plt.scatter(dataset.zipcode,dataset.price)
plt.title("Prices based on the zipcode")
plt.show()

names = []
train1 = dataset.drop(['price'],axis=1)
labels = dataset['price']

x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

reg = LinearRegression()
reg.fit(x_train,y_train)
acc1=reg.score(x_test,y_test)
print("Accuracy due to linear regression test is :"+str(acc1*100)+"%")


gbr = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
gbr.fit(x_train, y_train)
acc2=gbr.score(x_test,y_test)
print("Accuracy due to Gradient Boosting Regressor test is :"+str(acc2*100)+"%")
