import pandas as pd 
import numpy as np
df=pd.read_csv("D:\MLL\ML_Project\Dataset\HeartDisease.csv")
df.head()
df_num = df.select_dtypes(np.number)
df=df_num
inputs=df.drop('HeartDisease',axis='columns')
target=np.array(df.HeartDisease)
from sklearn.model_selection import train_test_split
X_train,Xtest,Y_train,Y_test=train_test_split(inputs,target,test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
y_pred=knn.predict(Xtest)
print(y_pred)
len(y_pred)
from sklearn import metrics
ac= metrics.accuracy_score(Y_test,y_pred)
print(ac)