import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

class MyLogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000):
        self.lr = lr
        self.num_iter = num_iter
        
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        # Run Gradient Descent
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
           # print(loss)
            
            
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

df = pd.read_csv('D:\SEM23\ML\heart.csv')
df['HeartDisease'].unique()
label_encoder = preprocessing.LabelEncoder()

df['HeartDisease'].unique()
df['Sex'].unique()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['ChestPainType'].unique()
df['ChestPainType']=label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG'].unique()
df['RestingECG']=label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'].unique()
df['ExerciseAngina']=label_encoder.fit_transform(df['RestingECG'])
df['ST_Slope'].unique()
df['ST_Slope']=label_encoder.fit_transform(df['RestingECG'])

X=df.drop(["HeartDisease"],axis=1)
target_values = df['HeartDisease'].unique()
y=np.array(df['HeartDisease'])
df['HeartDisease'].unique()
y_list = []
for i in range(0 , len(target_values)):
    y_list.append((y != i) * 1)
    print(y_list[i])
    print('----------------------------------------')
print(X)
model_lists = []
for i in range(0, len(y_list)):
    model = MyLogisticRegression(lr=0.1, num_iter=3000)
    model.fit(X, y_list[i])
    model_lists.append(model)

print(model_lists )
Z=np.array([60,1,3,157,130,1,1,120,1,2,1])

for j in range(0 , len(model_lists)):
    preds = model_lists[j].predict(Z)
    print(preds)

for j in range(0 , len(model_lists)):
    preds = model_lists[j].predict_prob(Z)
    print(preds)