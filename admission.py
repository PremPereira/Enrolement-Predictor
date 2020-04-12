import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model, metrics 
from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle


df=pd.read_csv("collegeenrollment.csv")
a = pd.get_dummies(df['college rank'])

df = pd.concat([a,df],axis = 1)
df = df.drop(['college rank'],axis = 1)

lenc = LabelEncoder()

for i in range(4,13):
    df.iloc[:,i] = lenc.fit_transform(df.iloc[:,i])

df = df.astype(object)

df.to_csv("encoded.csv",index=None)
#print(df.values)


data = pd.read_csv("encoded.csv")
data = data.astype(int)
array = data.values

x= array[:,:-1]
y = array[:,-1]

validation_size = 0.2

seed = 7

x_train,x_validation,y_train,y_validation = model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)

reg = linear_model.LinearRegression() 
reg.fit(x_train, y_train)
#print('Coefficients: \n', reg.coef_)

print('Variance score: {}'.format(reg.score(x_validation, y_validation)))

prediction = reg.predict(x_validation)

print("Mean squared error: %.2f"% mean_squared_error(y_validation, prediction))

# #Prediction
# x=[1	,0	,0,	453,	1,	1,	1,	1,	1,	1,	1,	0,	1,	3,	5,	95,	866]
# #16th row of the dataset..y=856
# x = np.array(x).reshape(1,-1)

# y = [856]
# p = reg.predict(x)
# #p is predicted value for x
# print(p)

#calulating accuracy
#print("Mean squared error: %.2f"% mean_squared_error(y, p))
file = open("saved_model.pkl","wb")
pickle.dump(reg,file)

