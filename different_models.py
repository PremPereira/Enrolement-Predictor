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
print("\n\n\n****LINEAR REG****")
print('Variance score: {}'.format(reg.score(x_validation, y_validation)))

prediction = reg.predict(x_validation)

print("Mean squared error: %.2f"% mean_squared_error(y_validation, prediction))
#x=[1	,0	,0,	453,	1,	1,	1,	1,	1,	1,	1,	0,	1,	3,	5,	95,	866]
#x = np.array(x).reshape(1,-1)
#y = [856]
#p = reg.predict(x)
#p is predicted value for x
#print(p)

#calulating accuracy
#print("Mean squared error: %.2f"% mean_squared_error(y, p))


print("\n\n\n****RIDGE REG****")

from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(x_train,y_train)
print('Variance score: {}'.format(ridgeReg.score(x_validation, y_validation)))
pred = ridgeReg.predict(x_validation)
print("Mean squared error: %.2f"% mean_squared_error(y_validation, pred))

print("\n\n\n****LASSO REG****")
from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=0.3, normalize=True)
lassoReg.fit(x_train,y_train)
print('Variance score: {}'.format(lassoReg.score(x_validation, y_validation)))
pred = lassoReg.predict(x_validation)
print("Mean squared error: %.2f"% mean_squared_error(y_validation, pred))


print("\n\n\n****ELASTIC NET REG****")
from sklearn.linear_model import ElasticNet
ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
ENreg.fit(x_train,y_train)
print('Variance score: {}'.format(ENreg.score(x_validation, y_validation)))
pred_cv = ENreg.predict(x_validation)
print("Mean squared error: %.2f"% mean_squared_error(y_validation, pred))


