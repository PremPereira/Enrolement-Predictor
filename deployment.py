from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
from math import ceil

app=Flask(__name__)

def predict_result(values):
    file = open("saved_model.pkl","rb")
    model = pickle.load(file)

    x = np.array(values)
    a = np.zeros(3,dtype=int)
    x = np.delete(x,0,0)
    a[values[1]] = 1
    x = np.concatenate((a,x),axis = 0)
    x = x.reshape(1,-1)

    prediction = model.predict(x)
    print(prediction)
    print(x)
    return prediction


@app.route('/home')
def home():
	return render_template("home.html")

@app.route('/result',methods = ['POST'])
def result():
    dict = request.form.to_dict()
    list1 = [int(x) for x in list(dict.values())]
    result = predict_result(list1)
    return "<center><h1>Estimated enrolment for the pesent year : {}</h1><center>".format(ceil(result[0]))

if __name__ == "__main__":
    app.debug =True
    app.run()



