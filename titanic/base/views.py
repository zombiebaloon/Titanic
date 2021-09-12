from django.shortcuts import render
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def index(request):
    return render(request, 'index.html')

def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    model = pickle.load(open('ml_model.sav', 'rb'))
    scaled = pickle.load(open('scaler.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [pclass, sex, age, sibsp, parch, fare, C, Q, S]
    ]))
    
    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    result = getPredictions(pclass, sex, age, sibsp,
                            parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'result': result})

def EDA(request):
    
    return render(request,'EDA.html')

def char():
    df = pd.read_csv("titanic.csv")
    df['Age']= df['Age'].fillna(df['Age'].mean())
    df.dropna(subset=['Embarked'],inplace=True)
    num_cols = df.select_dtypes([np.int64,np.float64]).columns.tolist()
    num_cols.remove('PassengerId')
    for col in num_cols:
        df.hist(column=col)
        plt.savefig('base/static/saved_figure.png')
    obj_cols = df.select_dtypes([np.object]).columns.tolist()
    for col in obj_cols:
        plt.savefig('base/static/saved_figures.png')
        df[col].value_counts().plot(kind='bar')


