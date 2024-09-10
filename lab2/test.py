import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import urllib.request
import urllib.request
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import os
print('-------------------')
print('|     lab1         |')
print('-------------------')

if os.path.exists('/semi.csv'):
    url = "https://drive.google.com/uc?export=download&id=1XCU0eo2xZ03xhxJhdrCnVjduCoaBQ7kJ"
    urllib.request.urlretrieve(url, "semi.csv") # save in a file
else:
    print('data already exist')

from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('semi.csv')



def NullHandler(df : pd.DataFrame):
    features = range(590)
    for fe in features:
        if df[str(fe)].isnull().sum() !=0 :
            df[str(fe)] = df[str(fe)].fillna(np.mean(df[str(fe)]))
    assert df.isnull().sum().sum() == 0
    return df

def DuplicatedHandler(df : pd.DataFrame):
    features = range(590)
    stack = []
    idx = []
    for fe in tqdm(features):
        st = df[str(fe)].std()
        me = df[str(fe)].mean()
        mx = df[str(fe)].max()
        mn = df[str(fe)].mean()
        if (st,me,mx,mn) in stack:
            idx.append(fe)
            print('duplicated!')
        else:
            stack.append((st,me,mx,mn))

    for i in idx:
        df = df.drop(str(i),axis = 1)
    df = df.drop('Time', axis=1)
    return df


def Encoder(df: pd.DataFrame):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    return df

def Norm(df: pd.DataFrame):
    numeric_features= df.columns[:-1]

    for feature in numeric_features:
        sk = StandardScaler()
        df[feature] = sk.fit_transform(df[[feature]])
    return df


def DataHandler(df: pd.DataFrame):
    df = NullHandler(df)
    df= DuplicatedHandler(df)
    df = Norm(df)
    return df


df = DataHandler(df)


print('is na data check',df.isna().sum().sum())




