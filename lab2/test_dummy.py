#%%
import numpy as np
import pandas as pd

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
    '''
    혹시 파일이 생기지 않는다면, 아래 두 줄의 스크립스틑 파이썬 .py파일로 만들어서 실행하면 됩니다. 
    '''
    url = "https://drive.google.com/uc?export=download&id=1XCU0eo2xZ03xhxJhdrCnVjduCoaBQ7kJ"
    urllib.request.urlretrieve(url, "semi.csv")  # save in a file
else:
    print('data already exist')

#%%
df = pd.read_csv('semi.csv')
#%%
df.isnull().sum()
#%%
df.info()
#%%
df.describe()
#%%

def NullHandler(df: pd.DataFrame):
    features = range(590)
    for fe in features:
        if df[str(fe)].isnull().sum() != 0:
            df[str(fe)] = df[str(fe)].fillna(np.mean(df[str(fe)]))
    assert df.isnull().sum().sum() == 0
    print('NULL PASSED!\n')
    return df


def DuplicatedHandler(df: pd.DataFrame):
    features = range(590)
    stack = []
    idx = []
    for fe in tqdm(features):
        st = df[str(fe)].std()
        me = df[str(fe)].mean()
        mx = df[str(fe)].max()
        mn = df[str(fe)].mean()
        if (st, me, mx, mn) in stack:
            idx.append(fe)
            print('duplicated!')
        else:
            stack.append((st, me, mx, mn))

    for i in idx:
        df = df.drop(str(i), axis=1)
    df = df.drop(['Time'], axis=1)
    return df

def Norm(df: pd.DataFrame):
    numeric_features = df.columns[:-1]

    for feature in numeric_features:
        sk = StandardScaler()
        df[feature] = sk.fit_transform(df[[feature]])
    return df


def DataHandler(df: pd.DataFrame):
    df = NullHandler(df)
    df = DuplicatedHandler(df)
    df = Norm(df)
    return df


df = DataHandler(df)
#%%
print('is NaN data check : ', df.isna().sum().sum())

#%%
df
#%%
feature = df.drop('Pass/Fail', axis=1)
target = df['Pass/Fail']
#%%
target
#%%
feature
#%%
X_train, X_test, y_train, y_test = train_test_split(feature, target, \
                                                    test_size=0.2, random_state=11,stratify=target)

param_grid = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0], 
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2] 
}

y_train = [0 if label == -1 else 1 for label in y_train]
y_test = [0 if label == -1 else 1 for label in y_test]
#%%
model = xgb.XGBClassifier(eval_metric='mlogloss')
#%%
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy', verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
#%%
from sklearn.metrics import f1_score, balanced_accuracy_score

f1 = f1_score(best_model.predict(X_test),y_test)
balance = balanced_accuracy_score(best_model.predict(X_test),y_test)
print(f'f1 score is : {f1}')
print(f'balanced score is : {balance}')