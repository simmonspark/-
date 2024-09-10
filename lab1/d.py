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

print('-------------------')
print('|     lab1         |')
print('-------------------')

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
urllib.request.urlretrieve(url, "housing.tgz")  # save in a file
import tarfile

tar = tarfile.open("housing.tgz")
tar.extractall()
tar.close()

housing = pd.read_csv("housing.csv")
print(housing.describe())
print(housing.info())
print(housing['median_income'])


class CombinedAttributesAdder():

    def fit_and_transform(self, df: pd.DataFrame):
        return self._DataHandler(df)

    @staticmethod
    def _NullHandler(df: pd.DataFrame):
        # null값이 없습니다.
        return df

    @staticmethod
    def _Dropfeatures(df: pd.DataFrame):
        # drop이 필요 없습니다.
        return df

    @staticmethod
    def _Encoder(df: pd.DataFrame):
        feature = 'ocean_proximity'
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        return df

    @staticmethod
    def _Norm(df: pd.DataFrame):
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households']
        for feature in numeric_features:
            sk = StandardScaler()
            df[feature] = sk.fit_transform(df[[feature]])
        return df

    def _DataHandler(self, df: pd.DataFrame):
        df = self._NullHandler(df)
        df = self._Dropfeatures(df)
        df = self._Norm(df)
        df = self._Encoder(df)
        return df


handler = CombinedAttributesAdder()
df = handler.fit_and_transform(housing)

feature = df.drop(['median_house_value'], axis=1)
print(feature)

housing["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
target = df['median_house_value']
print(target.info())
print(target)
print(feature)
X_train, X_test, y_train, y_test = train_test_split(feature, target, \
                                                    test_size=0.2, random_state=11)
xgb_reg = xgb.XGBRegressor(random_state=123)
param_grid = {
    'n_estimators': [30, 50, 100, 150, 200],  # 트리의 수
    'learning_rate': [0.01, 0.1, 0.3],  # 학습률
    'max_depth': [4, 5, 7, 10, 12, 15],  # 트리의 최대 깊이
}
grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

test_predictions = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("테스트 세트 성능:")
print("RMSE:", test_rmse)
print("R2:", test_r2)
