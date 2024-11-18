'''

given

Logistic Regression, SVM, Random Forest, Gradient Boosting
'''
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# predefined PARAM
model_num = 4

model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()
model4 = GradientBoostingClassifier()
search_space1 = {}  # tmp
search_space2 = {}  # tmp
search_space3 = {}  # tmp
search_space4 = {}  # tmp

models = [model1, model2, model3, model4]

model_param = dict(min_samples_rate=0.9, n_extimators=10, max_depth=6)

data = pd.read_csv('./train.csv')
dum = np.random.rand(len(data), 5)
train_df, test_df, _, __ = train_test_split(data, dum)
object_columns = data.select_dtypes(include=['object']).columns


def NullDropHandler(train, test):
    tmp_stack = []

    # 1. train과 test 데이터프레임의 공통 열만 처리
    common_columns = train.columns.intersection(test.columns)

    # 2. 결측값이 200개 이상인 열은 삭제
    for column in common_columns:
        if train[column].isnull().sum() > 500:
            train = train.drop([column], axis=1)
            test = test.drop([column], axis=1)
            tmp_stack.append(column)  # 제거한 열을 스택에 저장

    # 공통 열 리스트를 업데이트하여 삭제된 열을 제외
    common_columns = train.columns.intersection(test.columns)

    # 3. 결측값이 있는 열을 처리
    for column in common_columns:
        # train 데이터프레임에서 결측값이 있는 경우
        if train[column].isnull().sum() > 0:
            if train[column].dtype != 'object':  # 숫자형 데이터 확인
                value_train = train[column].mean()
                train[column] = train[column].fillna(value_train)
            else:
                train[column] = train[column].fillna(str('none'))

                # test 데이터프레임에서 결측값이 있는 경우
        if test[column].isnull().sum() > 0:
            if test[column].dtype != 'object':  # 숫자형 데이터 확인
                value_test = test[column].mean()
                test[column] = test[column].fillna(value_test)
            else:
                test[column] = test[column].fillna(str('none'))

    return train, test, tmp_stack  # 제거된 열 목록도 반환


def NaNHandler(train, test):
    tmp_stack = []
    common_columns = train.columns.intersection(test.columns)
    # 1. NaN 값이 200개 이상인 열은 삭제
    for column in common_columns:
        if train[column].isna().sum() > 200:
            train = train.drop([column], axis=1)
            test = test.drop([column], axis=1)
            tmp_stack.append(column)
    common_columns = train.columns.intersection(test.columns)
    # 2. NaN 값이 있는 열을 처리
    for column in common_columns:
        # train 데이터프레임에서 결측값이 있는 경우
        if train[column].isna().sum() > 0:
            if train[column].dtype != 'object':  # 숫자형 데이터 확인
                value_train = train[column].mean()
                train[column] = train[column].fillna(value_train)  # 평균값으로 NaN 채움
            else:
                train[column] = train[column].fillna(str('none'))

                # test 데이터프레임에서 결측값이 있는 경우
        if test[column].isna().sum() > 0:
            if test[column].dtype != 'object':  # 숫자형 데이터 확인
                value_test = test[column].mean()
                test[column] = test[column].fillna(value_test)  # 평균값으로 NaN 채움
            else:
                test[column] = test[column].fillna(str('none'))

    return train, test, tmp_stack


train_df, test_df, removed_columns = NaNHandler(train_df, test_df)

assert train_df.isna().sum().sum() == 0
assert test_df.isna().sum().sum()

print('corr call')


def remove_collinear_features(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold=0.8) -> tuple:
    corr_matrix = train_df.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            # 절대값을 씌우는 이유는
            # corr 절대값이 높은거를 제거하면 되기 때문에
            val = abs(item.values)

            if val >= threshold:
                print(col.values[0], '|', row.values[0], '|', round(val[0][0], 2))
                drop_cols.append(col.values[0])

    drops = set(drop_cols)
    drops.discard('SalePrice')
    train_df = train_df.drop(columns=drops)
    test_df = test_df.drop(columns=drops)

    return train_df, test_df


train_df, test_df = remove_collinear_features(train_df, test_df)
Y_train = train_df['SalePrice']
X_train = train_df.drop(['SalePrice'], axis=1)
Y_test = test_df['SalePrice']
X_test = test_df.drop(['SalePrice'], axis=1)
