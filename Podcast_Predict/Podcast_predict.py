
!pip install kaggle


import os

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

!kaggle competitions download -c playground-series-s5e4


import zipfile


with zipfile.ZipFile("playground-series-s5e4.zip", "r") as zip_ref:
    zip_ref.extractall("data") 

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
train.head()

# %%
train.isnull().sum()

# %%
train.info()

# %%
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(0).astype(float)

test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(0).astype(float)

# %%
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].median())
test['Number_of_Ads'] = test['Number_of_Ads'].fillna(test['Number_of_Ads'].median())

# %%
train.isnull().sum()
test.isnull().sum()

# %%

print("Object 타입 컬럼 목록:")
print(train.select_dtypes(include='object').columns)


drop_cols = ['Podcast_Name', 'Episode_Title']


categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


train_drop = train.drop(columns=drop_cols + ['Listening_Time_minutes'])  # 타깃도 잠깐 제외
test_drop = test.drop(columns=drop_cols)


all_data = pd.concat([train_drop, test_drop], axis=0)


all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols)


X_train = all_data_encoded.iloc[:len(train), :]
X_test = all_data_encoded.iloc[len(train):, :]
y_train = train['Listening_Time_minutes']


# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
print(train.dtypes)
print(test.dtypes)

# %%

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_tr, y_tr)


y_pred = model.predict(X_val)

# %%
test_preds = model.predict(X_test)

# %%
import pandas as pd

submission = pd.DataFrame({
    'id': test['id'].reset_index(drop=True),
    'Listening_Time_minutes': test_preds
})

# %%
submission.to_csv("submission0.csv", index=False)
print(submission.head())

# %%



