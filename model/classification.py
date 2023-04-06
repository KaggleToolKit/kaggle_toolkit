import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

target_name = list(set(df_train.columns) - set(df_test.columns))[0]  # determine target column name
df_train = df_train.dropna(0)
df_test = df_test.dropna(0)
y_train = df_train[target_name]
df_train = df_train.drop(target_name, axis=1)

numeric_columns = df_train.select_dtypes(include=['int64', 'float64']).columns
df_train = df_train[[col for col in df_train.columns if col in numeric_columns]]
df_test = df_test[[col for col in df_test.columns if col in numeric_columns]]

clf = LogisticRegression().fit(df_train, y_train)
y_pred = clf.predict(df_test)
pd.DataFrame(y_pred).to_csv('./storage/submission.csv', index=False, header=False)
