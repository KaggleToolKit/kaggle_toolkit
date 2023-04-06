import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
target_name = list(set(df_train.columns) - set(df_test.columns))[0]  # determine target column name

task = Task('binary', metric='f1')
automl = TabularAutoML(task=task, timeout=3600, cpu_limit=4, general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]})
oof_pred = automl.fit_predict(df_train, roles={'target': target_name})
  
test_pred = automl.predict(df_test)

pd.DataFrame(test_pred.data[:, 0]).to_csv('./storage/submission.csv', index=False, header=False)