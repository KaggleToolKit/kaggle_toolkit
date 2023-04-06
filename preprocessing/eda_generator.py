import pandas as pd
from pandas_profiling import ProfileReport


df = pd.read_csv('./data/train.csv')
prof = ProfileReport(df)
prof.to_file(output_file='./storage/eda.html')
