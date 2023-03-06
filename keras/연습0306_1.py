import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path= './_data/ddarung/'
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path+'test.csv',index_col=0) # index_col=0 각 행의 이름이 위치한 열 지정

print(train_csv) #[1459 rows x 10 columns]
print(test_csv) #[715 rows x 9 columns]






