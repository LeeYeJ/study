import numpy as np
import pandas as pd
import glob # 폴더의 모든 걸 가져와서 텍스트화한다.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error,r2_score

# / // \ \\ 같다 (\n처럼 예약어가 명시되면 노란색으로 표시될수있으니까 주의해서 쓰자)

path ='./_data/AI_SPARK/'
''' 
-data하단
    TRAIN
    TRAIN_AWS
    TEST_INPUT
    TEST_AWS
    META
    answer.sample.py
'''
save_path = './_save/AI_SPARK/'

# train_data = pd.read_csv(path + 'TRAIN.csv')
# train_aws = pd.read_csv(path + 'TRAIN_AWS.csv')
# test_data = pd.read_csv(path + 'TEST.csv')
# test_aws = pd.read_csv(path + 'TEST_AWS.csv')
# submission = pd.read_csv(path + 'answer_sample.csv')

train_files = glob.glob(path +'TRAIN/*.csv')
print(train_files) # 리스트 형태
test_input_files = glob.glob(path +'test_input/*.csv')
# print(test_input_files)

########## train ###########
li=[]
for filename in train_files:
    df = pd.read_csv(filename,index_col=None, header=0,
                     encoding='UTF-8-sig')
    li.append(df) # [35064 rows x 4 columns]
print(len(li)) #17

train_dataset = pd.concat(li,axis=0, ignore_index=True) 
print(train_dataset) #[596088 rows x 4 columns]
    

########### test #############
li=[]
for filename in test_input_files:
    df = pd.read_csv(filename,index_col=None, header=0,
                     encoding='UTF-8-sig')
    li.append(df) 
print(li)  #[7728 rows x 4 columns]
print(len(li)) #17

test_input_files = pd.concat(li,axis=0, ignore_index=True) 
# print(test_input_files) #[131376 rows x 4 columns]

############# 라벨 인코더 ################
le = LabelEncoder()         # 라벨링한 데이터를 스케일러 할수있을까? ->  no!!!!일걸. 원핫해줘 -> 컬럼이 늘어나 -> 과적합될수있어 -> PCA로 줄여 / 스케일러 말고 주기함수..?사용
train_dataset['locate']=le.fit_transform(train_dataset['측정소']) #locate 메모리 공간 만듦 (즉 컬럼이 다섯개 됨)
test_input_files['locate'] = le.transform(test_input_files['측정소'])
print(train_dataset) # [596088 rows x 5 columns]
print(test_input_files) # [131376 rows x 5 columns]

#측정소 필요없으니까 drop
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_files = test_input_files.drop(['측정소'], axis=1)
print(train_dataset) #[596088 rows x 4 columns]
print(test_input_files) #[131376 rows x 4 columns]

######## 일시 -> 월, 일, 시간으로 분리 #############
# 12-31 21:00 -> 12와 21 추출
# print(train_dataset.info()) # object -> 스트링 형태

train_dataset['month'] = train_dataset['일시'].str[:2]
print(train_dataset['month'])
train_dataset['hour'] = train_dataset['일시'].str[6:8] #str[6:8] 12-31 21:00 스트링 자릿수
print(train_dataset['hour'])

test_input_files['month'] = test_input_files['일시'].str[:2]
print(test_input_files['month'])
test_input_files['hour'] = test_input_files['일시'].str[6:8] #str[6:8] 12-31 21:00 스트링 자릿수
print(test_input_files['hour'])

train_dataset = train_dataset.drop(['일시'],axis=1)
test_input_files = test_input_files.drop(['일시'],axis=1)
print(train_dataset)

# str -> int   / str -> 수치형 데이터로 바꿔준다
# train_dataset['month'] = pd.to_numeric(train_dataset['month']) # to_numeric =>  str -> 수치형 데이터로 바꿔준다 (1방법)

# train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int8') # 메모리를 최소로 잡으면 낭비 줄임 (2방법)

train_dataset['month'] = train_dataset['month'].astype('int16') # 타입이 오브젝트라 바꿔줬음
train_dataset['hour'] = train_dataset['hour'].astype('int16')
print(train_dataset.info())

test_input_files['month'] = test_input_files['month'].astype('int16') # 타입이 오브젝트라 바꿔줬음
test_input_files['hour'] = test_input_files['hour'].astype('int16')
print(test_input_files.info())

################# 결측치 제거 PM2.5에 15542ro 있다 ####################
# 전체 596085 -> 580546으로 줄인다.
train_dataset = train_dataset.dropna()
print(train_dataset.info())
'''
 0   연도      580546 non-null  int64
 1   PM2.5   580546 non-null  float64
 2   locate  580546 non-null  int32
 3   month   580546 non-null  int16
 4   hour    580546 non-null  int16
'''
# 파생 피처 만드는 것도 중요함 ( ex. 시즌 계절 )

y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'],axis=1)
print(x,'\n',y)

x_input_test = test_input_files[test_input_files.isna().any(axis=1)]
# print(x_input_test)

x_input_test = x_input_test.drop(['PM2.5'],axis=1)
print(x_input_test.info())


# train/test 스플릿 하고 스케일러 해줘 / 모든 데이터는 스케일러 해주자(성능이 보통 괜찮아)
x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8,shuffle=True,random_state=456421
)

print(x_test.shape, x_input_test.shape) #(116110, 4) (78336, 4)

parameters = {
     'n_estimators': 3,   # 처음에 에러뜨는지 확인하기 위해 작은 수를 주고 테스트 해보자
    'learning_rate':  0.07,  # 0.07
    # 'max_depth': 2, 
    # 'gamma': 1,
    # 'min_child_weight': 1,
    # 'subsample': 0.7, 
    # 'colsample_bytree': 0.2,
    # 'colsample_bylevel': 1., 
    # 'colsample_bynode': 1.,
    # 'reg_alpha' : 0.1,
    # 'reg_lambda': 0.1,
    # 'random_state':337,
    # 'verbose':0,
    'n_jobs': -1
}
# 2.모델
model = XGBRegressor()

# 3.컴파일 훈련 
model.set_params(**parameters,         # 컴파일 부분이라고 생각하자
                 eval_metric='mae',
                 early_stopping_rounds =200)

start_time = time.time()
model.fit(
    x_train,y_train,
    eval_set = [(x_train,y_train),(x_test,y_test)]
)
end_time = time.time()

print('걸린 시간 :', round(end_time -start_time,2),'초')

# 평가 예측
y_predict = model.predict(x_input_test)

results = model.score(x_test,y_test) # r2스코어 나옴
print('model.score :', results)

r2 = r2_score(y_test,y_predict)
print('r2 :',r2)

mae = mean_absolute_error(y_test,y_predict)
print('mae :',mae)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

y_submit = model.predict(test_csv)
y_submit = pd.DataFrame(y_submit)
# y_submit = y_submit.fillna(y_submit.mean()) # mean -> nan값을 평균값으로 대체해준다 
y_submit = y_submit.fillna(y_submit.median()) # median -> nan값을 중간값으로 대체해준다
# y_submit = y_submit.fillna(y_submit.mode()[1]) # mode -> nan값을 최빈값으로 대체해준다                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(path_save + 'kcal_' + date + '.csv')




