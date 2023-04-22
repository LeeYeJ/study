# 27-1 for문 바꾸고 잔소리 없애라

import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline  
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#1.데이터
datasets = load_iris()
# datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
x,y,train_size=0.8, shuffle=True, random_state=337)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models_name =[
    'model1',
    'model2',
    'model3',
    'model4'
]

models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
]
#2. 모델 구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

#3. 컴파일, 훈련 및 평가
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
ax = axes.ravel()

for i, model in enumerate(models):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} accuracy_score: {acc:.4f}")
    print(f"{model.__class__.__name__} feature importances:", model.feature_importances_)

    # 그림그리기
    n_features = datasets.data.shape[1]
    ax[i].barh(np.arange(n_features), model.feature_importances_, align='center')
    ax[i].set_yticks(np.arange(n_features))
    ax[i].set_yticklabels(datasets.feature_names)
    ax[i].set_xlabel('Feature Importances')
    ax[i].set_ylabel('Features')
    ax[i].set_ylim(-1, n_features)
    ax[i].set_title(model.__class__.__name__)

plt.show()

# num=[1,2,3,4]

# for i,v in enumerate(models):
#     model = v
#     models_name[i] = v
#     v.fit(x_train,y_train)
    
#     # 컬럼 중요도 가시화 / 트리계열은 제공됨
#     def plot_feature_importances(model):
#         n_features = datasets.data.shape[1]
#         plt.barh(np.arange(n_features),model.feature_importances_, align='center')
#         plt.yticks(np.arange(n_features), datasets.feature_names)
#         plt.xlabel('Feature Importances')
#         plt.ylabel('Fetures')
#         plt.ylim(-1,n_features)        
#         plt.title(model)
        
#         for k in num:   
#             plt.subplot(2,2,k) # 2 by 2에 첫번째 칸에 쓰겠다
#             plot_feature_importances(model)
# plt.show()

    
    
        

# model1 = DecisionTreeClassifier()
# model2= RandomForestClassifier()
# model3 = GradientBoostingClassifier()
# model4 = XGBClassifier()

# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
# model3.fit(x_train,y_train)
# model4.fit(x_train,y_train)

# 컬럼 중요도 가시화 / 트리계열은 제공됨
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Fetures')
#     plt.ylim(-1,n_features)
#     plt.title(model)
    
# plt.subplot(2,2,1) # 2 by 2에 첫번째 칸에 쓰겠다
# plot_feature_importances(model1)

# plt.subplot(2,2,2) # 2 by 2에 두번째 칸에 쓰겠다
# plot_feature_importances(model2)

# plt.subplot(2,2,3) # 2 by 2에 세번째 칸에 쓰겠다
# plot_feature_importances(model3)

# plt.subplot(2,2,4) # 2 by 2에 네번째 칸에 쓰겠다
# plot_feature_importances(model4)

# plt.show()

# # 상관관계 가시화
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(test_data.corr())
# plt.figure(figsize=(10,8)) #새로운 그림(figure)을 생성
# sns.set(font_scale=1.2) # seaborn에서 그래프를 그릴 때 사용되는 기본 스타일, 폰트, 색상, 크기 등을 설정
# sns.heatmap(train_data.corr(),square=True,annot=True,cbar=True) #데이터 프레임, 배열, 시리즈 등을 입력받아 색상으로 나타내
# plt.show()





