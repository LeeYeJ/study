# #[실습]
# #피처임포턴스가 전체 중요도에서 하위 20-25% 컬럼들 제거
# #재구성후
# #모델을 돌려서 결과 도출

# # 기존 모델들과 성능 비교

# # 모델 4가지 구성

# import numpy as np
# from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_digits,load_boston,fetch_california_housing
# from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.pipeline import make_pipeline  
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt

# #1.데이터
# datasets1 = [load_iris(),
#             load_breast_cancer(),
#             load_wine(),
#             fetch_covtype(),
#             load_digits()]

# datasets2 = [load_boston(),
#              fetch_california_housing()]

# models_name1 =[
#     'load_iris',
#     'load_breast_cancer',
#     'load_wine',
#     'fetch_covtype',
#     'load_digits'
# ]

# model_name2 = [
#     'load_boston',
#     'fetch_california_housing'
# ]

# models = [
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     GradientBoostingClassifier(),
#     XGBClassifier()
# ]

# for i , v in enumerate(datasets1):
#     datasets1 = v
#     x = v.data
#     y = v.target

#     x_train,x_test,y_train,y_test = train_test_split(
#     x,y,train_size=0.8, shuffle=True, random_state=337)

#     scaler = MinMaxScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)


#     #3. 컴파일, 훈련 및 평가
#     # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
#     # ax = axes.ravel()

#     for i, model in enumerate(models):
#         model.fit(x_train, y_train)
#         y_pred = model.predict(x_test)
#         acc = accuracy_score(y_test, y_pred)
#         print(f"{model.__class__.__name__} accuracy_score: {acc:.4f}")
#         print(f"{model.__class__.__name__} feature importances:", model.feature_importances_)

#     # # 그림그리기
#     #     n_features = x.shape[1]
#     #     ax[i].barh(np.arange(n_features), model.feature_importances_, align='center')
#     #     ax[i].set_yticks(np.arange(n_features))
#     #     ax[i].set_yticklabels(v.feature_names)
#     #     ax[i].set_xlabel('Feature Importances')
#     #     ax[i].set_ylabel('Features')
#     #     ax[i].set_ylim(-1, n_features)
#     #     ax[i].set_title(model.__class__.__name__)

# # plt.show()

##########################################
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
num_classes = 3
params = {
    'objective': 'multi:softmax',
    'num_class': num_classes
}
#1. 데이터

data_list = [load_iris(), load_breast_cancer(), load_digits(), load_wine()]
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier(objective='multi:softmax', num_class=num_classes)]
# scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()] 

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state= 337, stratify=y
    )

    for j, model in enumerate(model_list): 
            # 모델 훈련
            model.fit(x_train, y_train)
            test_acc =model.score(x_test, y_test)
            

            # feature importance 계산
            feature_importance = model.feature_importances_
            feature_importance_percent = feature_importance / feature_importance.sum()

            # 중요도가 낮은 컬럼 제거
            threshold = np.percentile(feature_importance_percent, 25) # 하위 25% /  함수는 배열에서 지정된 백분율의 값을 계산합니다.
            
            #feature_importance_percent의 값이 threshold 이상인 위치를 선택하여 feature_idx에 저장
            feature_idx = np.where(feature_importance_percent >= threshold)[0] # 배열에서 25%에 해당하는 값 이하의 모든 값을 선택하려면 / where() 함수는 조건을 만족하는 위치를 반환
            
            #selected_x_train과 selected_x_test는 x_train과 x_test에서 선택된 feature만을 포함하는 새로운 배
            selected_x_train = x_train[:, feature_idx]
            selected_x_test = x_test[:, feature_idx]

            # 모델 훈련 후 정확도 계산
            model.fit(selected_x_train,y_train)
            test_acc_del = model.score(selected_x_test, y_test)

            # 결과 출력
            print(f"Dataset: {data_list[i].DESCR.splitlines()[0]}")
            print(f"Model: {type(model).__name__}")  # type(model).__name__는 모델 객체의 클래스 이름을 가져와서 출력
            
            #feature importance를 계산한 후 선택된 feature의 개수와 전체 feature의 개수를 출력
            print(f"Selected Features: {len(feature_idx)} / {x_train.shape[1]}") 
            
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Accuracy Del: {test_acc_del:.4f}")
            
            print("=" * 50)