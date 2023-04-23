#m24 카피
# 부스팅 계열?은 결측치에 강하다 / 범위로 찾기때문에 스케일러 사용안해줘도됨

import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline  
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#1.데이터
# datasets = load_iris()
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
x,y,train_size=0.8, shuffle=True, random_state=337)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier()

model.fit(x_train,y_train)

#4.평가 예측
result = model.score(x_test,y_test)
print('model.score :',result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('accuracy_score :', acc)

# # 컬럼 중요도 가시화 / 트리계열은 제공됨
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Fetures')
#     plt.ylim(-1,n_features)
#     plt.title(model)
    
# plot_feature_importances(model)
# plt.show()

# 중요한 컬럼 순으로 중요도를 뽑을수있다.
from xgboost.plotting import plot_importance
plot_importance(model) 
plt.show() 






