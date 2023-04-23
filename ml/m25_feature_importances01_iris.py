# 10개 데이터셋
# 10개의 파일을 만든다.
# 모델 RF
# [실습 과제] 피처 한개씩 삭제하고 성능 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline  
import matplotlib.pyplot as plt
from sklearn.svm import SVC

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y, random_state=1234, shuffle=True
)

sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = RandomForestClassifier()

model.fit(x_train,y_train)

#4.평가 예측
result = model.score(x_test,y_test)
print('model.score :',result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('accuracy_score :', acc)

# 컬럼 중요도 가시화
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Fetures')
    plt.ylim(-1,n_features)
    plt.title(model)
    
plot_feature_importances(model)
plt.show()