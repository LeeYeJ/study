import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape input data
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0

n_components = [154, 331, 486, 713]
accuracys =[]
for i in n_components:
    # Apply PCA
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # Define DNN model
    model = Sequential()
    model.add(Dense(128, input_shape=(i,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train DNN model
    model.fit(x_train_pca, y_train_one_hot, epochs=5, batch_size=32)

    # Evaluate DNN model
    _, accuracy = model.evaluate(x_test_pca, y_test_one_hot, verbose=0)
    accuracys.append(accuracy)
    print(f'PCA n_components={i}, Accuracy: {accuracy:.3f}')
print(accuracys)  

'''
PCA n_components=154, Accuracy: 0.975
PCA n_components=331, Accuracy: 0.974
PCA n_components=486, Accuracy: 0.974
PCA n_components=713, Accuracy: 0.971
[0.9746999740600586, 0.9740999937057495, 0.973800003528595, 0.9714999794960022]
'''  
    # result = model.evaluate(x_test,y_test)
    # y_pred = model.predict(x_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_test = np.argmax(y_test, axis=1)
    # acc = accuracy_score(y_pred,y_test)
    
    # results.append(acc)
    # print('PCA 가',i,'acc :',acc)