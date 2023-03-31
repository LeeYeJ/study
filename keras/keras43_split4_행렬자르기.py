import numpy as np

datasets = np.array(range(1,41)).reshape(10,4)
print(datasets)
print(datasets.shape) #(10, 4)

x_data = datasets[:,:-1]
y_data = datasets[:,-1]
print(x_data,y_data) #(10, 3) (10,)
print(x_data.shape,y_data.shape)

timesteps = 3
##############x만들기################
def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps ): # 이 두 줄을 잘 조절하자.
        subset = dataset[i:(i + timesteps)]    #
        aaa.append(subset)
    return np.array(aaa)

x_data = split_x(x_data, timesteps)
print(x_data)
print(x_data.shape)

###########y만들기###############
y_data = y_data[timesteps:]
print(y_data)

