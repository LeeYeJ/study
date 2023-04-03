import numpy as np

aaa = np.array([1,2,3])

bbb = aaa
print(bbb)

bbb[0] = 4
print(bbb)
print(aaa) # 넘파이 메모리 주소값이 공유돼서 aaa도 바뀜

print('======================')

ccc = aaa.copy() 
ccc[1] =7

print(ccc)
print(aaa) # copy() 써주면 안바뀜