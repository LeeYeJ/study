import matplotlib.pyplot as plt
import matplotlib
#선그어주기
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.figure(figsize=(8,6)) # 그래프 사이즈
plt.title('캘리포니아')
plt.xlabel('에포')
plt.ylabel('에러')
plt.legend()
plt.grid()
plt.show()




