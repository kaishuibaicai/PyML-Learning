import numpy as np 
import matplotlib.pyplot as plt 

x, y = [], []

for sample in open('C:\Users\Administrator\Desktop\PyML-Learning\\a_FirstExample\prices.txt', 'r'):
	_x, _y = sample.split(',')
	x.append(float(_x))
	y.append(float(_y))


x, y = np.array(x), np.array(y)

x = (x - x.mean()) / x.std()

plt.figure()
plt.scatter(x, y, c='g', s=6)
plt.show()