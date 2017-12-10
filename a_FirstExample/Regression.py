# 导入需要用的库
import numpy as np 
import matplotlib.pyplot as plt 

# 定义存储输入数据（x）和目标数据（y）的数组
x, y = [], []

# 遍历数据集，变量sample对应的正是一个个样本
for sample in open('C:\Users\Administrator\Desktop\PyML-Learning\\a_FirstExample\prices.txt', 'r'):
	# 由于数据是由逗号隔开的，所以调用Python中的split方法并将逗号作为参数传入
	_x, _y = sample.split(',')
	# 将字符串数据转换为浮点数
	x.append(float(_x))
	y.append(float(_y))

 
x, y = np.array(x), np.array(y)

x = (x - x.mean()) / x.std()

# plt.figure()
# plt.scatter(x, y, c='g', s=6)
# plt.show()

x0 = np.linspace(-2, 4, 100)
def get_model(deg):
	return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)


def get_cost(deg, input_x, input_y):
	return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


test_set = (1, 4, 10)
for d in test_set:
	print(get_cost(d, x, y))


plt.scatter(x, y, c="g", s=20)
for d in test_set:
	plt.plot(x0, get_model(d)(), label="degree = {}".format(d))

plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)

plt.legend()
plt.show()