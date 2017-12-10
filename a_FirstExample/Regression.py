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

# 读取完数据后，将它们转化为numpy数组以方便进一步的处理
x, y = np.array(x), np.array(y)

# 标准化
x = (x - x.mean()) / x.std()
# 将原始数据以散点图的形式画出
# plt.figure()
# plt.scatter(x, y, c='g', s=6)
# plt.show()

# 在（-2,4）这个区间上取100个点作为画图的基础
x0 = np.linspace(-2, 4, 100)
# 利用numpy的函数定义训练并且返回多项式回归模型的函数
# deg参数代表模型中的n，亦即模型中的多项式的次数
# 返回的模型能够根据输入的x(默认是x0)，返回相对应的预测的y
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