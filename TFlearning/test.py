import pandas as pd
print (pd.__version__)


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([94939, 39945,20239])

pd.DataFrame({'City names': city_names, 'Population': population})


california_housing_dataframe = pd.read_csv('C:\\Users\\Administrator\\Desktop\\PyML-Learning\\TFlearning\\california_housing_train.csv', sep=',')
california_housing_dataframe.describe()