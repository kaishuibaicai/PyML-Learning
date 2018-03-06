import pandas as pd
print (pd.__version__)


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([94939, 39945,20239])

cities = pd.DataFrame({'City names': city_names, 'Population': population})

print (cities['City names'])
california_housing_dataframe = pd.read_csv('C:\\Users\\Administrator\\Desktop\\PyML-Learning\\TFlearning\\california_housing_train.csv', sep=',')
print (california_housing_dataframe.describe())
print (california_housing_dataframe.head())