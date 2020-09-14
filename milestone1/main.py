import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Read the data
train = pd.read_csv('train.csv')
nptrain = train.to_numpy()

# pull data into target (y) and predictors (X)
train_y = train.Horizontal_Distance_To_Fire_Points

# Soil_Type
soil = train.Soil_Type
soil_1 = [int(x / 1000) for x in soil]
soil_2 = [int((x % 1000) / 100) for x in soil]
soil_1 = np.array(soil_1).reshape(len(soil_1), 1)
soil_2 = np.array(soil_2).reshape(len(soil_2), 1)

train_x = nptrain[:, 1:-2]

enc = OneHotEncoder()
cli = enc.fit_transform(soil_1).toarray()
geo = enc.fit_transform(soil_2).toarray()

train_x = np.hstack((train_x, np.delete(cli, 1, 1), np.delete(geo, 1, 1)))

# transform aspect into x, y
aspect = train.Aspect
asp_x = np.cos(aspect / 180 * np.pi)
asp_y = np.sin(aspect / 180 * np.pi)
asp_x = np.array(asp_x).reshape(len(asp_x), 1)
asp_y = np.array(asp_y).reshape(len(asp_y), 1)
train_x = np.hstack((train_x, asp_x, asp_y))
# train_x = np.delete(train_x, 1, 1)

# print(train_x)
# Create training predictors data
# train_X = train[predictor_cols]
#
my_model = LinearRegression().fit(train_x, train_y)
# print(my_model.score(train_x, train_y))
res = my_model.predict(train_x)
print(np.sqrt(mean_squared_error(res, train_y)))

# test = pd.read_csv('test.csv')
# # test_x = test.drop('Horizontal_Distance_To_Fire_Points', axis=1)
# test_x = test.drop('ID', axis=1)
# test_x = test.drop('Soil_Type', axis=1)
# test_y = test.Horizontal_Distance_To_Fire_Points
#
# soil = test.Soil_Type
# soil_1 = [int(x / 1000) for x in soil]
# soil_2 = [int((x % 1000) / 100) for x in soil]
# soil_1 = np.array(soil_1).reshape(len(soil_1), 1)
# soil_2 = np.array(soil_2).reshape(len(soil_2), 1)
# cli = enc.fit_transform(soil_1).toarray()
# geo = enc.fit_transform(soil_2).toarray()
# test_x = np.hstack((test_x, np.delete(cli, 1, 1), np.delete(geo, 1, 1)))
#
# aspect = test.Aspect
# asp_x = np.cos(aspect / 180 * np.pi)
# asp_y = np.sin(aspect / 180 * np.pi)
# asp_x = np.array(asp_x).reshape(len(asp_x), 1)
# asp_y = np.array(asp_y).reshape(len(asp_y), 1)
# train_x = np.hstack((train_x, asp_x, asp_y))
# train_x = np.delete(train_x, 1, 1)
# res = my_model.predict(test_x)
# print(np.sqrt(mean_squared_error(res, test_y)))
