import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# preprocess dataset
train = train.drop(['ID'], axis=1)
test = test.drop(['ID'], axis=1)

# extract first two digits of soil type feature
train['Soil_Type'] = train['Soil_Type'].astype(str)
test['Soil_Type'] = test['Soil_Type'].astype(str)
train_cli = train['Soil_Type'].str.get(0).astype(int)
test_cli = test['Soil_Type'].str.get(0).astype(int)
train_geo = train['Soil_Type'].str.get(1).astype(int)
test_geo = test['Soil_Type'].str.get(1).astype(int)
train['climate'] = train_cli
train['geology'] = train_geo
test['climate'] = test_cli
test['geology'] = test_geo

# train = pd.concat([train, train_cli, train_geo], axis=1).drop(['Soil_Type'], axis=1)
# test = pd.concat([test, test_cli,arwie test_geo], axis=1).drop(['Soil_Type'], axis=1)
# train = train.rename(columns={'train_cli':'climate', 'train_geo':'geology'})
# test = test.rename(columns={'test_cli':'climate', 'test_geo':'geology'})
# train = train.insert(12, 'climate', train_cli)
# test = test.insert(12, 'climate', test_cli)
# train = train.insert(13, 'geology', train_geo)
# test = test.insert(13, 'geology', test_geo)
train = train.drop(['Soil_Type'], axis=1)
test = test.drop(['Soil_Type'], axis=1)

# extract features and label
xTr = train.drop(['Cover_Type'], axis=1)
yTr = train['Cover_Type']
xTe = test

# dummies
xTr_area_dummies = pd.get_dummies(xTr['Wilderness_Area'], prefix='area_dummies')
xTr_climate_dummies = pd.get_dummies(xTr['climate'], prefix='climate_dummies')
xTr_geology_dummies = pd.get_dummies(xTr['geology'], prefix='geology_dummies')
xTr = pd.concat([xTr, xTr_area_dummies], axis=1).drop(['Wilderness_Area'], axis=1)
xTr = pd.concat([xTr, xTr_climate_dummies], axis=1).drop(['climate'], axis=1)
xTr = pd.concat([xTr, xTr_geology_dummies], axis=1).drop(['geology'], axis=1)

xTe_area_dummies = pd.get_dummies(xTe['Wilderness_Area'], prefix='area_dummies')
xTe_climate_dummies = pd.get_dummies(xTe['climate'], prefix='climate_dummies')
xTe_geology_dummies = pd.get_dummies(xTe['geology'], prefix='geology_dummies')
xTe = pd.concat([xTe, xTe_area_dummies], axis=1).drop(['Wilderness_Area'], axis=1)
xTe = pd.concat([xTe, xTe_climate_dummies], axis=1).drop(['climate'], axis=1)
xTe = pd.concat([xTe, xTe_geology_dummies], axis=1).drop(['geology'], axis=1)



# feature process after feature importance analysis
def feature_transform(data):
    data['Elevation_2'] = data['Elevation']**2
    data['Horizontal_Distance_To_Roadways_2'] = data['Horizontal_Distance_To_Roadways']**2
    data['Horizontal_Distance_To_Fire_Points_2'] = data['Horizontal_Distance_To_Fire_Points']**2
    data['Dist_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2 + data['Vertical_Distance_To_Hydrology']**2) ** 0.5
    
    #horizontal distance to hydrology
    data['Horizontal_Distance_To_Hydrology_add_Roadways'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']
    data['Horizontal_Distance_To_Hydrology_minus_Roadways'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Roadways']
    data['Horizontal_Distance_To_Hydrology_add_Fire_Points'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Fire_Points']
    data['Horizontal_Distance_To_Hydrology_minus_Fire_Points'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Fire_Points']
    data['Horizontal_Distance_To_Hydrology_add_Elevation'] = data['Horizontal_Distance_To_Hydrology'] - data['Elevation']
    data['Horizontal_Distance_To_Hydrology_minus_Elevation'] = data['Horizontal_Distance_To_Hydrology'] - data['Elevation']
    
    data['Horizontal_Distance_To_Fire_Points_times_Roadway'] = data['Horizontal_Distance_To_Fire_Points'] * data['Horizontal_Distance_To_Roadways']
    data['Horizontal_Distance_To_Fire_Points_times_Elevation'] = data['Horizontal_Distance_To_Fire_Points'] * data['Elevation']
    data['Horizontal_Distance_To_Roadways_times_Elevation'] = data['Horizontal_Distance_To_Roadways'] * data['Elevation']
    data['Horizontal_Distance_To_Roadways_times_Elevation_times_Fire_Points'] = data['Horizontal_Distance_To_Roadways'] * data['Elevation'] * data['Horizontal_Distance_To_Fire_Points']
    
    data['Elevation_add_Horizontal_Distance_To_Roadways'] = data['Horizontal_Distance_To_Roadways'] + data['Elevation']
    data['Elevation_add_Horizontal_Distance_To_Fire_Points'] = data['Horizontal_Distance_To_Fire_Points'] + data['Elevation']
    data['Horizontal_Distance_To_Roadways_add_Fire_Points'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Fire_Points']
    data['Elevation_minus_Horizontal_Distance_To_Roadways'] = data['Elevation'] - data['Horizontal_Distance_To_Roadways']
    data['Elevation_minus_Horizontal_Distance_To_Fire_Points'] = data['Elevation'] - data['Horizontal_Distance_To_Fire_Points']
    data['Horizontal_Distance_To_Roadways_minus_Fire_Points'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Fire_Points']
    return data


# xTr = feature_transform(xTr)
# xTe = feature_transform(xTe)


# normalization
def normal(data):
    for col in data.columns:
        if ('area_dummies' not in col) and ('climate_dummies' not in col) and ('geology_dummies' not in col):
            # print(col)
            data[col] = stats.zscore(data[col])
    return data


xTr = normal(xTr)
xTe = normal(xTe)
# xTr = stats.zscore(xTr.iloc[:, :(xTr.columns.get_loc("Wilderness_Area"))])
# xTe = stats.zscore(xTe.iloc[:, :(xTe.columns.get_loc("Wilderness_Area"))])

# encode label
label_encoder = LabelEncoder()
yTr = label_encoder.fit_transform(yTr)

# train model
# model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 40, 40, 40, 20), random_state=1)
# model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1, learning_rate='adaptive')
model = ExtraTreesClassifier()
model.fit(xTr, yTr)

#
importance = model.feature_importances_ # clf is the model
sorted_idx = np.argsort(importance)[::-1]
for index in sorted_idx:
   print(xTr.columns[index], importance[index])

# train prediction
# train_pred = model.predict(xTr)
# train_pred = label_encoder.inverse_transform(train_pred)
# print("Train accuracy:", np.sum(train_pred == train['Cover_Type']))

# test prediction
# test_pred = model.predict(xTe)
# test_pred = label_encoder.inverse_transform(test_pred)
#
# # output res
# res = pd.read_csv('sample_submission.csv')
# res.loc[:, 'Cover_Type'] = test_pred
# res.to_csv('test_res2.csv', index=False)
