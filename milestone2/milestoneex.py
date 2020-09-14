import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb


from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('/content/drive//My Drive/517_application/train_2.csv')
test = pd.read_csv('/content/drive//My Drive/517_application/test_2.csv')


X = train[['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area', 'Soil_Type']]
y = train['Cover_Type']

def z_score(data):
    for col in data.columns:
        if ('Soil_Type' not in col) and ('Wilderness_Area' not in col):
            data[col] = zscore(data[col])
    return data


XX = X[['Wilderness_Area', 'Soil_Type']]
Wilderness_Area_dummies = pd.get_dummies(X['Wilderness_Area'], prefix='Wilderness_Area')
X = pd.concat([X, Wilderness_Area_dummies], axis=1).drop(columns=['Wilderness_Area'], axis=1)
Soil_Type_dummies = pd.get_dummies(X['Soil_Type'], prefix='Soil_Type')
X = pd.concat([X, Soil_Type_dummies], axis=1).drop(columns=['Soil_Type'], axis=1)

##### feature engineering
X['Dist_to_Hydrolody'] = (X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2) ** 0.5
X['Elevation_2'] = X['Elevation']**2
X['Horizontal_Distance_To_Roadways_2'] = X['Horizontal_Distance_To_Roadways']**2
X['Horizontal_Distance_To_Fire_Points_2'] = X['Horizontal_Distance_To_Fire_Points']**2

X['Horizontal_Distance_To_Fire_Points_Roadway'] = X['Horizontal_Distance_To_Fire_Points']*X['Horizontal_Distance_To_Roadways']  # 3
X['Horizontal_Distance_To_Roadways_Elevation'] = X['Horizontal_Distance_To_Roadways']*X['Elevation']  # 3
X['Horizontal_Distance_To_Fire_Points_Elevation'] = X['Horizontal_Distance_To_Fire_Points']*X['Elevation']  # 3 add one

X['Elev_p_HDR'] = X['Elevation']+X['Horizontal_Distance_To_Roadways']  # 4
X['Elev_m_HDR'] = X['Elevation']-X['Horizontal_Distance_To_Roadways']  # 4
X['Elev_p_HDFP'] = X['Elevation']+X['Horizontal_Distance_To_Fire_Points']  # 4
X['Elev_m_HDFP'] = X['Elevation']-X['Horizontal_Distance_To_Fire_Points']  # 4
X['Fire_p_Road'] = X['Horizontal_Distance_To_Fire_Points']+X['Horizontal_Distance_To_Roadways']  # 4
X['Fire_m_Road'] = X['Horizontal_Distance_To_Fire_Points']-X['Horizontal_Distance_To_Roadways']  # 4


X['Hydro_p_Road'] = X['Horizontal_Distance_To_Hydrology']+X['Horizontal_Distance_To_Roadways']  # 2
X['Hydro_p_Road'] = X['Horizontal_Distance_To_Hydrology']-X['Horizontal_Distance_To_Roadways']  # 2
X['Hydro_p_Fire'] = X['Horizontal_Distance_To_Hydrology']+X['Horizontal_Distance_To_Fire_Points']  # 2
X['Hydro_m_Fire'] = X['Horizontal_Distance_To_Hydrology']-X['Horizontal_Distance_To_Fire_Points']  # 2
X['Elev_p_VDH'] = X['Elevation']+X['Vertical_Distance_To_Hydrology']
X['Elev_m_VDH'] = X['Elevation']-X['Vertical_Distance_To_Hydrology']
X['Elev_p_HDH'] = X['Elevation']+X['Horizontal_Distance_To_Hydrology']  # 2
X['Elev_m_HDH'] = X['Elevation']-X['Horizontal_Distance_To_Hydrology']  # 2

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# z_score(X_train)
# z_score(X_test)
z_score(X)



from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier()
et_clf.fit(X_train, y_train)
score = et_clf.score(X_test, y_test)
score