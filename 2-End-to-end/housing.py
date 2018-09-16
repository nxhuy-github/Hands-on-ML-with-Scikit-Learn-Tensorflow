import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


housing = pd.read_csv('housing.csv')
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

# Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace=True)

# Visualization California housing prices
#housing = strat_train_set.copy()
#housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1,
#             s=housing['population']/100, label='population',
#             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
#plt.legend()
#plt.show()

'''
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, .2)
print(housing.describe())
print(len(train_set), 'train +', len(test_set), 'test')
'''

# Correlations between the attributes
#housing['rooms_per_household'] = housing['total_rooms']/housing['households']
#housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
#housing['population_per_household'] = housing['population']/housing['households']
#corr_matrix = housing.corr()
#print(corr_matrix)
#print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Data cleaning
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
#print(imputer.statistics_)
#print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr= pd.DataFrame(X, columns=housing_num.columns)
#print(housing_tr.head())

# Handling text and categorical attributes
# Label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
#print(housing_cat_encoded)
#print(encoder.classes_)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot.toarray())






























