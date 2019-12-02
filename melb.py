#!/usr/bin/env python

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

melbourne_file_path = 'melb_data.csv'

# read file and store data in pandas DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
print(melbourne_data.describe())

# print the column headers
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# dot-notation pulls out a Series, which is broadly like a DataFrame with only a single column
# select the column we want to predict (called the prediction target, called y by convention)
y = melbourne_data.Price

# select multiple features by providing a list of column names. By convention, this data is called X
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(X.describe())

# print the top few rows
print(X.head())

# define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
print(melbourne_model.fit(X, y))

print("\nMaking predictions for the following 5 houses:")
print(X.head())
print("\nThe predictions are:")
print(melbourne_model.predict(X.head()))

print("\nThe mean absolute error is:")
predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
