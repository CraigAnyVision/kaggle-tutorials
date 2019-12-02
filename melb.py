#!/usr/bin/env python

from sys import maxsize

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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

print("\nThe MAE is:")
predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define & fit model
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print("\nThe validated MAE is:")
print(mean_absolute_error(val_y, val_predictions), '\n')


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


min_mae = maxsize
opt_mln = 0

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
    if my_mae < min_mae:
        min_mae = my_mae
        opt_mln = max_leaf_nodes

print("Min MAE: %d, \t Optimal MLN: %d" % (min_mae, opt_mln))
