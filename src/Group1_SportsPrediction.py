# %% [markdown]
# ## Import Modules

# %%
# import modules
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

# %% [markdown]
# ## Loading Data

# %%
# read data
training_data = pd.read_csv('../data/players_21.csv')
new_testing_data = pd.read_csv('../data/players_22.csv')

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Remove columns with na values that exceed 30%

# %%
# Calculate the percentage of na values in each column
na_percentages = training_data.isna().sum() / len(training_data) * 100

# Select the columns where the percentage of na values exceeds 30%
cols_to_drop = na_percentages[na_percentages > 30].index

# Drop the selected columns
training_data.drop(cols_to_drop, axis=1, inplace=True)
new_testing_data.drop(cols_to_drop, axis=1, inplace=True)

# %% [markdown]
# ### Remove columns where the values do not obviously contribute a player's overall rating

# %%
# urls do not help to predict a player's rating
# remove columns that have 'url' in their name
cols_to_drop = [col for col in training_data.columns if 'url' in col]
training_data.drop(cols_to_drop, axis=1, inplace=True)
new_testing_data.drop(cols_to_drop, axis=1, inplace=True)

# %%
# columns that obviously do not contribute to a player's rating
cols_to_drop = [
    "age",
    "sofifa_id",
    "short_name",
    "long_name",
    "real_face",
]

# drop the columns
training_data.drop(cols_to_drop, axis=1, inplace=True)
new_testing_data.drop(cols_to_drop, axis=1, inplace=True)

# %%
# columns where input would not be available at the time of prediction
cols_to_drop = [
    "gk", "rb", "rcb", "cb", "lcb", "lb", "rwb", "rdm", "cdm", "ldm", "lwb", "rm", "rcm", "cm", "lcm", "lm", "ram", "cam", "lam", "rw", "rf", "cf", "lf", "lw", "rs", "st", "ls", "club_joined", "club_contract_valid_until"
]

# drop the columns
training_data.drop(cols_to_drop, axis=1, inplace=True)
new_testing_data.drop(cols_to_drop, axis=1, inplace=True)

# %%
# remove club, national and league info. They do not explicitly determine a player's rating
cols_to_drop = [
    "club_name", "league_name", "league_level", "club_jersey_number", "nationality_id", "nationality_name", "value_eur", "release_clause_eur", "club_team_id"
]

# drop the columns
training_data.drop(cols_to_drop, axis=1, inplace=True)
new_testing_data.drop(cols_to_drop, axis=1, inplace=True)

# %% [markdown]
# #### Encoding data

# %%
training_data.info()

# %%
# use pd.factorize to convert categorical columns to numerical
# check if dtype is object

# get categorical columns
cat_cols = [col for col in training_data.columns if training_data[col].dtype == 'object']

# factorize the categorical columns
for col in cat_cols:
    training_data[col], c1 = pd.factorize(training_data[col])
    new_testing_data[col], c2 = pd.factorize(new_testing_data[col])

# %%
training_data.info()

# %% [markdown]
# #### Imputing Data

# %%
imputer = SimpleImputer(strategy='most_frequent')
training_data = pd.DataFrame(imputer.fit_transform(training_data), columns=training_data.columns)
new_testing_data = pd.DataFrame(imputer.transform(new_testing_data), columns=new_testing_data.columns)

# %%
training_data.info()

# %% [markdown]
# ### Setup training and testing data

# %%
trainX = training_data.drop('overall', axis=1)
trainY = training_data['overall']
new_testX = new_testing_data.drop('overall', axis=1)
new_testY = new_testing_data['overall']

# %% [markdown]
# #### Scaling the independent variables

# %%
scaler = StandardScaler()
trainX = pd.DataFrame(scaler.fit_transform(trainX), columns=trainX.columns)
new_testX = pd.DataFrame(scaler.transform(new_testX), columns=new_testX.columns)

# %%
trainX.info()

# %% [markdown]
# #### Create feature subsets that better correlate with the overall rating

# %%
# create feature subsets which show better correlation with the overall rating

# create a list of all the columns with a correlation greater than 0.5
feature_cols = list(trainX.corrwith(trainY)[abs(trainX.corrwith(trainY)) > 0.5].index)

print(feature_cols)
print(len(feature_cols))

# %%
# set trainX and testX to the new feature subset
trainX = trainX[feature_cols]
new_testX = new_testX[feature_cols]

# %%
trainX.info()

# %% [markdown]
# ## Training & Evaluating Models

# %%
X = trainX
y = trainY

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Cross validation with RandomForest

# %%
# do cross validation training with either RandomForest, XGBoost, Gradient Boost Regressors that can predict a player rating.

# RandomForestRegressor cross validation training
cv = KFold(n_splits=3)

# parameters for the RandomForestRegressor
PARAMETERS = {
    "max_depth": [12,35, 40],
    "n_estimators": [100, 500, 1000]

}

rf = RandomForestRegressor()
model_rf = GridSearchCV(rf, cv=cv, param_grid=PARAMETERS, scoring="neg_mean_squared_error")
model_rf.fit(X_train, y_train)
model_rf.best_params_

# %%

y_pred = model_rf.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# #### Fine Tuning

# %%
model_rf = RandomForestRegressor(max_depth=40, n_estimators=1000)
model_rf.fit(trainX, trainY)

# %% [markdown]
# ### Cross validation with XGBoost

# %%
cv = KFold(n_splits=3)

# parameters for the XGBRegressor
PARAMETERS = {
    "max_depth": [12,35, 40],
    "learning_rate":[0.3, 0.1, 0.03],
    "n_estimators": [100, 500, 1000]
}

model_xgb = XGBRegressor()
model_xgb_gs = GridSearchCV(model_xgb, cv=cv, param_grid=PARAMETERS, scoring="neg_mean_absolute_error")
model_xgb_gs.fit(X_train, y_train)
model_xgb_gs.best_params_

# %%
y_pred = model_xgb_gs.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# #### Fine Tuning

# %%
model_xgb = XGBRegressor(learning_rate=0.03, max_depth=12, n_estimators=500)
model_xgb.fit(trainX, trainY)

# %% [markdown]
# Cross validation with AdaBoost

# %%
cv = KFold(n_splits=4)

PARAMETERS ={
    "random_state":[12,25, 36, 48],
    # "min_child_weight":[1,5,15],
    "learning_rate":[0.003, 0.1, 0.03],
    "n_estimators":[100,500,1000]
}

ada = AdaBoostRegressor()
model_ada = GridSearchCV(ada,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
model_ada.fit(X_train, y_train)
model_ada.best_params_

# %%
y_pred = model_ada.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# #### Fine Tuning

# %%
model_ada = AdaBoostRegressor(random_state=12, learning_rate=0.03, n_estimators=500)
model_ada.fit(X_train, y_train)

# %% [markdown]
# ## Testing with new dataset

# %% [markdown]
# Random Forest

# %%
y_pred = model_rf.predict(new_testX)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(new_testY, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(new_testY, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(new_testY, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(new_testY, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# XGBoost

# %%
y_pred = model_xgb.predict(new_testX)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(new_testY, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(new_testY, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(new_testY, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(new_testY, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# AdaBoost

# %%
y_pred = model_ada.predict(new_testX)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(new_testY, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(new_testY, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(new_testY, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(new_testY, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% [markdown]
# ## Save model

# %%
# save the best modelimport pickle
pickle.dump(model_xgb, open('../models/model_xgb.pkl', 'wb'))

# %% [markdown]
# Best model is XGBoost.


