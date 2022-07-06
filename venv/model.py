from dmba import classificationSummary
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression # for running logistic regression model
from sklearn.model_selection import train_test_split # for splitting the data into train and test 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # we will use a function "logit" from "smf" to fit an logistic regression model
import pickle

df = pd.read_csv('neo.csv')
#print(df.head())

nasa_df = df.drop(columns=['id', 'name', 'orbiting_body', 'sentry_object', 'est_diameter_min', 'est_diameter_max'])

nasa_df['hazardous'] = nasa_df['hazardous'].astype('int')

#Explanatory Model

# logit_reg = smf.logit(formula = "hazardous ~ relative_velocity + miss_distance + absolute_magnitude", data = nasa_df).fit()

# print(logit_reg.summary())

#Predictive Model

X = nasa_df.drop(columns = ['hazardous'])
y = nasa_df['hazardous']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)

pred_model = LogisticRegression() #initialize the model

pred_model = pred_model.fit(train_X, train_y)

# Predicting the test set results
test_pred = pred_model.predict(test_X)

# print(classificationSummary(train_y, train_pred))
# print(classificationSummary(test_y, test_pred))

#Saving model to disk
pickle.dump(pred_model, open('model.pkl', 'wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
