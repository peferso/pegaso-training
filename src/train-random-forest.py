import os
import pymysql
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def fetch_database_data():
    connection = pymysql.connect(host=os.environ['DBHOST'],
                                 user=os.environ['DBUSER'],
                                 passwd=os.environ['DBPASS'],
                                 db="pegaso_db",
                                 charset='utf8')
    sql_query = pd.read_sql_query("""select  
                                        brand, model, price_c, kilometers, power, 
                                        doors, professional_vendor, automatic_gearbox, year, batch_ts 
                                      from 
                                        raw_data;""", connection)
    dfrd = pd.DataFrame(sql_query,
                        columns=['brand', # One-hot
                                 'model', # One-hot
                                 'price_c',
                                 'kilometers',
                                 'power',
                                 'doors',
                                 'professional_vendor', # One-hot
                                 'automatic_gearbox',  # One-hot
                                 'year',
                                 'batch_ts'
                                 ])
    return dfrd

def build_features(df):
    # Compute how many years old is each item
    l_years = []
    for index, row in df.iterrows():
        years = row['batch_ts'].year - int(row['year'])
        l_years.append(years)
    df['years'] = l_years

    # drop useless columns
    df = df.drop('batch_ts', axis=1)
    df = df.drop('year', axis=1)

    df = df.drop('professional_vendor', axis=1)
    df = df.drop('automatic_gearbox', axis=1)
    df = df.drop('model', axis=1)

    df = df.dropna()

    # One-hot encoding TO TEST https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    features = pd.get_dummies(df)  # Display the first 5 rows of the last 12 columns
    return features

def convert_to_arrays(features):
    # Labels are the values we want to predict
    labels = np.array(features['price_c'])  # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('price_c', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)
    return feature_list, features, labels

df = fetch_database_data()

features = build_features(df)

feature_list, features, labels = convert_to_arrays(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)# Calculate the absolute errors
errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
