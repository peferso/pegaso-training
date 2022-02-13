import os
import pymysql
import datetime
import pandas as pd
import numpy as np
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

logging.basicConfig(
    format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
    level=logging.INFO)

def fetch_database_data():
    time_start = time.time()
    logging.info('Start')
    connection = pymysql.connect(host=os.environ['DBHOST'],
                                 user=os.environ['DBUSER'],
                                 passwd=os.environ['DBPASS'],
                                 db="pegaso_db",
                                 charset='utf8')
    sql_query = pd.read_sql_query("""SELECT  
                                        brand, LTRIM(model), price_c, kilometers, power, 
                                        doors, professional_vendor, automatic_gearbox, year, batch_ts 
                                      FROM 
                                        raw_data
                                      WHERE
                                        brand IN (SELECT nb.brand FROM brands_count nb WHERE nb.num_cars>100)
                                      ;""", connection)
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
    time_end = time.time()
    logging.info('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')
    return dfrd

def build_features(df):
    time_start = time.time()
    logging.info('Start')

    logging.info('Compute new variable \'years\'...')
    l_years = []
    for index, row in df.iterrows():
        years = row['batch_ts'].year - int(row['year'])
        l_years.append(years)

    df['years'] = l_years
    logging.info('Compute new variable \'years\'. Done.')

    logging.info('Dropping useless columns...')
    df = df.drop('batch_ts', axis=1)
    df = df.drop('year', axis=1)
    df = df.drop('professional_vendor', axis=1)
    df = df.drop('automatic_gearbox', axis=1)
    df = df.drop('model', axis=1)
    logging.info('Dropping useless columns. Done.')

    logging.info('Dropping rows with \'nans\'...')
    df = df.dropna()
    logging.info('Dropping rows with \'nans\'. Done.')

    l_avprice = []

    logging.info('Getting average price of each car based on brand...')
    t1 = time.time()
    irow = 0
    for index, row in df.iterrows():
        t2 = time.time()
        brand = row['brand']
        avprice = np.mean(df[df['brand'] == brand]['price_c'])
        l_avprice.append(avprice)
        if t2 - t1 > 10:
            logging.info('  ' + str(index) + ' rows processed. ' +
                         str(round(t2 - time_start, 2)) + ' seconds elapsed - ' +
                         str(round((df.shape[0] - irow)/(index - irow)*(t2 - t1), 2)) +
                         ' seconds to finish...')
            t1 = time.time()

    logging.info('Getting average price of each car based on brand. Done.')

    df_baseline = pd.DataFrame(l_avprice, columns=['av_price'])

    # One-hot encoding TO TEST https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    features = pd.get_dummies(df)
    time_end = time.time()
    logging.info('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')
    return features, df_baseline

def convert_to_arrays(features):
    # Labels are the values we want to predict
    labels = np.array(features['price_c'])  # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('price_c', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)
    return feature_list, features, labels

def initial_checks(data_folder):
    time_start = time.time()
    logging.info('Start')
    if not os.path.exists(data_folder):
        logging.warning('Folder ' + data_folder + 'does not exist: creating...')
        os.makedirs(data_folder)
    else:
        logging.info('Folder \'' + data_folder + '\' exists: not creating.')
        logging.info('Folder \'' + data_folder + '\' contains the following files:')
        ic = 0
        for i in os.listdir(data_folder):
            ic += 1
            logging.info('File ' + str(ic) + ': \'' + str(i) + '\'')
    time_end = time.time()
    logging.info('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')

# Variables
THIS_SCRIPT_PATH = os.environ['PEGASO_TRAIN_DIR']
execution_timestamp = datetime.datetime.now()
model_folder = 'models'
model_file = model_folder + '/rf_' + str(execution_timestamp).replace(':', '-').replace('.', '').replace(' ', '_')

initial_checks(model_folder)

df = fetch_database_data()

features, df_baseline = build_features(df)

feature_list, features, labels = convert_to_arrays(features)
#logging.info(str(feature_list))
#logging.info(str(features[0,:]))

train_features, test_features, train_labels, test_labels, train_indx, test_indx = train_test_split(features, labels, np.arange(features.shape[0]), test_size=0.25, random_state=42)

n_esti = 1000
n_rdst = 42
n_train = train_features.shape[0]
n_test = test_features.shape[0]

logging.info(' * Size of train set: ' + str(n_train))
logging.info(' * Size of test set: ' + str(n_test))

time_start = time.time()
logging.info('Computing baseline mean absolute error (MAE) and mean absolute percentage error (MAPE)...')
it = 0
baseline_mae = 0.0
baseline_mape = 0.0
for i in test_indx:
    baseline_mae = baseline_mae + abs( df_baseline.iloc[i, 0] - labels[i] )/n_test
    baseline_mape = baseline_mape + abs( df_baseline.iloc[i, 0]/labels[i] - 1.0 ) * 100.0 / n_test
    it += 1
logging.info('Computing baseline mean absolute error (MAE) and mean absolute percentage error (MAPE). Done.')
logging.info('  * baseline Mean Absolute Error (MAE): ' + str(round(baseline_mae, 2)) + ' Euros.')
logging.info('  * baseline Mean Absolute Percentage Error (MAPE): ' + str(round(baseline_mape, 2)) + ' %.')
logging.info('  * baseline Accuracy: ' + str(round( 100 - baseline_mape, 2)) + ' %.')

logging.info('Instantiate random forest with ' + str(n_esti) + ' decission trees...')
rf = RandomForestRegressor(n_estimators=n_esti, random_state=n_rdst)# Train the model on training data
logging.info('Instantiate random forest with ' + str(n_esti) + ' decission trees. Done')

logging.info('Training begins...')
time_start = time.time()
rf.fit(train_features, train_labels)
time_end = time.time()
logging.info('Training ends. Elapsed time: ' + str(time_end - time_start) + ' seconds.')

logging.info('Predicting test data...')
predictions = rf.predict(test_features)
logging.info('Predicting test data. Done.')

logging.info('Computing MAE and MAPE on test set...')
mae = round(np.mean(abs(predictions - test_labels)), 2)
mape = round(np.mean(abs((predictions - test_labels)/test_labels * 100.0)), 2)
logging.info('Computing MAE and MAPE on test set. Done.')

logging.info(' * Mean Absolute Error (MAE): ' + str(round(mae, 2)) + ' Euros.')
logging.info(' * Mean Absolute Percentage Error (MAPE): ' + str(round(mape, 2)) + ' %.')
logging.info(' * Accuracy: ' + str(round(100 - mape, 2)) + ' %.')

logging.info('Export the model...')
joblib.dump(rf, model_file + ".joblib", compress=0)
logging.info('Export the model. Done.')