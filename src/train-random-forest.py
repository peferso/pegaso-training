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
from multiprocessing import Pool
import pickle
from random import randint


def fetch_database_data():
    time_start = time.time()
    print('Start')
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
                                        brand IN (SELECT nb.brand FROM brands_count nb WHERE nb.num_cars>1000)
                                      ;""", connection)
    dfrd = pd.DataFrame(sql_query,
                        columns=['brand',  # One-hot
                                 'model',  # One-hot
                                 'price_c',
                                 'kilometers',
                                 'power',
                                 'doors',
                                 'professional_vendor',  # One-hot
                                 'automatic_gearbox',  # One-hot
                                 'year',
                                 'batch_ts'
                                 ])
    time_end = time.time()
    print('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')
    return dfrd


def build_features(df):
    time_start = time.time()
    print('Start')

    print('Compute new variable \'years\'...')
    l_years = []
    for index, row in df.iterrows():
        years = row['batch_ts'].year - int(row['year'])
        l_years.append(years)

    df['years'] = l_years
    print('Compute new variable \'years\'. Done.')

    print('Dropping useless columns...')
    df = df.drop('batch_ts', axis=1)
    df = df.drop('year', axis=1)
    df = df.drop('professional_vendor', axis=1)
    df = df.drop('automatic_gearbox', axis=1)
    df = df.drop('model', axis=1)
    print('Dropping useless columns. Done.')

    print('Dropping rows with \'nans\'...')
    df = df.dropna()
    print('Dropping rows with \'nans\'. Done.')

    l_avprice = []

    print('Getting average price of each car based on brand...')
    t1 = time.time()
    irow = 0
    for index, row in df.iterrows():
        t2 = time.time()
        brand = row['brand']
        avprice = 1  # np.mean(df[df['brand'] == brand]['price_c'])
        l_avprice.append(avprice)
        if t2 - t1 > 10:
            print('  ' + str(index) + ' rows processed. ' +
                         str(round(t2 - time_start, 2)) + ' seconds elapsed - ' +
                         str(round((df.shape[0] - irow) / (index - irow) * (t2 - t1), 2)) +
                         ' seconds to finish...')
            t1 = time.time()
    print('Getting average price of each car based on brand. Done.')

    df_baseline = pd.DataFrame(data={'av_price': l_avprice, 'price_c': df['price_c']})

    # Shuffle rows and keep apart a set to finally evaluate accuracy
    df.sample(frac=1).reset_index(drop=True)

    # One-hot encoding TO TEST https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    features = pd.get_dummies(df)
    time_end = time.time()
    print('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')
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
    print('Start')
    if not os.path.exists(data_folder):
        logging.warning('Folder ' + data_folder + 'does not exist: creating...')
        os.makedirs(data_folder)
    else:
        print('Folder \'' + data_folder + '\' exists: not creating.')
        print('Folder \'' + data_folder + '\' contains the following files:')
        ic = 0
        for i in os.listdir(data_folder):
            ic += 1
            print('File ' + str(ic) + ': \'' + str(i) + '\'')
    time_end = time.time()
    print('End. Elapsed time: ' + str(time_end - time_start) + ' seconds.')


def write_report(rep_est, rep_mft, prec_list, mape_list, folds_list, rep_tms, report_file):
    dfreport = pd.DataFrame(list(zip(rep_est, rep_mft, prec_list, mape_list, folds_list, rep_tms)),
                            columns=['estimators', 'max_features', 'average_accuracy', 'average_mape', 'folds',
                                     'train_time'])
    dfreport.to_csv(report_file, index=None, header=True)


class Model:

    def __init__(self, features, labels, id):
        logging.basicConfig(
            format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
            level=logging.INFO)
        # Hyperparameters
        self.n_estimators = 5
        self.random_state = None
        self.max_features = 0.75
        self.criterion = 'squared_error'
        # Input data
        self.features = features
        self.labels = labels
        # train set data
        self.train_features = None
        self.train_labels = None
        self.train_indx = None
        # test set data
        self.test_features = None
        self.test_labels = None
        self.test_indx = None

        self.id = id

    def split_train_and_test_sets(self):
        random_state = randint(0, 42)
        self.train_features, self.test_features, self.train_labels, self.test_labels, self.train_indx, self.test_indx = \
            train_test_split(self.features,
                             self.labels,
                             np.arange(self.features.shape[0]),
                             test_size=0.20,
                             random_state=random_state)

    def set_hyperparameters(self, n_estimators, random_state, max_features, criterion):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.criterion = criterion

    def train_a_random_forest(self):
        self.split_train_and_test_sets()
        rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                   criterion=self.criterion,
                                   max_features=self.max_features,
                                   random_state=self.random_state)
        rf.fit(self.train_features, self.train_labels)
        predictions = rf.predict(self.test_features)
        rf = None
        mape = round(np.mean(abs((predictions - self.test_labels) / self.test_labels * 100.0)), 2)
        return mape

def cross_validation_training(model):
    return model.train_a_random_forest()

# Variables
THIS_SCRIPT_PATH = os.environ['PEGASO_TRAIN_DIR']
execution_timestamp = datetime.datetime.now()
model_folder = 'models'
model_file = model_folder + '/rf_' + str(execution_timestamp).replace(':', '-').replace('.', '').replace(' ', '_')
report_file = model_file + '.csv'
initial_checks(model_folder)

df = fetch_database_data()

features, df_baseline = build_features(df)

feature_list, features, labels = convert_to_arrays(features)

f = open(model_file + '-feature_list.txt', 'w')
s1 = '\n'.join(feature_list)
f.write(s1)
f.close()

f = open(model_file + '-feature_list.list', 'wb')
pickle.dump(feature_list, f)
f.close()

# Split the data into features + evaluation features. The latter will not be used in training nor hyperparameters tuning
features, eval_features, labels, eval_labels, indx, eval_indx = train_test_split(features, labels,
                                                                                                   np.arange(
                                                                                                       features.shape[
                                                                                                           0]),
                                                                                                   test_size=0.30,
                                                                                                   random_state=42)

print(' * Size of features set: ' + str(features.shape[0]))
print(' * Size of evaluation set: ' + str(eval_features.shape[0]))

time_start = time.time()
print('Computing baseline mean absolute error (MAE) and mean absolute percentage error (MAPE) on evaluation set...')
it = 0
baseline_mae = 0.0
baseline_mape = 0.0
for i in eval_indx:
    baseline_mae += abs(df_baseline.iloc[i, 0] - df_baseline.iloc[i, 1]) / df_baseline.shape[0]
    baseline_mape += abs(df_baseline.iloc[i, 0] / df_baseline.iloc[i, 1] - 1.0) * 100.0 / df_baseline.shape[0]
    it += 1
print('Computing baseline mean absolute error (MAE) and mean absolute percentage error (MAPE) on evaluation set. Done.')
print('  * baseline Mean Absolute Error (MAE): ' + str(round(baseline_mae, 2)) + ' Euros.')
print('  * baseline Mean Absolute Percentage Error (MAPE): ' + str(round(baseline_mape, 2)) + ' %.')
print('  * baseline Accuracy: ' + str(round(100 - baseline_mape, 2)) + ' %.')

n_est_list = range(1, 501, 1)
max_features_list = [x / 100.0 for x in range(10, 105, 5)]
criterion = 'squared_error'
random_state = None
mape_min = 100.0
times, rep_est, rep_mft, prec_list, mape_list, folds_list, rep_tms = ([] for i in range(7))
interations_remaining = len(n_est_list) * len(max_features_list)

folds = 16
cpu_cores = 4
models = []
for i in range(1, folds + 1):
    model = Model(features, labels, i)
    models.append(model)

print('Computing grid of parameters:')
f = open(model_file + '-grid_cross_val_data.csv', 'w')
print('n_estimators', 'max_features', *['mape_fold' + str(i) for i in range(1, folds + 1)]
                                    , 'average_mape', 'stderr_mape', sep=',', end='\n', file=f)
f.close()
for n_estimators in n_est_list:
    for max_features in max_features_list:

        #print('\tn_estimators:' + str(n_estimators))
        #print('\tmax_features:' + str(max_features))
        #print('\tCross validating over ' + str(folds) + '-folds...')
        #print('\tFold computations parallelized over ' + str(cpu_cores) + ' cores...')

        #print('\t\tMultiprocessing begins...')
        ti = time.time()
        for model in models:
            model.set_hyperparameters(n_estimators, random_state, max_features, criterion)
        p = Pool(processes=cpu_cores)
        result = p.map(cross_validation_training, models)
        p.close()
        p.join()
        tf = time.time()
        #print('\t\tMultiprocessing ends.')
        #print('\t\t', result)
        #print('\tCross validation finished.')
        #print('\tElapsed:' + str(tf - ti))

        mape = np.average(np.array(result))
        mape_var = np.std(np.array(result))

        f = open(model_file + '-grid_cross_val_data.csv', 'a')
        print(n_estimators, max_features, *result, mape, mape_var, sep=',', end='\n', file=f)
        f.close()

        if mape < mape_min:
            mape_min = mape
            n_estimators_min = n_estimators
            max_features_min = max_features

        rep_est.append(n_estimators)
        rep_mft.append(max_features)
        mape_list.append(round(mape, 4))
        prec_list.append(round(100.0 - mape, 4))
        folds_list.append(folds)
        rep_tms.append(tf - ti)

        print('\tacc.', round(100.0 - mape, 2), n_estimators, max_features,
              ' - mape ', mape, mape_var,
              '  --- max acc.', round(100.0 - mape_min, 2), n_estimators_min, max_features_min)
        #print('\tMinimum average mape accross folds found: ' + str(mape_min))
        #print('\t  max. accuracy: ' + str(100.0 - mape_min))
        #print('\t  n_estimators: ' + str(n_estimators_min))
        #print('\t  max_features: ' + str(max_features_min))
#
        write_report(rep_est, rep_mft, mape_list, prec_list, folds_list, rep_tms, report_file)

print('\nTraining the best model.\n')
rf = RandomForestRegressor(n_estimators=n_estimators_min, criterion=criterion, max_features=max_features_min, random_state=random_state)
print('Training begins...')
time_start = time.time()
rf.fit(self.train_features, self.train_labels)
time_end = time.time()
print('Training ends. Elapsed time: ' + str(time_end - time_start) + ' seconds.')

print('Predicting evaluation data...')
predictions = rf.predict(eval_features)
print('Predicting test data. Done.')
print('Computing MAE and MAPE on test set...')
mae = round(np.mean(abs(predictions - eval_labels)), 2)
mape = round(np.mean(abs((predictions - eval_labels) / eval_labels * 100.0)), 2)
print('Computing MAE and MAPE on test set. Done.')
print(' * Mean Absolute Error (MAE): ' + str(round(mae, 2)) + ' Euros.')
print(' * Mean Absolute Percentage Error (MAPE): ' + str(round(mape, 2)) + ' %.')
print(' * Accuracy: ' + str(round(100 - mape, 2)) + ' %.')

print('Export the model...')
joblib.dump(rf, model_file + ".joblib", compress=0)
print('Export the model. Done.')
