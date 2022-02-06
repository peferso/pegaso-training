import os
import pymysql
import datetime
import pandas as pd

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

    return df

df = fetch_database_data()

build_features(df)
