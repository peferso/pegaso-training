import pandas as pd
import os
import base64
from io import BytesIO
import requests
import urllib
import pymysql
import numpy as np
import datetime

def predict(brd, kms, pwr, dor, yyr, model):
    API_ENDPOINT = os.environ['RF_API_ENDPOINT']
    response_API = requests.get('http://' +
                                API_ENDPOINT +
                                '/' + str(model) + '/' +
                                str(brd) + '/' +
                                str(kms) + '/' +
                                str(pwr) + '/' +
                                str(dor) + '/' +
                                str(yyr)
                                )
    price = round(float(response_API.text), 2)
    return price

def send_query(query):
    connection = pymysql.connect(host=os.environ['DBHOST'],
                                 user=os.environ['DBUSER'],
                                 passwd=os.environ['DBPASS'],
                                 db="pegaso_db",
                                 charset='utf8')
    con_cursor = connection.cursor()
    con_cursor.execute(query)
    output = con_cursor.fetchall()
    return output

def send_query_and_commit(query):
    connection = pymysql.connect(host=os.environ['DBHOST'],
                                 user=os.environ['DBUSER'],
                                 passwd=os.environ['DBPASS'],
                                 db="pegaso_db",
                                 charset='utf8')
    con_cursor = connection.cursor()
    con_cursor.execute(query)
    connection.commit()
    return

def feed_predictions_table():
    output = send_query("select price_c, kilometers, power, year, batch_ts, doors, id, brand " +
                            "from                        " +
                            "    raw_data                " +
                            "where                       " +
                            "    power is not null       " +
                            "and                         " +
                            "    kilometers is not null  " +
                            "and                         " +
                            "    year is not null        " +
                            "and                         " +
                            "    doors is not null       " +
                            "and                         " +
                            "    brand in (              " +
                            "               'AUDI',      " +
                            "               'BMW',       " +
                            "               'CITROEN',   " +
                            "               'FIAT',      " +
                            "               'FORD',      " +
                            "               'HYUNDAI',   " +
                            "               'JAGUAR',    " +
                            "               'KIA',       " +
                            "               'LANDROVER', " +
                            "               'MAZDA',     " +
                            "               'MERCEDES',  " +
                            "               'MINI',      " +
                            "               'NISSAN',    " +
                            "               'OPEL',      " +
                            "               'PEUGEOT',   " +
                            "               'PORSCHE',   " +
                            "               'RENAULT',   " +
                            "               'SEAT',      " +
                            "               'SKODA',     " +
                            "               'SMART',     " +
                            "               'TOYOTA',    " +
                            "               'VOLKSWAGEN'," +
                            "               'VOLVO'      " +
                             "            );")
    k = 1
    for i in output:
        prc = i[0]
        kms = i[1]
        pwr = i[2]
        yyr = ( int(str(i[4]).split('-')[0]) - int(i[3]) )
        dor = i[5]
        id = i[6]
        brd = i[7]
        pred_prc = int(predict(brd, kms, pwr, dor, yyr, 'predict_rf'))
        prc_new = int(predict(brd, 1, pwr, dor, 0, 'predict_rf'))
        query =          "insert ignore into predicted_prices_random_forest values " + \
                         "(" + \
                         "'" + str(id) + "'," + \
                         "'" + str(prc) + "'," + \
                         "'" + str(pred_prc) + "'," + \
                         "'" + str(prc_new) + "'," + \
                         "'" + str(datetime.datetime.now().date()) + "'" + \
                         ") ; "
        print(query)
        send_query_and_commit(query)
        print('Executed query', k, 'of', len(output))
        k += 1
    return

feed_predictions_table()