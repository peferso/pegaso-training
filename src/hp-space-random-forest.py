import pandas as pd
import matplotlib.pyplot as plt

data = 'models/rf_2022-04-11_12-38-30401528-grid_cross_val_data'

df = pd.read_csv(data + '.csv', sep=',', header=0)

print(df.head())

l_n_estimators, l_max_features, l_mape = ([] for i in range(3))

for index, row in df.iterrows():
    for i in range(2, 17):
        l_n_estimators.append(row['n_estimators'])
        l_max_features.append(row['max_features'])
        l_mape.append(row[i])

df_all = pd.DataFrame(data={
    'n_estimators': l_n_estimators,
    'max_features': l_max_features,
    'mape': l_mape
})

print(df_all.head())

plt.figure(figsize=(10, 10))
plt.tight_layout()
plt.scatter(df_all['n_estimators'], df_all['mape'])
plt.savefig(data + '.png')
