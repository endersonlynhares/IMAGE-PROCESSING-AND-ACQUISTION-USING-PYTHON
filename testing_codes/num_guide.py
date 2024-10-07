import numpy as np
import csv

with open('nyc_taxis.csv', 'r') as file:
    taxi_csv = csv.reader(file)

    taxi = np.array(list(taxi_csv))

taxis_columns = taxi[0]
# print(taxis_columns)
tarifas_auxiliar = taxi[1:,9].astype(np.float32)
tarifas = np.sum(tarifas_auxiliar)
print(f'R$ {tarifas:,.2f}')
# for i in taxi_csv:
#     print(i)

# data_ndarray = np.array([5, 10, 15, 20])
# print(type(data_ndarray))
