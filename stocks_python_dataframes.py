# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:43:50 2024

@author: Alejandro
"""

import os
import csv
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


data_base_cvs = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror\MVR.csv"
data_base_excel = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror\MVR_excel.xlsx"
data_base_db = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror\sql-murder-mystery.db"
data_base_txt = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror\die_ISO-8859-1.txt"
data_base_txt2 = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror\MVR_txt.txt"
data_base_folder = r"C:\Users\Alejandro\Desktop\Jobs2024\Atradius\R\new\stocksVisatryerror"

Data_frame_csv = pd.read_csv(data_base_cvs, index_col="Date")
Data_frame_excel = pd.read_excel(data_base_excel)

conn = sqlite3.connect(data_base_db)
query = "SELECT * FROM person"
Data_frame_sql = pd.read_sql_query(query, conn)

Data_frame_txt = pd.read_csv(data_base_txt2, header = None)



# List all files in the directory
files = os.listdir(data_base_folder)
# Filter the list to include only CSV files
csv_files = [file for file in files if file.endswith('.csv')]

# Create an empty list to store DataFrames
dataframes = []

# Loop through the CSV files and read each one into a DataFrame
for file in csv_files:
    file_path = os.path.join(data_base_folder, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Example: Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

with open(data_base_txt, 'r') as file:
    text_data = file.read()
    
# Open and read a CSV file using csv.reader
with open(data_base_cvs, mode='r') as file:
    csv_reader = csv.reader(file)
    
    
    
    
###### vamos a operar ahora el dataframe de csv ###########
Data_frame_csv = Data_frame_csv[['Open_M', 'High_M', 'Low_M', 'Close_M', 'Adj Close_M', 'Volume_M']]
cols_pos = Data_frame_csv.iloc[:, [0, 2]]
# Check for empty rows
Data_frame_csv.describe(include='all')

null_rows = Data_frame_csv.isnull().sum()
#print(null_rows)
null_rows_all = Data_frame_csv.isnull()
nan_rows = Data_frame_csv.isna().sum()
#print(nan_rows)
nan_rows_all = Data_frame_csv.isna()
empty_rows =  (Data_frame_csv == '').sum()
#print(empty_rows)
empty_rows_all =  (Data_frame_csv == '')


############ Eliminate the rows that are not neccesary anymore #########
Data_frame_csv.replace('', pd.NA, inplace=True)
Data_frame_csv.replace('None', pd.NA, inplace=True)
Data_frame_csv_clear = Data_frame_csv.dropna(how='any')
cols_pos = Data_frame_csv.iloc[:, [0, 2]]
mean_data_frame_csv = Data_frame_csv['Open_M'].mean()
std_data_frame_csv = Data_frame_csv['Open_M'].std()
Data_frame_csv.index = pd.to_datetime(Data_frame_csv.index)

def remove_outliers(df):
    # Calcular el IQR para cada columna numérica
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Filtrar las filas que están dentro de los límites
    df_clean = df[~((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR))).any(axis=1)]
    return df_clean

df_clean = remove_outliers(df = Data_frame_csv[['Open_M']])
df_clean = remove_outliers(df = Data_frame_csv)

def remove_outliers(df, column, window_size=30, z_threshold=3):
    # Crear una copia del DataFrame para no modificar el original
    df_filtered = df.copy()

    # Calcular el rolling mean y rolling std (desviación estándar) con una ventana de 30 días
    rolling_mean = df_filtered[column].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df_filtered[column].rolling(window=window_size, min_periods=1).std()

    # Calcular el z-score
    z_scores = (df_filtered[column] - rolling_mean) / rolling_std

    # Eliminar los outliers basados en el umbral del z-score
    df_filtered = df_filtered[np.abs(z_scores) <= z_threshold]

    return df_filtered

# Aplicar la función al DataFrame
for column in df_clean.columns:
    df_clean = remove_outliers(df_clean, column)

# Definir la fecha límite
date_limit = '2016-01-05'

# Convertir la fecha límite a datetime (por si acaso no es datetime)
date_limit = pd.to_datetime(date_limit)

# Filtrar el DataFrame para eliminar filas anteriores a la fecha límite
filtered_df = df_clean[df_clean.index >= date_limit]


def plot_and_save_histograms_scatterplots(df, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for column in df.columns:
        # Histograma
        plt.figure()
        sns.histplot(df[column], kde=True, color="red")
        plt.title(f'Histogram of {column}')
        plt.savefig(f'{output_dir}/histogram_{column}.png')
        plt.show()
        
        # Scatter plot con las fechas
        plt.figure()
        plt.scatter(df.index, df[column], color="white", edgecolor="red")
        plt.title(f'Scatter plot of {column} vs Date')
        plt.xlabel('Date')
        plt.ylabel(column)
        
        # Ajustar las etiquetas del eje x
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limitar a 10 etiquetas
        plt.xticks(rotation=45)
        
        plt.savefig(f'{output_dir}/scatterplot_{column}.png')
        plt.show()

# Llamada a la función
plot_and_save_histograms_scatterplots(df_clean)
