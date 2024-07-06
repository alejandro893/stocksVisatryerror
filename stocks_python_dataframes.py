# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:43:50 2024

@author: Alejandro
"""

import os
import csv
import pandas as pd
import sqlite3

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


