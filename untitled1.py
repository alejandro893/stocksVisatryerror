# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:29:38 2024

@author: Alejandro
"""
import numpy as np
import pandas as pd


# Crear un DataFrame de ejemplo grande con errores
data = {
    'A': [1, 2, np.nan, 4, 5, None, 7, 8, 9, 10, '', 12, 13, 14, 15],
    'B': ['foo', None, 'baz', 'bar', '', 'foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo', 'baz', 'bar', 'foo'],
    'C': [10, 20, 30, 40, 50, 60, 70, np.nan, 90, 100, 110, 120, None, 140, ''],
    'D': [None, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
}

# Duplicar algunas filas para crear duplicados
df = pd.DataFrame(data)
df = pd.concat([df, df.iloc[:3]], ignore_index=True)

# Eliminar filas duplicadas
df = df.drop_duplicates()

# Reemplazar valores vacíos con NaN para uniformidad
df.replace('', np.nan, inplace=True)

# Detectar y eliminar filas con valores NaN
df = df.dropna()

# Convertir tipos de datos de columnas si es necesario
df['A'] = pd.to_numeric(df['A'], errors='coerce')
df['C'] = pd.to_numeric(df['C'], errors='coerce')

# Volver a eliminar filas con NaN si las conversiones introdujeron nuevos NaN
df = df.dropna()

print(df.describe(include='all'))