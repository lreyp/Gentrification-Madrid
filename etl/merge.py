import os
import pandas as pd
from functools import reduce
import numpy as np


def merge_indicadores(folder):
    # Paths and files
    file = os.path.join(os.getcwd(), 'data\poblacion', 'poblacion_barrio_15_20.csv')
    print('loading:', file, '...')
    df_poblacion = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\\renta', 'renta_barrio_15_20.csv')
    print('loading:', file, '...')
    df_renta = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\padron', 'padron_barrio_15_20.csv')
    print('loading:', file, '...')
    df_padron = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\\vivienda', 'vivienda_barrio_15_20.csv')
    print('loading:', file, '...')
    df_vivienda = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\locales', 'locales_barrio_15_20.csv')
    print('loading:', file, '...')
    df_locales = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\policia', 'policia_distrito_15_20.csv')
    print('loading:', file, '...')
    df_policia = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    df_vivienda = df_vivienda.sort_values(by=['distrito', 'barrio', 'year'])

    # List of DataFrames to be merged
    dfs_to_merge = [df_vivienda, df_poblacion, df_renta, df_padron, df_locales, df_policia]

    # Define the common merge keys
    merge_keys_barrio = ['barrio', 'year']
    merge_keys_distrito = ['distrito', 'year']

    # Merge DataFrames using reduce
    df_indicadores = reduce(lambda left, right: pd.merge(left, right, on=merge_keys_barrio, how='outer'),
                            dfs_to_merge[:-1])
    df_indicadores = pd.merge(df_indicadores, dfs_to_merge[-1], on=merge_keys_distrito, how='outer')

    df_indicadores['renta_media'] = df_indicadores['renta_media'].replace('.', np.NaN)
    df_indicadores['renta_media'] = df_indicadores.renta_media.astype(str).astype(float)

    df_indicadores.rename(columns={'Administración pública': 'admin_publica', 'Comercio al por mayor': 'comercio_mayor',
                                   'Comercio al por menor': 'comercio_menor', 'Construcción': 'construccion',
                                   'Educación': 'educacion', 'Finanzas y seguros': 'finanzas', 'Hoteles': 'hoteles',
                                   'Inmobiliarias': 'inmobiliarias', 'Ocio y cultura': 'ocio',
                                   'Restaurantes': 'restaurantes', 'Sanidad': 'sanidad'}, inplace=True)

    df_indicadores['barrio'] = df_indicadores['barrio'].replace(
        ['LOS ANGELES', 'CONCEPCION', 'CASCO H.VALLECAS', 'CASCO H.BARAJAS', 'ARGUELLES',
         'PEÑA GRANDE', 'CASCO H.VICALVARO', 'LAS AGUILAS', 'EL PILAR', 'PALOS DE MOGUER'],
        ['ANGELES', 'LA CONCEPCION', 'CASCO HISTORICO DE VALLECAS', 'CASCO HISTORICO DE BARAJAS', 'ARGÜELLES',
         'PEÑAGRANDE', 'CASCO HISTORICO DE VICALVARO', 'AGUILAS', 'PILAR', 'PALOS DE LA FRONTERA'])

    df_indicadores.to_csv(os.path.join(folder, 'indicadores_15_20.csv'), encoding='latin1', index=False)

