import os
import pandas as pd
import glob
import numpy as np


def load_data(folder):

    df_list = []

    if folder == 'data\locales':
        # Load files locales
        directory = os.path.join(os.getcwd(), folder, '*.csv')
        for file in glob.glob(directory):
            print('loading:', file, '...')
            df = pd.read_csv(file, sep=';', encoding='unicode_escape', low_memory=False, header=0)
            df.columns.values[0] = 'id_local'
            df['name'] = os.path.basename(os.path.normpath(file))
            df.drop(df.columns[14:40], axis=1, inplace=True)
            df.drop(df.columns[10:12], axis=1, inplace=True)
            df = df.drop(['cod_barrio_local', 'id_seccion_censal_local', 'desc_seccion_censal_local',
                          'desc_seccion', 'desc_division', 'id_epigrafe', 'desc_epigrafe'],
                         axis=1)
            df_list.append(df)

        df_locales = pd.concat(df_list, ignore_index=True)
        df_locales.to_csv(os.path.join(folder, 'locales_15_20.csv'), encoding='latin1', index=False)

    if folder == 'data\padron':
        # Load files padron
        directory = os.path.join(os.getcwd(), folder, '*.csv')
        for file in glob.glob(directory):
            print('loading:', file, '...')
            df = pd.read_csv(file, sep=';', encoding='unicode_escape', low_memory=False, header=0)
            df.columns.values[0] = 'COD_DISTRITO'
            df.columns.values[8] = 'EspanolesHombres'
            df.columns.values[9] = 'EspanolesMujeres'
            df.columns.values[10] = 'ExtranjerosHombres'
            df.columns.values[11] = 'ExtranjerosMujeres'
            df['name'] = os.path.basename(os.path.normpath(file))
            df_list.append(df)

        df_padron = pd.concat(df_list, ignore_index=True)
        df_padron.drop(['COD_DISTRITO', 'DESC_DISTRITO', 'COD_DIST_BARRIO', 'COD_BARRIO',
                        'COD_DIST_SECCION', 'COD_SECCION'], axis=1, inplace=True)
        df_padron.to_csv(os.path.join(folder, 'padron_15_20.csv'), encoding='latin1', index=False)

    if folder == 'data\policia':
        # Load files policia
        directory = os.path.join(os.getcwd(), folder, '*.xlsx')
        for file in glob.glob(directory):
            print('loading:', file, '...')
            df = pd.read_excel(file, sheet_name=0, header=2)
            df['name'] = os.path.basename(os.path.normpath(file))
            df_list.append(df)

        df_policia = pd.concat(df_list, ignore_index=True)
        df_policia.to_csv(os.path.join(folder, 'policia_15_20.csv'), encoding='latin1', index=False)

    if folder == 'data\poblacion':
        # Load files poblacion
        directory = os.path.join(os.getcwd(), folder, '*.xlsx')
        for file in glob.glob(directory):
            print('loading:', file, '...')
            df = pd.read_excel(file, sheet_name=0, header=6)
            df = df.dropna(how='all')
            df = df[:-1]
            df = df.iloc[1:]
            df = df.drop(['Unnamed: 0', 'Porcentaje de población menor de 18 años', 'Porcentaje de población de 65 y más años',
                          'Porcentaje de hogares unipersonales', 'Porcentaje de población menor de 18 años.1',
                          'Porcentaje de población de 65 y más años.1', 'Porcentaje de hogares unipersonales.1'], axis=1)
            df_1 = df[['Unnamed: 1', 'Edad media de la población', 'Tamaño medio del hogar', 'Población']].copy()
            df_2 = df[['Unnamed: 1', 'Edad media de la población.1', 'Tamaño medio del hogar.1', 'Población.1']].copy()
            df_2.rename(columns={'Edad media de la población.1': 'Edad media de la población',
                                 'Tamaño medio del hogar.1': 'Tamaño medio del hogar',
                                 'Población.1': 'Población'}, inplace=True)
            df = pd.DataFrame()
            dfs_to_concat = [df, df_1, df_2]
            df = pd.concat(dfs_to_concat, ignore_index=True)
            df = df.rename(columns={'Unnamed: 1': 'barrio', 'Edad media de la población': 'edad_media',
                                    'Tamaño medio del hogar': 'tamano_medio', 'Población': 'poblacion'})
            df['barrio'] = df['barrio'].str.split('.').str[-1]
            df['barrio'] = df['barrio'].str.strip()
            df['barrio'] = df['barrio'].str.upper()
            df['name'] = os.path.basename(os.path.normpath(file))
            df_list.append(df)

        df_poblacion = pd.concat(df_list, ignore_index=True)
        df_poblacion.to_csv(os.path.join(folder, 'poblacion_15_20.csv'), encoding='latin1', index=False)

    if folder == 'data\\renta':
        # Load files renta
        directory = os.path.join(os.getcwd(), folder, '*.xlsx')
        for file in glob.glob(directory):
            print('loading:', file, '...')
            df = pd.read_excel(file, sheet_name=0, header=6)
            df = df.dropna(how='all')
            df = df[:-1]
            df = df.iloc[1:]
            df = df.drop(['Renta media por persona ', 'Renta media por persona .1'], axis=1)
            df_1 = df[['Unnamed: 0', 'Renta media por hogar']].copy()
            df_2 = df[['Unnamed: 0', 'Renta media por hogar.1']].copy()
            df_2.rename(columns={'Renta media por hogar.1': 'Renta media por hogar'}, inplace=True)
            df = pd.DataFrame()
            dfs_to_concat = [df, df_1, df_2]
            df = pd.concat(dfs_to_concat, ignore_index=True)
            df = df.rename(columns={'Unnamed: 0': 'barrio', 'Renta media por hogar': 'renta_media'})
            df['barrio'] = df['barrio'].str.split('.').str[-1]
            df['barrio'] = df['barrio'].str.strip()
            df['barrio'] = df['barrio'].str.upper()
            df['name'] = os.path.basename(os.path.normpath(file))
            df_list.append(df)

        df_renta = pd.concat(df_list, ignore_index=True)
        df_renta.to_csv(os.path.join(folder, 'renta_15_20.csv'), encoding='latin1', index=False)

    if folder == 'data\\vivienda':
        # Load file alquiler
        file = os.path.join(os.getcwd(), folder, 'alquiler_14-23.xlsx')
        print('loading:', file, '...')
        df = pd.read_excel(file, sheet_name=0, header=4)
        df = df[:-3]
        df = df.iloc[2:23]
        df = df.rename(columns={'Unnamed: 0': 'distrito'})
        df = df.drop(['2014', '2021', '2022', '2023'], axis=1)
        df = pd.melt(df, id_vars=['distrito'], value_name="alquiler")
        df = df.rename(columns={'variable': 'year'})
        df['distrito'] = df['distrito'].str.split('.').str[-1]
        df['distrito'] = df['distrito'].str.strip()
        df['distrito'] = df['distrito'].str.upper()
        tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
        df['distrito'] = df['distrito'].replace(tildes, regex=True)
        df['alquiler'] = df['alquiler'].str.replace(',', '.')
        df['alquiler'] = df.alquiler.astype(float)

        df_alquiler = df
        df_alquiler.to_csv(os.path.join(folder, 'alquiler_distrito_15_20.csv'), encoding='latin1', index=False)

        # Load files venta, clean and merge
        def clean_data(df):
            df = df[:-4].iloc[1:]
            df = df.rename(columns={'Unnamed: 0': 'distrito', 'Unnamed: 1': 'barrio'})
            df['distrito'] = df['distrito'].str.split('.').str[-1].str.strip().str.upper()
            df['barrio'] = df['barrio'].str.split('.').str[-1].str.strip().str.upper()
            list_distritos = ('Centro', 'Arganzuela', 'Retiro', 'Salamanca', 'Chamartín', 'Tetuán', 'Chamberí',
                              'Fuencarral-El Pardo', 'Moncloa-Aravaca', 'Latina', 'Carabanchel', 'Usera',
                              'Puente de Vallecas', 'Moratalaz', 'Ciudad Lineal', 'Hortaleza', 'Villaverde',
                              'Villa de Vallecas', 'Vicálvaro', 'San Blas-Canillejas', 'Barajas')
            list_distritos = [x.upper() for x in list_distritos]
            df = df[~df['barrio'].isin(list_distritos)]
            tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
            df['barrio'] = df['barrio'].replace(tildes, regex=True)
            df['distrito'] = df['distrito'].replace(tildes, regex=True)
            df['barrio'] = df['barrio'].replace(
                ['CASCO HISTORICO DE BARAJAS', 'CASCO HISTORICO DE VALLECAS', 'CASCO HISTORICO DE VICALVARO',
                 'JERONIMOS', 'PEÑAGRANDE', 'PILAR', 'SALVADOR'],
                ['CASCO H.BARAJAS', 'CASCO H.VALLECAS', 'CASCO H.VICALVARO',
                 'LOS JERONIMOS', 'PEÑA GRANDE', 'EL PILAR', 'EL SALVADOR'])
            return df

        file_path_1 = os.path.join(os.getcwd(), 'data\\vivienda', 'precio_14-19.xlsx')
        print('loading:', file_path_1, '...')
        df1 = clean_data(pd.read_excel(file_path_1, sheet_name=0, header=4))

        file_path_2 = os.path.join(os.getcwd(), 'data\\vivienda', 'precio_17-22.xlsx')
        print('loading:', file_path_2, '...')
        df2 = clean_data(pd.read_excel(file_path_2, sheet_name=0, header=4))

        # Merge to have 2020 data
        df1['2020'] = df2['2020']
        df = df1.drop(['2014'], axis=1)
        df = pd.melt(df, id_vars=['distrito', 'barrio'], value_name="venta")
        df = df.rename(columns={'variable': 'year'})

        df['venta'] = df['venta'].str.replace('.', '')
        df['venta'] = df['venta'].replace('-', np.NaN)
        df['venta'] = df['venta'].astype(float)

        df_venta = df
        df_venta.to_csv(os.path.join(folder, 'venta_barrio_15_20.csv'), encoding='latin1', index=False)
