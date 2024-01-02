import os
import pandas as pd
import numpy as np
from itertools import product
import plotly.express as px
import utm


def transform_locales(folder):
    # Load DataFrame
    file = os.path.join(os.getcwd(), folder, 'locales_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', low_memory=False, header=0)

    # Remove unwanted entries
    secciones = [
        'COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACI0N DE VEHICULOS DE MOTOR Y MOTOCICLETAS',
        'HOSTELERIA',
        '-1']

    for seccion in secciones:
        count = len(df[df['id_seccion'] == seccion])
        print(f'Error de clasificación ({seccion}): {count}')

    divisiones = [
        'PT',
        'COMERCIO AL POR MENOR, EXCEPTO DE VEHÍCULOS DE MOTOR Y MOTOCICLETAS',
        'SERVICIOS DE COMIDAS Y BEBIDAS',
        '-1']

    for division in divisiones:
        count = len(df[df['id_division'] == division])
        print(f'Error de clasificación ({division}): {count}')

    for seccion in secciones:
        df.drop(df[df['id_seccion'] == seccion].index, inplace=True)

    for division in divisiones:
        df.drop(df[df['id_division'] == division].index, inplace=True)

    # Remove rows with missing values in 'id_seccion' or 'id_division'
    df = df[df['id_seccion'].notna() & df['id_division'].notna()]

    # Convert 'id_division' to numeric
    df['id_division'] = df['id_division'].astype(str).astype(float).astype(int)

    # Mapping for id_seccion
    seccion_mapping = {'R': 'Ocio y cultura', 'F': 'Construcción', 'K': 'Finanzas y seguros',
                       'O': 'Administración pública', 'P': 'Educación', 'Q': 'Sanidad'}

    # Mapping for id_division
    division_mapping = {46: 'Comercio al por mayor', 47: 'Comercio al por menor',
                        55: 'Hoteles', 56: 'Restaurantes', 68: 'Inmobiliarias'}

    # Apply mappings to create 'category' column
    df['category'] = np.nan
    df['category'] = df['category'].combine_first(df['id_division'].map(division_mapping)).combine_first(
        df['id_seccion'].map(seccion_mapping))

    # Extract 'year' from 'name' column
    df['year'] = df['name'].str[-10:-6]

    # Drop unnecessary columns
    df = df.drop(['id_seccion', 'id_division', 'name'], axis=1)

    # Drop rows with missing values in 'category'
    df = df[df['category'].notna()]

    # Create df_local
    df_local = df

    # Drop unnecessary columns
    df_local = df_local.drop(['id_distrito_local', 'id_barrio_local', 'id_situacion_local'], axis=1)

    # Replace strings and types
    df_local['coordenada_x_local'] = df_local['coordenada_x_local'].str.replace(',', '.')
    df_local['coordenada_y_local'] = df_local['coordenada_y_local'].str.replace(',', '.')
    df_local['coordenada_x_local'] = df_local['coordenada_x_local'].astype(str).astype(float)
    df_local['coordenada_y_local'] = df_local['coordenada_y_local'].astype(str).astype(float)
    df_local['year'] = df_local['year'].astype(str).astype(float).astype(int)

    # Drop rows with errors in coords (utm conditions)
    df_local = df_local[df_local.coordenada_x_local != 0]
    df_local = df_local[df_local.coordenada_x_local > 100000]
    df_local = df_local[df_local.coordenada_x_local < 999999]
    df_local = df_local[df_local.coordenada_y_local != 0]
    df_local = df_local[df_local.coordenada_y_local > 0]
    df_local = df_local[df_local.coordenada_y_local < 10000000]

    # Striping and replacing
    df_local['desc_barrio_local'] = df_local['desc_barrio_local'].apply(lambda x: x.strip())
    df_local['desc_barrio_local'] = df_local['desc_barrio_local'].str.replace('Ã\x91', 'Ñ')
    df_local = df_local.drop_duplicates()

    # Transform coords from utm to lat/lon
    df_local[['coordenada_x_local', 'coordenada_y_local']] = (df_local.apply(
        lambda x: utm.to_latlon(x['coordenada_x_local'], x['coordenada_y_local'], 30, northern=True),
        axis=1, result_type='expand'))

    # Rename the columns
    df_local.rename(columns={"coordenada_x_local": "latitude", "coordenada_y_local": "longitude"}, inplace=True)

    # Save df_local
    df_local.to_csv(os.path.join(folder, 'locales_coord_15_20.csv'), encoding='latin1', index=False)

    # Create df_radar
    df_radar = df
    df_radar['frequency'] = df_radar.groupby((['desc_barrio_local', 'year', 'category']))['category'].transform('count')
    df_radar = df_radar.drop(['id_local', 'id_distrito_local', 'desc_distrito_local', 'id_barrio_local',
                              'coordenada_x_local', 'coordenada_y_local', 'id_situacion_local', 'desc_situacion_local'],
                             axis=1)
    df_radar['desc_barrio_local'] = df_radar['desc_barrio_local'].apply(lambda x: x.strip())
    df_radar['desc_barrio_local'] = df_radar['desc_barrio_local'].str.replace('Ã\x91', 'Ñ')
    df_radar = df_radar.drop_duplicates()
    df_radar.rename(columns={"desc_barrio_local": "barrio"}, inplace=True)
    df_radar = df_radar.sort_values(by=['barrio', 'year', 'category'])

    # Create df_radar_structure
    barrios = df_radar['barrio'].unique()
    categories = df_radar['category'].unique()
    years = df_radar['year'].unique()

    df_radar_structure = pd.DataFrame(list(product(barrios, categories, years)), columns=['barrio', 'category', 'year'])
    df_radar_structure = df_radar_structure.merge(df_radar, how='left')
    df_radar_structure = df_radar_structure.sort_values(by=['barrio', 'year', 'category'])
    df_radar_structure['frequency'] = df_radar_structure['frequency'].fillna(0)

    df_radar_structure['proportion'] = df_radar_structure['frequency'] / df_radar_structure.groupby(['barrio', 'year'])[
        'frequency'].transform('sum') * 100

    # Create df_radar_madrid
    df_radar_madrid = df_radar_structure.groupby(['category', 'year'], as_index=False, sort=False)[
        ['frequency', 'proportion']].mean()
    df_radar_madrid.insert(0, 'barrio', 'MADRID')
    df_radar_complete = pd.concat([df_radar_structure, df_radar_madrid], ignore_index=True)

    # Save to csv
    df_radar_complete.to_csv(os.path.join(folder, 'locales_radar_15_20.csv'), encoding='latin1', index=False)

    # Plot radar charts for each barrio and year
    barrios = df_radar_complete['barrio'].unique()
    # print('Existen', len(barrios) - 1, 'barrios en Madrid.')

    if not os.path.exists("images"):
        os.mkdir("images")

    if not os.path.exists("images/radar"):
        os.mkdir("images/radar")

    years = ('2015', '2020')

    for barrio in barrios:
        for year in years:
            # print(barrio, year)
            df_madrid = df_radar_madrid.loc[(df_radar_madrid['year'] == year)]
            df_barrio = pd.concat([
                df_radar_complete[(df_radar_complete['barrio'] == barrio) & (df_radar_complete['year'] == year)],
                df_madrid
            ], ignore_index=True)
            fig = px.line_polar(df_barrio, r='proportion', theta='category', color='barrio',
                                line_close=True, log_r=True)
            fig.update_traces(fill='toself')
            filename = f"images/radar/radar_{barrio}_{year}.png"
            fig.write_image(filename)
            # fig.show()

    df = df_radar_complete
    df = df.drop(['proportion'], axis=1)
    df = df.pivot_table('frequency', ['barrio', 'year'], 'category')
    df.reset_index(drop=False, inplace=True)
    df.reindex(['barrio', 'category', 'Administración pública', 'Comercio al por mayor', 'Comercio al por menor',
                'Construcción', 'Educación', 'Finanzas y seguros', 'Hoteles', 'Inmobiliarias', 'Ocio y cultura',
                'Restaurantes', 'Sanidad'], axis=1)
    df.drop(df[df['barrio'] == 'MADRID'].index, inplace=True)

    # Save to csv
    df.to_csv(os.path.join(folder, 'locales_barrio_15_20.csv'), encoding='latin1', index=False)

def transform_padron(folder):
    # Load DataFrame
    file = os.path.join(os.getcwd(), folder, 'padron_15_20.csv')
    print('loading:', file, '...')
    col_names = pd.read_csv(file, nrows=0).columns
    types_dict = {'DESC_BARRIO': str, 'COD_EDAD_INT': str, 'name': str}
    types_dict.update({col: float for col in col_names if col not in types_dict})
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', dtype=types_dict, header=0)
    df = df.iloc[1:]
    columns_to_replace = ['EspanolesHombres', 'EspanolesMujeres', 'ExtranjerosHombres', 'ExtranjerosMujeres']

    for col in columns_to_replace:
        df[col].fillna(0, inplace=True)

    # Extract 'year' from 'name' column
    df['year'] = df['name'].str[-10:-6]
    df = df.drop(['name'], axis=1)

    df['H_ESP'] = df.groupby((['DESC_BARRIO', 'year']))['EspanolesHombres'].transform('sum')
    df['M_ESP'] = df.groupby((['DESC_BARRIO', 'year']))['EspanolesMujeres'].transform('sum')
    df['H_EXT'] = df.groupby((['DESC_BARRIO', 'year']))['ExtranjerosHombres'].transform('sum')
    df['M_EXT'] = df.groupby((['DESC_BARRIO', 'year']))['ExtranjerosMujeres'].transform('sum')

    df = df.drop(['COD_EDAD_INT', 'EspanolesHombres', 'EspanolesMujeres', 'ExtranjerosHombres',
                  'ExtranjerosMujeres'], axis=1)

    df['DESC_BARRIO'] = df['DESC_BARRIO'].str.replace('Ã\x91', 'Ñ')
    df = df.drop_duplicates()
    df.rename(columns={'DESC_BARRIO': "barrio"}, inplace=True)
    df['inmigracion'] = (df['H_EXT'] + df['M_EXT']) / (df['H_ESP'] + df['M_ESP'] + df['H_EXT'] + df['M_EXT'])
    df = df.drop(['H_ESP', 'M_ESP', 'H_EXT', 'M_EXT'], axis=1)
    df['barrio'] = df['barrio'].apply(lambda x: x.strip())
    df = df.drop(df[df.barrio == 'BARRIOS EN EDIF. BDC'].index)
    df = df.drop(df[df.barrio == 'VILLAVERDE ALTO C.H.'].index)

    df.to_csv(os.path.join(folder, 'padron_barrio_15_20.csv'), encoding='latin1', index=False)


def transform_policia(folder):
    # Load DataFrame
    file = os.path.join(os.getcwd(), folder, 'policia_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    # Extract 'year' from 'name' column
    df['year'] = df['name'].str[-11:-7]
    df = df.drop(['name'], axis=1)

    df['incidentes'] = (df['RELACIONADAS CON LAS PERSONAS'] + df['RELACIONADAS CON EL PATRIMONIO'] +
                        df['POR TENENCIA DE ARMAS'] + df['POR TENENCIA DE DROGAS'] + df['POR CONSUMO DE DROGAS'])

    df = df.drop(['RELACIONADAS CON LAS PERSONAS', 'RELACIONADAS CON EL PATRIMONIO', 'POR TENENCIA DE ARMAS',
                    'POR TENENCIA DE DROGAS', 'POR CONSUMO DE DROGAS'], axis=1)

    df.rename(columns={"DISTRITOS": "distrito"}, inplace=True)

    tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
    df['distrito'] = df['distrito'].replace(tildes, regex=True)

    df = df.drop(df[df.distrito.isin(['OTRAS ZONAS', 'TOTAL', 'SIN DISTRITO ASIGNADO',])].index)

    df['distrito'] = df['distrito'].replace(
        ['FUENCARRAL - EL PARDO', 'SAN BLAS - CANILLEJAS', 'MONCLOA - ARAVACA'],
        ['FUENCARRAL-EL PARDO', 'SAN BLAS-CANILLEJAS', 'MONCLOA-ARAVACA'])

    df.to_csv(os.path.join(folder, 'policia_distrito_15_20.csv'), encoding='latin1', index=False)


def transform_poblacion(folder):
    # Load DataFrame
    file = os.path.join(os.getcwd(), folder, 'poblacion_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    df.barrio[df.barrio == 'CENTRO'].index.tolist()

    df.loc[0:148, 'name'] = 2015
    df.loc[149:297, 'name'] = 2016
    df.loc[298:449, 'name'] = 2017
    df.loc[450:601, 'name'] = 2018
    df.loc[602:753, 'name'] = 2019
    df.loc[754:905, 'name'] = 2020

    df.rename(columns={'name': 'year'}, inplace=True)

    list_distritos = ('Centro', 'Arganzuela', 'Retiro', 'Salamanca', 'Chamartín', 'Tetuán', 'Chamberí',
                      'Fuencarral-El Pardo', 'Moncloa-Aravaca', 'Latina', 'Carabanchel', 'Usera',
                      'Puente de Vallecas', 'Moratalaz', 'Ciudad Lineal', 'Hortaleza', 'Villaverde',
                      'Villa de Vallecas', 'Vicálvaro', 'San Blas-Canillejas', 'Barajas')
    list_distritos = [x.upper() for x in list_distritos]
    df = df[~df['barrio'].isin(list_distritos)]
    tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
    df['barrio'] = df['barrio'].replace(tildes, regex=True)
    df['barrio'] = df['barrio'].replace(
        ['CASCO HISTORICO DE BARAJAS', 'CASCO HISTORICO DE VALLECAS', 'CASCO HISTORICO DE VICALVARO',
         'JERONIMOS', 'PEÑAGRANDE', 'PILAR', 'SALVADOR'],
        ['CASCO H.BARAJAS', 'CASCO H.VALLECAS', 'CASCO H.VICALVARO',
         'LOS JERONIMOS', 'PEÑA GRANDE', 'EL PILAR', 'EL SALVADOR'])

    df.to_csv(os.path.join(folder, 'poblacion_barrio_15_20.csv'), encoding='latin1', index=False)


def transform_renta(folder):
    # Load DataFrame
    file = os.path.join(os.getcwd(), folder, 'renta_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    df.barrio[df.barrio == 'CENTRO'].index.tolist()

    df.loc[0:148, 'name'] = 2015
    df.loc[149:297, 'name'] = 2016
    df.loc[298:449, 'name'] = 2017
    df.loc[450:601, 'name'] = 2018
    df.loc[602:753, 'name'] = 2019
    df.loc[754:905, 'name'] = 2020

    df.rename(columns={'name': 'year'}, inplace=True)

    list_distritos = ('Centro', 'Arganzuela', 'Retiro', 'Salamanca', 'Chamartín', 'Tetuán', 'Chamberí',
                      'Fuencarral-El Pardo', 'Moncloa-Aravaca', 'Latina', 'Carabanchel', 'Usera',
                      'Puente de Vallecas', 'Moratalaz', 'Ciudad Lineal', 'Hortaleza', 'Villaverde',
                      'Villa de Vallecas', 'Vicálvaro', 'San Blas-Canillejas', 'Barajas')
    list_distritos = [x.upper() for x in list_distritos]
    df = df[~df['barrio'].isin(list_distritos)]
    tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
    df['barrio'] = df['barrio'].replace(tildes, regex=True)
    df['barrio'] = df['barrio'].replace(
        ['CASCO HISTORICO DE BARAJAS', 'CASCO HISTORICO DE VALLECAS', 'CASCO HISTORICO DE VICALVARO',
         'JERONIMOS', 'PEÑAGRANDE', 'PILAR', 'SALVADOR'],
        ['CASCO H.BARAJAS', 'CASCO H.VALLECAS', 'CASCO H.VICALVARO',
         'LOS JERONIMOS', 'PEÑA GRANDE', 'EL PILAR', 'EL SALVADOR'])

    df.to_csv(os.path.join(folder, 'renta_barrio_15_20.csv'), encoding='latin1', index=False)

def transform_vivienda(folder):
    # Load DataFrames
    file = os.path.join(os.getcwd(), 'data\\vivienda', 'alquiler_distrito_15_20.csv')
    print('loading:', file, '...')
    df_alquiler = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    file = os.path.join(os.getcwd(), 'data\\vivienda', 'venta_barrio_15_20.csv')
    print('loading:', file, '...')
    df_venta = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    df_vivienda = pd.merge(df_venta, df_alquiler, on=['distrito', 'year'])

    # Replace specific values in the 'barrio' column
    df_vivienda['barrio'] = df_vivienda['barrio'].replace(
        ['ROSALES', 'CASCO HISTORICO DE VALLECAS (ENSANCHE DE VALLECAS - VALDECARROS- LA GAVIA)',
         'CASCO HISTORICO DE VICALVARO (VALDEBERNARDO - VALDERRIBAS)', 'AGUILAS', 'ANGELES',
         'CASCO HISTORICO DE VICALVARO (EL CAÑAVERAL - LOS BERROCALES)', 'PALOS DE LA FRONTERA'],
        ['LOS ROSALES', 'ENSANCHE DE VALLECAS',
         'VALDEBERNARDO', 'LAS AGUILAS', 'LOS ANGELES',
         'EL CAÑAVERAL', 'PALOS DE MOGUER'])

    # Transform 'EL GOLOSO' data
    df_vivienda = df_vivienda.drop(df_vivienda[df_vivienda.barrio == 'EL GOLOSO'].index)
    df_vivienda['barrio'] = df_vivienda['barrio'].replace(['EL GOLOSO (PAU MONTECARMELO)'], ['EL GOLOSO'])

    # Transform 'VALDEFUENTES' data
    df_vivienda = df_vivienda.drop(df_vivienda[df_vivienda.barrio.isin(['VALDEFUENTES (VALDEBEBAS - VALDEFUENTES)',
                                                                        'VALDEFUENTES (VIRGEN DEL CORTIJO - MANOTERAS)',
                                                                        'VALDEFUENTES (EL ENCINAR DE LOS REYES)',
                                                                        'VALDEFUENTES'])].index)
    df_vivienda['barrio'] = df_vivienda['barrio'].replace(['VALDEFUENTES (PAU SANCHINARRO)'], ['VALDEFUENTES'])

    # Transform 'VALVERDE' data
    df_vivienda = df_vivienda.drop(df_vivienda[df_vivienda.barrio.isin(['VALVERDE',
                                                                        'VALVERDE (PAU LAS TABLAS)'])].index)
    df_vivienda['barrio'] = df_vivienda['barrio'].replace(['VALVERDE (TRES OLIVOS)'], ['VALVERDE'])

    # Drop 'BUENAVISTA (PAU DE CARABANCHEL)'
    df_vivienda = df_vivienda.drop(df_vivienda[df_vivienda.barrio == 'BUENAVISTA (PAU DE CARABANCHEL)'].index)

    # Add 'VALDERRIVAS' with the same data as 'VALDEBERNARDO'
    duplicated = ['VALDEBERNARDO']
    df1 = df_vivienda[df_vivienda['barrio'].isin(duplicated)].assign(barrio='VALDERRIVAS')
    df_vivienda = pd.concat([df_vivienda, df1], ignore_index=True)

    df_vivienda.to_csv(os.path.join(folder, 'vivienda_barrio_15_20.csv'), encoding='latin1', index=False)
