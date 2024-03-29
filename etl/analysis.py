import os
import pandas as pd
import numpy as np
from sklearn import preprocessing, cluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def graph_indicadores(folder):
    # Paths and files
    file = os.path.join(os.getcwd(), folder, 'indicadores_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    barrios = df['barrio'].unique()
    indicadores = df.columns[3:]
    indicadores = indicadores.tolist()

    for indicador in indicadores:
        # print(indicador)
        ind_20_15 = []
        for barrio in barrios:
            value_2020 = df.loc[(df['year'] == 2020) & (df['barrio'] == barrio), indicador].iloc[0]
            value_2015 = df.loc[(df['year'] == 2015) & (df['barrio'] == barrio), indicador].iloc[0]
            if value_2015 != 0:
                delta = (value_2020 - value_2015) / value_2015 * 100
            ind_20_15.append({'barrio': barrio, 'delta': delta})
        df_indicador = pd.DataFrame(ind_20_15).sort_values(by='delta', ascending=False).nlargest(10, 'delta')
        # print(df_indicador)
        filename = f"{indicador}_evolution.csv"
        df_indicador.to_csv(os.path.join(folder, filename), encoding='latin1', index=False)

    if not os.path.exists(f"images/indicadores"):
        os.mkdir(f"images/indicadores")

    print('Saving images for each indicator / barrio...')

    for barrio in barrios:
        for indicador in indicadores:
            df_ind_barrio = df.loc[(df['barrio'] == barrio)]

            if not os.path.exists(f"images/indicadores/{indicador}"):
                os.mkdir(f"images/indicadores/{indicador}")

            filename = f"images/indicadores/{indicador}/{barrio}.png"

            # Plot the figure
            fig, ax = plt.subplots()
            df_ind_barrio.plot(x='year', y=indicador, ax=ax)

            # Save the figure
            fig.savefig(filename)

            # Close the figure
            fig.clf()
            plt.close()
            ax.cla()

    print('Images have been saved.')


def analyse_indicadores(folder):
    # Paths and files
    file = os.path.join(os.getcwd(), folder, 'indicadores_15_20.csv')
    print('loading:', file, '...')
    df = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)

    df['barrio'] = df['barrio'].replace(
        ['LOS ANGELES', 'CONCEPCION', 'CASCO H.VALLECAS', 'CASCO H.BARAJAS', 'ARGUELLES',
         'PEÑA GRANDE', 'CASCO H.VICALVARO', 'LAS AGUILAS', 'EL PILAR', 'PALOS DE MOGUER'],
        ['ANGELES', 'LA CONCEPCION', 'CASCO HISTORICO DE VALLECAS', 'CASCO HISTORICO DE BARAJAS', 'ARGÜELLES',
         'PEÑAGRANDE', 'CASCO HISTORICO DE VICALVARO', 'AGUILAS', 'PILAR', 'PALOS DE LA FRONTERA'])

    dataset = df.copy()

    df = df[df['year'].isin([2015, 2020])]

    indicadores = df.columns[3:]
    indicadores = indicadores.tolist()

    for indicador in indicadores:
        df[indicador] = np.where(df['year'] == 2015, -1 * df[indicador], df[indicador])

    df = (
        df
        .groupby(['barrio'])
        .agg(venta=('venta', 'sum'),
             alquiler=('alquiler', 'sum'),
             edad_media=('edad_media', 'sum'),
             tamano_medio=('tamano_medio', 'sum'),
             poblacion=('poblacion', 'sum'),
             renta_media=('renta_media', 'sum'),
             inmigracion=('inmigracion', 'sum'),
             admin_publica=('admin_publica', 'sum'),
             comercio_mayor=('comercio_mayor', 'sum'),
             comercio_menor=('comercio_menor', 'sum'),
             construccion=('construccion', 'sum'),
             educacion=('educacion', 'sum'),
             finanzas=('finanzas', 'sum'),
             hoteles=('hoteles', 'sum'),
             inmobiliarias=('inmobiliarias', 'sum'),
             ocio=('ocio', 'sum'),
             restaurantes=('restaurantes', 'sum'),
             sanidad=('sanidad', 'sum'),
             incidentes=('incidentes', 'sum'),
             )
        .reset_index()
    )

    print('Scaling...')

    df2 = df.copy()
    df2 = df2.drop(columns=['barrio'])
    x = df2.values  # returns a numpy array
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    df2 = pd.DataFrame(x_scaled)

    print('Launching correlation matrix...')

    corr_matrix = df2.corr()
    corr_matrix = np.round(corr_matrix, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt="g", cmap='viridis', ax=ax,
                xticklabels=indicadores, yticklabels=indicadores)
    plt.tight_layout()

    if not os.path.exists(f"images/analysis"):
        os.mkdir(f"images/analysis")

    filename = f"images/analysis/corr_matrix.png"

    # Save the figure
    fig.savefig(filename)

    # Close the figure
    fig.clf()
    plt.close()
    ax.cla()

    lista = [4]
    [df2.pop(x) for x in lista]
    [indicadores.pop(x) for x in lista]

    lista = [3]
    [df2.pop(x) for x in lista]
    [indicadores.pop(x) for x in lista]

    lista = [2]
    [df2.pop(x) for x in lista]
    [indicadores.pop(x) for x in lista]

    corr_matrix = df2.corr()
    corr_matrix = np.round(corr_matrix, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt="g", cmap='viridis', ax=ax,
                xticklabels=indicadores, yticklabels=indicadores)
    plt.tight_layout()

    filename = f"images/analysis/corr_matrix_clean.png"
    # Save the figure
    fig.savefig(filename)

    # Close the figure
    fig.clf()
    plt.close()
    ax.cla()

    print('Correlation matrixes saved.')

    print('Launching PCA and Kmeans...')

    pca = PCA(n_components=6)
    pca.fit(df2)
    print(f'Explained variance: {pca.explained_variance_ratio_}')

    def plot_elbow(sse, ks):
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.set_title('Elbow method for optimal k')
        axis.set_xlabel('k')
        axis.set_ylabel('SSE')
        plt.plot(ks, sse, marker='o')
        plt.tight_layout()
        #plt.show()
        filename = f"images/analysis/elbow.png"
        fig.savefig(filename)
        fig.clf()
        plt.close()
        axis.cla()

    def plot_silhouette(sils, ks):
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.set_title('Silhouette method')
        axis.set_xlabel('k')
        axis.set_ylabel('Silhouette')
        plt.plot(ks, sils, marker='o')
        plt.tight_layout()
        #plt.show()
        filename = f"images/analysis/silhouette.png"
        fig.savefig(filename)
        fig.clf()
        plt.close()
        axis.cla()

    def elbow_method(df):
        sse = []
        ks = range(2, 10)
        for k in ks:
            k_means_model = cluster.KMeans(n_clusters=k, random_state=55, n_init='auto')
            k_means_model.fit(df)
            sse.append(k_means_model.inertia_)
        plot_elbow(sse, ks)

    def silhouette_method(df):
        ks = range(2, 10)
        sils = []
        for k in ks:
            clusterer = KMeans(n_clusters=k, random_state=55, n_init='auto')
            cluster_labels = clusterer.fit_predict(df)
            silhouette_avg = silhouette_score(df, cluster_labels)
            sils.append(silhouette_avg)
            # print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        plot_silhouette(sils, ks)

    silhouette_method(df2)
    elbow_method(df2)

    print('Kmeans methods passed.')

    df = df.drop(['edad_media', 'tamano_medio', 'poblacion'], axis=1)

    indicadores = ['renta_media', 'venta', 'hoteles', 'restaurantes', 'inmigracion']

    if not os.path.exists(f"analysis"):
        os.mkdir(f"analysis")

    clusterer = KMeans(n_clusters=4, random_state=55, n_init='auto')
    cluster_labels = clusterer.fit_predict(df2)
    df['cluster_k4'] = cluster_labels
    k4 = df[['barrio', 'cluster_k4']]
    dataset_k4 = pd.merge(dataset, k4, on=['barrio'])
    dataset_k4.to_csv('analysis/dataset_k4.csv', index=False)

    barrios_out = ['AMBROZ', 'EL CAÑAVERAL', 'VALDEBERNARDO', 'VALDERRIVAS', 'ENSANCHE DE VALLECAS']
    dataset_k3 = dataset_k4.copy()
    dataset_k3 = dataset_k3[~dataset_k3['barrio'].isin(barrios_out)]
    dataset_k3 = dataset_k3.rename(columns={'cluster_k4': 'cluster_k3'})
    dataset_k3['cluster_k3'] = dataset_k3['cluster_k3'].apply(str)
    dataset_k3['cluster_k3'] = dataset_k3['cluster_k3'].replace(['0'], 'Sin riesgo')
    dataset_k3['cluster_k3'] = dataset_k3['cluster_k3'].replace(['1'], 'Con riesgo')
    dataset_k3['cluster_k3'] = dataset_k3['cluster_k3'].replace(['2'], 'Gentrificado')
    dataset_k3.to_csv('analysis/dataset_k3.csv', index=False)

    dataset_k4 = (
        dataset_k4
        .groupby(['year', 'cluster_k4'])
        .agg(venta=('venta', 'mean'),
             alquiler=('alquiler', 'mean'),
             renta_media=('renta_media', 'mean'),
             inmigracion=('inmigracion', 'mean'),
             admin_publica=('admin_publica', 'mean'),
             comercio_mayor=('comercio_mayor', 'mean'),
             comercio_menor=('comercio_menor', 'mean'),
             construccion=('construccion', 'mean'),
             educacion=('educacion', 'mean'),
             finanzas=('finanzas', 'mean'),
             hoteles=('hoteles', 'mean'),
             inmobiliarias=('inmobiliarias', 'mean'),
             ocio=('ocio', 'mean'),
             restaurantes=('restaurantes', 'mean'),
             sanidad=('sanidad', 'mean'),
             incidentes=('incidentes', 'mean'),
             )
        .reset_index()
    )

    if not os.path.exists(f"images/clusters_k4"):
        os.mkdir(f"images/clusters_k4")

    for clusters in cluster_labels:
        for indicador in indicadores:
            df_ind_clust = dataset_k4.loc[(dataset_k4['cluster_k4'] == clusters)]

            filename = f"images/clusters_k4/{clusters}_{indicador}.png"

            # Plot the figure
            fig, ax = plt.subplots()
            df_ind_clust.plot(x='year', y=indicador, ax=ax)

            # Save the figure
            fig.savefig(filename)

            # Close the figure
            fig.clf()
            plt.close()
            ax.cla()

    print('K4 images have been saved.')

    dataset_k4.to_csv('analysis/dataset_clusters_4.csv', index=False)

    print('Kmeans with 4 clusters saved.')

    clusterer = KMeans(n_clusters=6, random_state=55, n_init='auto')
    cluster_labels = clusterer.fit_predict(df2)
    df['cluster_k6'] = cluster_labels
    k6 = df[['barrio', 'cluster_k6']]
    dataset_k6 = pd.merge(dataset, k6, on=['barrio'])
    dataset_k6.to_csv('analysis/dataset_k6.csv', index=False)

    dataset_k6 = pd.read_csv(file, sep=',', encoding='unicode_escape', header=0)
    dataset_k4b = dataset_k6.copy()
    dataset_k4b = dataset_k4b[~dataset_k4b['barrio'].isin(barrios_out)]
    dataset_k4b = dataset_k4b.rename(columns={'cluster_k6': 'cluster_k4b'})
    dataset_k4b['cluster_k4b'] = dataset_k4b['cluster_k4b'].apply(str)
    dataset_k4b['cluster_k4b'] = dataset_k4b['cluster_k4b'].replace(['0'], 'Sin riesgo')
    dataset_k4b['cluster_k4b'] = dataset_k4b['cluster_k4b'].replace(['1'], 'Riesgo medio')
    dataset_k4b['cluster_k4b'] = dataset_k4b['cluster_k4b'].replace(['4'], 'Riesgo alto')
    dataset_k4b['cluster_k4b'] = dataset_k4b['cluster_k4b'].replace(['5'], 'Gentrificado')
    dataset_k4b.to_csv('analysis/dataset_k4b.csv', index=False)

    dataset_k6 = (
        dataset_k6
        .groupby(['year', 'cluster_k6'])
        .agg(venta=('venta', 'mean'),
             alquiler=('alquiler', 'mean'),
             renta_media=('renta_media', 'mean'),
             inmigracion=('inmigracion', 'mean'),
             admin_publica=('admin_publica', 'mean'),
             comercio_mayor=('comercio_mayor', 'mean'),
             comercio_menor=('comercio_menor', 'mean'),
             construccion=('construccion', 'mean'),
             educacion=('educacion', 'mean'),
             finanzas=('finanzas', 'mean'),
             hoteles=('hoteles', 'mean'),
             inmobiliarias=('inmobiliarias', 'mean'),
             ocio=('ocio', 'mean'),
             restaurantes=('restaurantes', 'mean'),
             sanidad=('sanidad', 'mean'),
             incidentes=('incidentes', 'mean'),
             )
        .reset_index()
    )

    if not os.path.exists(f"images/clusters_k6"):
        os.mkdir(f"images/clusters_k6")

    for clusters in cluster_labels:
        for indicador in indicadores:
            df_ind_clust = dataset_k6.loc[(dataset_k6['cluster_k6'] == clusters)]

            filename = f"images/clusters_k6/{clusters}_{indicador}.png"

            # Plot the figure
            fig, ax = plt.subplots()
            df_ind_clust.plot(x='year', y=indicador, ax=ax)

            # Save the figure
            fig.savefig(filename)

            # Close the figure
            fig.clf()
            plt.close()
            ax.cla()

    print('K6 images have been saved.')

    dataset_k6.to_csv('analysis/dataset_clusters_6.csv', index=False)

    print('Kmeans with 6 clusters saved.')
