#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Daniel Bolaños
    Basado en la plantilla aportada por Jorge Casillas
Fecha:
    Noviembre 2019
Asignatura:
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering, Birch, MiniBatchKMeans

from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns

from scipy.cluster import hierarchy
import warnings

def norm_to_zero_one(df):
  return (df - df.min()) * 1.0 / (df.max() - df.min())

# Dibujar Scatter Matrix
def ScatterMatrix(X, name, path):
  print("\nGenerando scatter matrix...")
  sns.set()
  variables = list(X)
  variables.remove('cluster')
  sns_plot = sns.pairplot(X, vars=variables, hue="cluster", palette='Paired', 
                          plot_kws={"s": 25}, diag_kind="hist") 
  #en 'hue' indicamos que la columna 'cluster' define los colores
  sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
   
  plt.savefig(path+"scatmatrix"+name+".png")
  plt.clf()

# Dibujar Heatmap
def Heatmap(X, name, path, dataset, labels):
  print("\nGenerando heat-map...")
  cluster_centers = X.groupby("cluster").mean()
  centers = pd.DataFrame(cluster_centers, columns=list(dataset))
  centers_desnormal = centers.copy()
  #se convierten los centros a los rangos originales antes de normalizar
  for var in list(centers):
    centers_desnormal[var] = dataset[var].min()+centers[var]*(dataset[var].max()-dataset[var].min())

  plt.figure(figsize=(11, 13))
  sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
  plt.savefig(path+"heatmap"+name+".png")
  plt.clf()

# Dibujar Dendogramas (con y sin scatter matrix)
def Dendrograms(X, name, path):
  print("\nGenerando dendogramas...")
  #Para sacar el dendrograma en el jerárquico, no puedo tener muchos elementos.
  #Hago un muestreo aleatorio para quedarme solo con 1000, 
  #aunque lo ideal es elegir un caso de estudio que ya dé un tamaño así
  if len(X)>1000:
     X = X.sample(1000, random_state=seed)
  #Normalizo el conjunto filtrado
  X_filtrado_normal = preprocessing.normalize(X, norm='l2')
  linkage_array = hierarchy.ward(X_filtrado_normal)
  #Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
  hierarchy.dendrogram(linkage_array, orientation='left')
    
  plt.savefig(path+"dendrogram"+name+".png")
  plt.clf()
    
  X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X.index,columns=usadas)
  sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

  plt.savefig(path+"dendscat"+name+".png")
  plt.clf()
  
# Dibujar KdePlot
def KPlot(X, name, k, usadas, path):
  print("\nGenerando kplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharex='col', figsize=(15,10))
  fig.subplots_adjust(wspace=0.2)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)

  for i in range(k):
    dat_filt = X.loc[X['cluster']==i]
    for j in range(n_var):
      sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
  
  plt.savefig(path+"kdeplot"+name+".png")
  plt.clf()
  
# Dibujar BoxPlot
def BoxPlot(X, name, k, usadas, path):
  print("\nGenerando boxplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(16, 16))
  fig.subplots_adjust(wspace=0.4, hspace=0.4)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)
  rango = []

  for i in range(n_var):
    rango.append([X[usadas[i]].min(), X[usadas[i]].max()])

  for i in range(k):
    dat_filt = X.loc[X['cluster']==i]
    for j in range(n_var):
      ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i, j])
      ax.set_xlim(rango[j][0], rango[j][1])
      
  plt.savefig(path+"boxplot"+name+".png")
  plt.clf()


def ejecutarAlgoritmos(algoritmos, X, etiq, usadas, path):
    
  # Crea el directorio si no existe  
  try:
    os.stat(path)
  except:
    os.mkdir(path)    
    
  X_normal = X.apply(norm_to_zero_one)

  # Listas para almacenar los valores
  nombres = []
  tiempos = []
  numcluster = []
  metricaCH = []
  metricaSC = []
    
  for name,alg in algoritmos:    
    print(name,end='')
    t = time.time()
    cluster_predict = alg.fit_predict(X_normal) 
    tiempo = time.time() - t
    k = len(set(cluster_predict))
    print(": clusters: {:3.0f}, ".format(k),end='')
    print("{:6.2f} segundos".format(tiempo))

    # Calculamos los valores de cada métrica
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
    print("\nCalinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. 
    #Si son muchos datos, más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
      m_sil = 0.2
    else:
      m_sil = 1.0
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(m_sil*len(X)), random_state=seed)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    #y se añade como columna a X
    X_cluster = pd.concat([X_normal, clusters], axis=1)
 
    print("\nTamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()

    for num,i in size.iteritems():
      print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
    
    nombre = name+str(etiq)
    
    # Dibujamos el Scatter Matrix
    ScatterMatrix(X = X_cluster, name = nombre, path = path)
    # Dibujamos el Heatmap
    Heatmap(X = X_cluster, name = nombre, path = path, dataset=X, labels = cluster_predict)
    
    # Dibujamos KdePlot
    KPlot(X = X_cluster, name = nombre, k = k, usadas = usadas, path = path)
    
    # Dibujamos BoxPlot
    BoxPlot(X = X_cluster, name = nombre, k = k, usadas = usadas, path = path)

    if name=='AggCluster':
      #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
      min_size = 5
      X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
      k_filtrado = len(set(X_filtrado['cluster']))
      print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
      X_filtrado = X_filtrado.drop('cluster', 1)
      Dendrograms(X = X_filtrado, name = nombre, path = path)
    
    # Almacenamos los datos para generar la tabla comparativa
    nombres.append(name)   
    tiempos.append(tiempo)
    numcluster.append(len(set(cluster_predict)))
    metricaCH.append(metric_CH)
    metricaSC.append(metric_SC)
    
    print("\n-------------------------------------------\n")
    
  # Generamos la tabla comparativa  
  resultados = pd.concat([pd.DataFrame(nombres, columns=['Name']), 
                          pd.DataFrame(numcluster, columns=['Num Clusters']), 
                          pd.DataFrame(metricaCH, columns=['CH']), 
                          pd.DataFrame(metricaSC, columns=['SC']), 
                          pd.DataFrame(tiempos, columns=['Time'])], axis=1)
  print(resultados)

if __name__ == '__main__':  
    
  datos = pd.read_csv('mujeres_fecundidad_INE_2018.csv')
  seed = 76592621
  warnings.filterwarnings(action='ignore', category=FutureWarning)
  warnings.filterwarnings(action='ignore', category=RuntimeWarning)
  
  # Crea el directorio images si no existe  
  try:
    os.stat("./imagenes/")
  except:
    os.mkdir("./imagenes/")    

  #Se pueden reemplazar los valores desconocidos por un número
  #datos = datos.replace(np.NaN,0)
    
  #O imputar, por ejemplo con la media      
  for col in datos:
    datos[col].fillna(datos[col].mean(), inplace=True)
      
  #********************************************   
  # CASO DE ESTUDIO 1 (MUJERES SIN TRABAJO CON AL MENOS 1 HIJO BIOLÓGICO)
  subset = datos.loc[(datos['TRABAJAACT']==6) & (datos['NHIJOBIO']>=1)]
  usadas = ['EDADHIJO1', 'EDADIDEAL', 'ESTUDIOSA', 'NHIJOS', 'EDAD', 'TEMPRELA']
  X = subset[usadas]
  # CASO DE ESTUDIO 2 (MUJERES MENORES DE 30 AÑOS)
  subset2 = datos.loc[datos['EDAD']<=30]
  usadas2 = ['EDAD', 'EDADIDEAL', 'TEMPRELA', 'SATISRELAC', 'NHIJOSDESEO', 'NINTENHIJOS']
  X2 = subset2[usadas2]
  # CASO DE ESTUDIO 3 (MUJERES CASADAS DE MENOS DE 45 AÑOS)
  subset3 = datos.loc[(datos['EC']==2) & (datos['EDAD']<=45)]
  usadas3 = ['EDAD', 'NHIJOS', 'INGREHOG_INTER', 'ABUELOS', 'TEMPRELA', 'PCUIDADOHIJOS']
  X3 = subset3[usadas3]
  #********************************************
       
  # En clustering hay que normalizar para las métricas de distancia
  X_normal = preprocessing.normalize(X, norm='l2')
  X_normal2 = preprocessing.normalize(X2, norm='l2')
  X_normal3 = preprocessing.normalize(X3, norm='l2')
   
  # Algoritmos de clustering utilizados 
  # Caso 1
  k_means1 = KMeans(init='k-means++', n_clusters=3, n_init=5, random_state=seed)
  # Caso 2
  k_means2 = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=seed)
  # Caso 3
  k_means3 = KMeans(init='k-means++', n_clusters=3, n_init=5, random_state=seed)  
  # Caso 1
  ms1 = MeanShift(bandwidth=estimate_bandwidth(X_normal, quantile=0.96, n_samples=400), bin_seeding=True)
  # Caso 2
  ms2 = MeanShift(bandwidth=estimate_bandwidth(X_normal2, quantile=0.90, n_samples=400), bin_seeding=True)  
  # Caso 3
  ms3 = MeanShift(bandwidth=estimate_bandwidth(X_normal3, quantile=0.4, n_samples=400), bin_seeding=True)   
  # Caso 1
  ward1 = AgglomerativeClustering(n_clusters=3, linkage="ward")
  # Caso 2
  ward2 = AgglomerativeClustering(n_clusters=2, linkage="ward")
  # Caso3
  ward3 = AgglomerativeClustering(n_clusters=3, linkage="ward")
  # Caso 1
  db1 = DBSCAN(eps=0.14)
  # Caso 2
  db2 = DBSCAN(eps=0.12)
  # Caso 3
  db3 = DBSCAN(eps=0.18)
  # Caso 1
  brc1 = Birch(branching_factor=25, n_clusters=2, threshold=0.25, compute_labels=True)
  # Caso 2
  brc2 = Birch(branching_factor=25, n_clusters=5, threshold=0.25, compute_labels=True)
  # Caso 3
  brc3 = Birch(branching_factor=25, n_clusters=3, threshold=0.25, compute_labels=True)
  
  algoritmos1 = {('K-Means', k_means1), ('MeanShift', ms1), 
               ('AggCluster', ward1), ('DBSCAN', db1), ('Birch', brc1)}
  
  algoritmos2 = {('K-Means', k_means2), ('MeanShift', ms2), 
               ('AggCluster', ward2), ('DBSCAN', db2), ('Birch', brc2)}
  
  algoritmos3 = {('K-Means', k_means3), ('MeanShift', ms3), 
               ('AggCluster', ward3), ('DBSCAN', db3), ('Birch', brc3)}
 
  # EJECUCIONES DIFERENTES CASOS DE ESTUDIO
  
  path = "./imagenes/"
  
  print("\nCaso de estudio 1 , tamaño: "+str(len(X))+"\n")
  print("-------------------------------------------\n")
  ejecutarAlgoritmos(algoritmos1, X, "caso1", usadas, path+"caso1/")
  print("-------------------------------------------")
  print("\nCaso de estudio 2 , tamaño: "+str(len(X2))+"\n")
  ejecutarAlgoritmos(algoritmos2, X2, "caso2", usadas2, path+"caso2/")
  print("-------------------------------------------")
  print("\nCaso de estudio 3 , tamaño: "+str(len(X3))+"\n")
  ejecutarAlgoritmos(algoritmos3, X3, "caso3", usadas3, path+"caso3/")