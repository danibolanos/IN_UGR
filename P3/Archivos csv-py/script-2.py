# -*- coding: utf-8 -*-
"""
Autor:
    Daniel Bolaños Martínez
Fecha:
    Diciembre/2019
Contenido:
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

def eliminaLabels(x, y, tst, etiquetas):
  #se quitan las columnas que no se usan
  x.drop(labels=etiquetas, axis=1, inplace = True)
  y.drop(labels=etiquetas[0], axis=1, inplace = True)
  tst.drop(labels=etiquetas, axis=1, inplace = True)
    
'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''

def catToNum(data):
  mask = data.isnull()
  data_tmp = data.fillna(9999)
  data_tmp = data_tmp.astype(str).apply(LabelEncoder().fit_transform)
  return data_tmp.where(~mask, data_x)

'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

def validacion_cruzada(clf, X, y, splits=5, min_max_scaler = False, normalizer = False):
  cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=76592621)
  validation = 0
  y_train_all, y_test_all = [], []
  for train, test in cv.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    if min_max_scaler:
      X_train = MinMaxScaler().fit_transform(X_train)
      X_test = MinMaxScaler().fit_transform(X_test)
    if normalizer:
     transformer = Normalizer().fit(X_train)
     X_train = transformer.transform(X_train)
     X_test = transformer.transform(X_test)
    t = time.time()
    clf = clf.fit(X_train,y_train)
    training_time = time.time() - t
    predictions_train = clf.predict(X_train)
    predictions = clf.predict(X_test)
    print("--------- Validation ", validation+1, " --------- ")
    print("Tiempo en segundos: ", training_time)
    print ("Train F1-score: ", f1_score(y_train, predictions_train, average='micro'))
    print ("Test F1-score: ", f1_score(y_test, predictions, average='micro'))
    print("")
    y_test_all = np.concatenate([y_test_all, y_test])
    y_train_all = np.concatenate([y_train_all, y_train])
    validation += 1
    
  return clf, y_train_all, y_test_all

'''Función que genera el archivo submission y muestra el F1 score de train total'''
def submission(clf):
  clf = clf.fit(X,y)
  y_pred_tra = clf.predict(X)
  print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
  y_pred_tst = clf.predict(X_tst)
  df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
  df_submission['damage_grade'] = y_pred_tst
  df_submission.to_csv("submission.csv", index=False)
    
if __name__ == '__main__': 
  le = preprocessing.LabelEncoder()
  #los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
  print("Leyendo datos: nepal_earthquake")
  data_x = pd.read_csv('nepal_earthquake_tra.csv')
  data_y = pd.read_csv('nepal_earthquake_labels.csv')
  data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')
  eliminaLabels(data_x, data_y, data_x_tst, ['building_id'])
  X = catToNum(data_x).values
  y = np.ravel(data_y.values)
  X_tst = catToNum(data_x_tst).values
  print("------ RandomForest...")
  rfm = RandomForestClassifier(max_features = 'sqrt', n_estimators=130, max_depth=22, random_state=5)
  # Hago la validación cruzada para el algoritmo
  rfm, y_train_clf, y_test_clf = validacion_cruzada(rfm, X, y)
  print("------ Generando submission...")
  submission(rfm)