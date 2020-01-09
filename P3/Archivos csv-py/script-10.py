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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import smote_variants as sv 

def GraficoComprobarVar(labels_tra, label, path="./imagenes/"):
  plt.figure(figsize=(10,8))
  ax = sns.countplot(label, data=labels_tra)
  for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.2, i.get_height()+3, \
            str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
  plt.savefig(path+label+".png")
  plt.clf()
  
def ComprobarValPer(data_tra, path="./imagenes/"):
  data_tra.isnull().sum().plot.bar()
  plt.savefig(path+"valores_perdidos"+".png")
  plt.clf()
  
def MatrizCorrelacion(data_tra, path="./imagenes/"):
  sns.heatmap(data_tra.corr())
  plt.xticks(rotation =45)
  plt.savefig(path+"matriz_corr"+".png")
  plt.clf()
  
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
  return data_tmp.where(~mask, data)

'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

def validacion_cruzada(clf, X, y, splits=5):
  cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=76592621)
  validation = 1
  y_train_all, y_test_all = [], []
  for train, test in cv.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    t = time.time()
    clf = clf.fit(X_train,y_train)
    training_time = time.time() - t
    predictions_train = clf.predict(X_train)
    predictions = clf.predict(X_test)
    print("--------- Validation ", validation, " --------- ")
    print("Tiempo en segundos: ", training_time)
    print ("Train F1-score: ", f1_score(y_train, predictions_train, average='micro'))
    print ("Test F1-score: ", f1_score(y_test, predictions, average='micro'))
    print("")
    y_test_all = np.concatenate([y_test_all, y_test])
    y_train_all = np.concatenate([y_train_all, y_train])
    validation += 1
    
  return clf, y_train_all, y_test_all

'''Función que genera el archivo submission y muestra el F1 score de train total'''
def submission(X, y, X_tst, clf):
  clf = clf.fit(X,y)
  y_pred_tra = clf.predict(X)
  print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
  y_pred_tst = clf.predict(X_tst)
  df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
  y_pred_tst = y_pred_tst.astype(np.uint8)
  df_submission['damage_grade'] = y_pred_tst
  df_submission.to_csv("submission.csv", index=False)
    
if __name__ == '__main__': 
  #le = preprocessing.LabelEncoder()
  #los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
  print("Leyendo datos: nepal_earthquake")
  data_x = pd.read_csv('nepal_earthquake_tra.csv')
  data_y = pd.read_csv('nepal_earthquake_labels.csv')
  data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')
  # Comprueba balanceo de clases
  # GraficoComprobarVar(data_y, "damage_grade")
  # Comprueba valores perdidos
  #ComprobarValPer(data_x)
  # Calcula la matriz de correlación
  #MatrizCorrelacion(data_x)
  # Elimina etiquetas
  eliminaLabels(data_x, data_y, data_x_tst, ['building_id'])
  # Preprocesado category to number

  X = catToNum(data_x).values
  y = np.ravel(data_y.values)
  X_tst = catToNum(data_x_tst).values
  
  oversampler = sv.MulticlassOversampling(sv.distance_SMOTE(proportion=0.25))

  X_sample, y_sample = oversampler.sample(X,y)
  '''
  print("------ RandomForest...")
  rfm = RandomForestClassifier(max_features = 'sqrt', criterion='gini', n_estimators=500, \
                               max_depth=25, random_state=76592621, \
                               n_jobs=-1)
  # Hago la validación cruzada para el algoritmo
  #rfm, y_train_clf, y_test_clf = validacion_cruzada(rfm, X_sample, y_sample)
  print("------ Generando submission...")
  submission(X_sample, y_sample, X_tst, rfm)
  '''
  '''
  print("------ Catboost...")
  cbc = CatBoostClassifier(n_estimators=500, learning_rate=0.12, eval_metric='TotalF1',
  od_pval=0.001, od_type='IncToDec', random_seed=76592621, bootstrap_type='Bayesian',
  max_depth=13, best_model_min_trees=250, loss_function='MultiClass')
  #cbc, y_train_clf, y_test_clf = validacion_cruzada(cbc, X_sample, y_sample, 2)
  print("------ Generando submission...")
  submission(X_sample, y_sample, X_tst, cbc)
  param_dist = {
  'max_depth': range(45, 50),
  'n_estimators': [250, 300],
  'class_weight': [{1:p, 2:s, 3:t} for p in [0.6, 0.5] for s in [0.1, 0.2] for t in [0.2, 0.3]]
  }
  '''
  print("------ Catboost...")
  cbc = CatBoostClassifier(n_estimators=600, learning_rate=0.12, eval_metric='TotalF1',
  od_pval=0.001, od_type='IncToDec', random_seed=76592621, bootstrap_type='MVS',
  max_depth=12, best_model_min_trees=250, loss_function='MultiClass')
  cbc, y_train_clf, y_test_clf = validacion_cruzada(cbc, X_sample, y_sample, 2)
  print("------ Generando submission...")
  submission(X_sample, y_sample, X_tst, cbc)
  '''
  fit_alg = CatBoostClassifier(eval_metric='TotalF1',
  od_pval=0.001, od_type='IncToDec', random_seed=76592621, 
  best_model_min_trees=250, loss_function='MultiClass')
  param_dist = {
  'max_depth': [12],
  'n_estimators': [500, 450, 600],
  'learning_rate': [0.10, 0.12],
  'bootstrap_type':['MVS']
  }
  clf = GridSearchCV(fit_alg, param_dist, verbose=1, cv=2, scoring='f1_micro', n_jobs=2)
  clf = clf.fit(X_sample,y_sample)
  best_param1 = clf.best_params_['max_depth']
  best_param2 = clf.best_params_['n_estimators']
  best_param3 = clf.best_params_['learning_rate']
  best_param4 = clf.best_params_['bootstrap_type']
  print ("Mejor valor para max_depth: ", best_param1)
  print ("Mejor valor para n_estimators: ", best_param2)
  print ("Mejor valor para learning_rate: ", best_param3)
  print ("Mejor valor para bootstrap_type: ", best_param4)
  '''