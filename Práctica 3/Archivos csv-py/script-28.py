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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import smote_variants as sv 
from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier
import warnings
from sklearn.linear_model import LogisticRegression

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
def submission(X_sample, y_sample, X_tst, clf, cat_features=[]):
  if len(cat_features) == 0:
    clf = clf.fit(X_sample, y_sample)
  else:
    clf = clf.fit(X_sample, y_sample, cat_features=cat_features)
  y_pred_tra = clf.predict(X_sample)
  print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
  y_pred_tst = clf.predict(X_tst)
  df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
  y_pred_tst = y_pred_tst.astype(np.uint8)
  df_submission['damage_grade'] = y_pred_tst
  df_submission.to_csv("submission.csv", index=False)
  
def truncarData(data, label, n):
  etq = data[label].values.tolist()
  new_etq = [n if x >= n else x for x in etq]
  data.drop([label], axis = 1, inplace = True)
  new_etq = np.array(new_etq)
  etq = (new_etq - np.mean(new_etq)) / np.std(new_etq)
  data[label] = etq.T
  return data

def normData(data, label):
  etq = data[label].values
  etq = (etq - min(etq)) / (max(etq) - min(etq))
  data.drop([label], axis = 1, inplace = True)
  data[label] = etq.T
  return data
    
if __name__ == '__main__': 
  warnings.simplefilter('ignore')
  #le = preprocessing.LabelEncoder()
  #los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
  print("Leyendo datos: nepal_earthquake")
  data_x = pd.read_csv('nepal_earthquake_tra.csv')
  data_y = pd.read_csv('nepal_earthquake_labels.csv')
  data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')
  # Elimina etiquetas
  eliminaLabels(data_x, data_y, data_x_tst, ['building_id'])
  # Preprocesado Variables
  '''
  data_x = truncarData(data_x, 'age', 100)
  data_x_tst = truncarData(data_x_tst, 'age', 100)
  data_x = truncarData(data_x, 'count_floors_pre_eq', 4)
  data_x_tst = truncarData(data_x_tst, 'count_floors_pre_eq', 4)
  data_x = truncarData(data_x, 'count_families', 3)
  data_x_tst = truncarData(data_x_tst, 'count_families', 3)
  data_x = normData(data_x, 'area_percentage')
  data_x_tst = normData(data_x_tst, 'area_percentage')
  data_x = normData(data_x, 'height_percentage')
  data_x_tst = normData(data_x_tst, 'height_percentage')
  '''
  # Comprueba balanceo de clases
  # GraficoComprobarVar(data_y, "damage_grade")
  # Comprueba valores perdidos
  #ComprobarValPer(data_x)
  # Calcula la matriz de correlación
  # MatrizCorrelacion(data_x)
  # Preprocesado dummies
  index_categories = np.where(data_x.dtypes =='object')[0]
  X = pd.get_dummies(data_x)
  X_tst = pd.get_dummies(data_x_tst)
  #X = data_x.values
  #X_tst = data_x_tst.values
  y = np.ravel(data_y.values)
  
  #oversampler = sv.MulticlassOversampling(sv.distance_SMOTE(proportion=0.55))

  #X_sample, y_sample = oversampler.sample(X,y)

  #X, y = shuffle(X, y, random_state=76592621)
  #X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=76592621)

  print("------ LightGBM...")
  lgbm = lgb.LGBMClassifier(objective='multiclassova', n_estimators=2695, n_jobs=2,
                            num_leaves=40, max_bin=300,
                            random_state=76592621, learning_rate=0.09)
  clf = BaggingClassifier(base_estimator=lgbm, n_estimators=35, n_jobs=2,
                           bootstrap=False, random_state=76592621)
  '''
  rfm = RandomForestClassifier(max_features = 'auto', criterion='gini', n_estimators=400, \
                               max_depth=23, random_state=76592621, n_jobs=2)
  xgbm = xgb.XGBClassifier(n_estimators=5000, eta=0.1, random_state=76592621)
  lr = LogisticRegression()
  sclf = StackingClassifier(classifiers=[lgbm, rfm, xgbm], 
                          meta_classifier=lr)
  '''
  print("------ Generando submission...")
  submission(X, y, X_tst, clf)
