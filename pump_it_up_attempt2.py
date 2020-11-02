#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:00:04 2020

@author: matt
"""

#%% import packages 
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#%% read data
train = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')

test = pd.read_csv('test_values.csv')

#%%
numeric_recode = {'functional' : 1,
                  'non functional' : 2,
                  'functional needs repair' : 3
                  }

train_labels['status_group'] = train_labels.status_group.map(numeric_recode)

train_labels.set_index("id", inplace = True)


#%% clean up data

dta = pd.concat([train, test])

features = ['id', 
            #'amount_tsh', 
            'funder', 'gps_height',
               'installer', 'longitude', 'latitude', 'date_recorded',
               'basin',  'region_code', 'district_code', 'lga',
               'ward', 'population', 
               'scheme_management', 
               'management', 
               'permit', 'construction_year',
               'extraction_type_group', 
               'payment',
               'water_quality', 'quantity',
               'source_type','waterpoint_type']

dta = dta[features]

# dta['month_recorded'] = [i.month for i in pd.to_datetime(dta['date_recorded'])]
#tdiff = [pd.to_datetime("2014-01-01") - i for i in pd.to_datetime(dta['date_recorded'])]
#dta['recorded_offset_days '] = [i.days for i in tdiff]
# clean up text vars -- all text to lowercase
dta = dta.applymap(lambda col:col.lower() if type(col) == str else col)

#dta.loc[dta['installer'].isin(['central government', 'gover', 'gove']), 'installer'] = "government"
#dta.loc[~dta['installer'].isin(dta['installer'].value_counts()[:50].index.tolist()), 'installer'] = 'other'

# dta.loc[dta['funder'].isin(['roman', 'rc']), 'funder'] = "rc church"
# dta.loc[dta['funder'].isin(['water']), 'funder'] = "ministry of water"
# dta.loc[dta['funder'].isin(['private']), 'funder'] = "private individual"
# dta.loc[dta['funder'].isin(['ces (gmbh)']), 'funder'] = "ces(gmbh)"
# dta.loc[dta['funder'].isin(['w.b']), 'funder'] = "world bank"

dta[['permit']] = pd.to_numeric(dta['permit'])

dta.loc[dta['construction_year'] == 0, 'construction_year'] = np.nan
dta.loc[dta['funder'] == '0', 'funder'] = np.nan
dta.loc[dta['gps_height'] == '0', 'gps_height'] = np.nan
dta.loc[dta['population'] == '0', 'population'] = np.nan
# dta.loc[dta['amount_tsh'] == '0', 'amount_tsh'] = np.nan

# clean up non-word characters
numeric_cols = dta.select_dtypes(include = np.number)

text_cols = dta.select_dtypes('object')
text_cols = text_cols.apply(lambda col: col.str.replace('\\W+', '_'), axis = 1)

dta = pd.concat([numeric_cols, text_cols], axis = 1, ignore_index = False)

dta['msg_construction_year'] = dta.construction_year.isnull().astype(int)

#dta['pre_1980'] = dta.construction_year.values < 1980
#dta['pre_1980'] = dta.pre_1980.astype(int)
#dta['post_2000'] = dta.construction_year.values >= 2000
#dta['post_2000'] = dta.post_2000.astype(int)
#dta['pump_age'] = 2013 - dta.construction_year.values
dta['msg_population'] = dta.population.isnull().astype(int)
#dta['msg_amount_tsh'] = dta.amount_tsh.isnull().astype(int)
#dta['msg_gps_height'] = dta.gps_height.isnull().astype(int)
#dta.drop('amount_tsh', axis = 1, inplace = True)
# dta['qty_enough'] = dta.quantity.isin(['insufficient', 'dry', 'seasonal', 'unknown']).astype(int)

#%% separate data again
train = dta.loc[dta.id.isin(train.id)]
test = dta.loc[dta.id.isin(test.id)]

#train = train.join(train_labels, on = 'id')

#%% categorical encoding

targ_enc = ce.TargetEncoder(cols = None)
targ_enc.fit(train, train_labels['status_group'])
train = targ_enc.transform(train)

#%% impute

imp = IterativeImputer(max_iter=10, max_value = 2013)
imp.fit(train)

train = pd.DataFrame(imp.transform(train), columns = train.columns)
train['construction_year'] = train['construction_year'].round(0)
train['permit'] = train['permit'].round(0)

#%% scale
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(train.drop('id', axis = 1))
# train = pd.DataFrame(imp.transform(train), columns = train.columns)

#%%
cat_recode = {1 : 'functional',
              2 : 'non functional',
              3 : 'functional needs repair'}

#%% model 1

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf_train = train.drop(['id'], axis = 1)
rf_vals = train_labels.status_group.map(cat_recode)
rf = RandomForestClassifier(max_depth = 20, random_state = 23456)

cvs = cross_val_score(rf, rf_train, rf_vals, scoring = 'accuracy', cv = 5)
print(cvs)
#array([0.81372054, 0.81094276, 0.81405724, 0.80968013, 0.81220539]) # added population and amount_tsh
#array([0.81372054, 0.80968013, 0.81498316, 0.81144781, 0.81127946]) # added quantity_enough flag. Modest improvement
#array([0.81388889, 0.80951178, 0.81531987, 0.80816498, 0.81195286]) # added month_recorded, Modest improvement
#array([0.81632997, 0.81102694, 0.81329966, 0.81085859, 0.81161616]) # added time diff since recorded, generally better
#array([0.81776094, 0.81052189, 0.81355219, 0.80976431, 0.81565657]) # recoded government installers, modest improvement in most folds
#array([0.81776094, 0.81069024, 0.81430976, 0.81144781, 0.81271044]) # bucketized funder a bit, modest improvement
#array([0.81734007, 0.81380471, 0.81464646, 0.80984848, 0.80799663]) # optimized xgb

rf.fit(rf_train, rf_vals)

#%% feature importance
# feature_importances = pd.DataFrame(rf.feature_importances_,
#                                    index = train.drop('id', axis = 1).columns,
#                                     columns=['importance']).sort_values('importance',   
#                                                                         ascending=False)
                                                                        
# print(feature_importances)


#%% boosting
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state = 23456,
                                 max_depth=10
                                 )

cvs = cross_val_score(gbc, rf_train, rf_vals, scoring = 'accuracy', cv = 5)
print(cvs)

#%%
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

xgb = XGBClassifier(objective = 'multi:softmax',
                    random_state = 23456)

xgb_params_dists = {"max_depth" : range(1, 51, 1),
                    "min_child_weight" : range(1, 21, 1),
                    "learning_rate" : np.arange(0.0, 1.01, 0.01),
                    "gamma" : reciprocal(0.001, 0.1),
                    "max_delta_step" : range(0, 11, 1)
                    }

xgb_optimized = RandomizedSearchCV(xgb, xgb_params_dists, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

best_xgb_model = xgb_optimized.fit(rf_train, rf_vals)

#%%
best_xgb_model.best_estimator_.get_params()
best_xgb_model.best_score_
cross_val_score(xgb, rf_train, rf_vals, scoring = 'accuracy', cv = 5)

#%% optimized params for xgb
xgb_params =  {'objective': 'multi:softprob',
  'base_score': 0.5,
  'booster': 'gbtree',
  'colsample_bylevel': 1,
  'colsample_bynode': 1,
  'colsample_bytree': 1,
  'gamma': 0.029521058580772683,
  'gpu_id': -1,
  'importance_type': 'gain',
  'interaction_constraints': '',
  'learning_rate': 0.23,
  'max_delta_step': 0,
  'max_depth': 15,
  'min_child_weight': 15,
  'missing': np.nan,
  'monotone_constraints': '()',
  'n_estimators': 100,
  'n_jobs': 0,
  'num_parallel_tree': 1,
  'random_state': 23456,
  'reg_alpha': 0,
  'reg_lambda': 1,
  'scale_pos_weight': None,
  'subsample': 1,
  'tree_method': 'exact',
  'validate_parameters': 1,
  'verbosity': None}

xgb_params_upd = {'objective': 'multi:softprob',
 'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0.001369048068388758,
 'gpu_id': -1,
 'importance_type': 'gain',
 'interaction_constraints': '',
 'learning_rate': 0.11,
 'max_delta_step': 10,
 'max_depth': 15,
 'min_child_weight': 9,
 'missing': np.nan,
 'monotone_constraints': '()',
 'n_estimators': 100,
 'n_jobs': 0,
 'num_parallel_tree': 1,
 'random_state': 23456,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': None,
 'subsample': 1,
 'tree_method': 'exact',
 'validate_parameters': 1,
 'verbosity': None}
#xgb.fit(rf_train, rf_vals)
#%% xgb 2
from xgboost import XGBClassifier

xgb = XGBClassifier(**xgb_params_upd)
cross_val_score(xgb, rf_train, rf_vals, scoring = 'accuracy', cv = 5)

#%% stack attempt 

from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', RandomForestClassifier(max_depth = 20, random_state = 23456)),
    ('xgb', XGBClassifier(**xgb_params_upd)),
    ('gbc', GradientBoostingClassifier(random_state = 23456, max_depth = 10))
    ]
#%%

clf = StackingClassifier(estimators)
clf.fit(rf_train, rf_vals)

cvs = cross_val_score(clf, rf_train, rf_vals, scoring = 'accuracy', cv = 5)
print(cvs)

 #%% generate rf submission

test = targ_enc.transform(test)
test = pd.DataFrame(imp.transform(test), columns = test.columns)
test['construction_year'] = test['construction_year'].round(0)
test['permit'] = test['permit'].round(0)

test['status_group'] = rf.predict(test.drop('id', axis = 1))
test[['id', 'status_group']].to_csv("rf_preds_best_sub_plus_recorded_installer_recodes.csv", index = False)
# scored .8228 with pop

test['status_group'] = xgb.predict(test.drop('id', axis = 1))
test[['id', 'status_group']].to_csv("xgb_preds_md15_mcw3_etapt1.csv", index = False)
# scored .8154

test['status_group'] = clf.predict(test.drop('id', axis = 1))
test[['id', 'status_group']].to_csv("rf_xgb_gbc_ensemble2.csv", index = False)

#%%% svc sub
test = targ_enc.transform(test)
test = pd.DataFrame(imp.transform(test), columns = test.columns)
test['construction_year'] = test['construction_year'].round(0)
test['permit'] = test['permit'].round(0)
scaled = scaler.transform(test.drop('id', axis = 1))

test['status_group'] = best_model.predict(scaled)

test['status_group'] = test.status_group.map(cat_recode)
test[['id', 'status_group']].to_csv("svm_preds_optmizedCandGamma.csv", index = False)
# scored .7908
