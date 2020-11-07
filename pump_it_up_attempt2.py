#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:00:04 2020

@author: matt
"""

#%% import packages 
import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
#from scipy.stats import reciprocal

#%% read data
train = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')

test = pd.read_csv('test_values.csv')

#%%
numeric_recode = {'functional' : 1,
                  'non functional' : 2,
                  'functional needs repair' : 3
                  }

cat_recode = {1 : 'functional',
              2 : 'non functional',
              3 : 'functional needs repair'}

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

#dta['month_recorded'] = [i.month for i in pd.to_datetime(dta['date_recorded'])]
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


#%% model 1

#rf_train = train.drop(['id'], axis = 1)
#rf_vals = train_labels.status_group.map(cat_recode)

#rf_train = train.drop(['id'], axis = 1)
#rf_vals = train_labels.status_group

#%% rf
rf = RandomForestClassifier(max_features = 9, max_depth = 20, random_state = 45678)
cvs = cross_val_score(rf, rf_train, rf_vals, scoring = 'accuracy', cv = 5)
print(cvs)
print('Mean Accuracy : {}'.format(np.mean(cvs).round(6)))
# random state 23456, part of best model
# [0.81338384 0.80951178 0.81094276 0.80841751 0.81026936]
# Mean Accuracy : 0.810505

# rs 45678 up in folds 1 and 3
# [0.81574074 0.80808081 0.81254209 0.80698653 0.80791246]
# Mean Accuracy : 0.810253

# rs 56789, up in folds 2,3
# [0.81262626 0.81069024 0.81329966 0.80538721 0.80909091]
# Mean Accuracy : 0.810219

#rf.fit(rf_train, rf_vals)

#%% feature importance
# feature_importances = pd.DataFrame(rf.feature_importances_,
#                                    index = train.drop('id', axis = 1).columns,
#                                     columns=['importance']).sort_values('importance',   
#                                                                         ascending=False)
                                                                        
# print(feature_importances)



#%% boosting

gbc = GradientBoostingClassifier(init = RandomForestClassifier(max_depth = 20),
                                  subsample = .85,
                                  n_iter_no_change = 10,
                                  n_estimators=(1000),
                                  learning_rate=(.25),
                                  random_state=(23456))

cvs = cross_val_score(gbc, rf_train, rf_vals, scoring = 'accuracy', cv = 5, n_jobs = -1)

print(cvs)
print('Mean Accuracy : {}'.format(np.mean(cvs).round(6)))
# [0.81254209 0.80858586 0.80867003 0.80589226 0.80858586] # md 10
# Mean Accuracy : 0.808855
# [0.81464646 0.80968013 0.81464646 0.80782828 0.80681818] # md 10 and subsample .85
# Mean Accuracy : 0.810724
# [0.81203704 0.80698653 0.80909091 0.80521886 0.80606061] # md 10, subsample .85 and 500 estimators: actually worse
# Mean Accuracy : 0.807879

#%%
bag = BaggingClassifier(n_estimators = 500,
                        n_jobs = -1,
                        max_samples = .85,
                        oob_score = (True),
                        random_state=(23456),
                        max_features=(.7)
                        )

#%%

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
cross_val_score(xgb, rf_train, rf_vals, scoring = 'accuracy', cv = 5, n_jobs = -1)

#%% optimized params for xgb

xgb_params_upd = {'objective': 'multi:softmax',
 'gamma': 0.001369048068388758,
 'eta': 0.11,
 'max_depth': 15,
 'min_child_weight': 9,
 'num_class' : 4,
 'subsample' : .85,
 'colsample_bytree' : .9
 }
#xgb.fit(rf_train, rf_vals)
#%% xgb 2

xgb = XGBClassifier(**xgb_params_upd)
cross_val_score(xgb, rf_train, rf_vals, scoring = 'accuracy', cv = 5, n_jobs = -1)

#%% stack attempt 

# best model to date
estimators = [
    ('rf1', RandomForestClassifier(max_depth = 20, max_features = 9, random_state = 23456)),
    ('rf2', RandomForestClassifier(max_depth = 20, max_features = 9, random_state = 45678)),
    ('rf3', RandomForestClassifier(max_depth = 20, max_features = 9, random_state = 56789)),
    ('xgb', XGBClassifier(**xgb_params_upd)),
    ('gbc', GradientBoostingClassifier(init = RandomForestClassifier(max_depth = 20, random_state = 45678),
                                  subsample = .85,
                                  n_iter_no_change = 10,
                                  n_estimators = 1000,
                                  learning_rate = .25,
                                  random_state = 23456)),
      ('bag', BaggingClassifier(n_estimators = 500,
                        #n_jobs = -1,
                        max_samples = .85,
                        oob_score = True,
                        random_state = 23456,
                        max_features = .7))
    ]
#%%
from sklearn.linear_model import LogisticRegression

clf = StackingClassifier(estimators, final_estimator = LogisticRegression(max_iter = 1000), cv = 5)
clf.fit(train.drop('id', axis = 1), train_labels.status_group)

cvs = cross_val_score(clf, train.drop('id', axis = 1), train_labels.status_group, scoring = 'accuracy', cv = 5
                      #, n_jobs = -1
                      )
print(cvs)
print('Mean Accuracy : {}'.format(np.mean(cvs).round(6)))


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
test['status_group'] = test.status_group.map(cat_recode)
test[['id', 'status_group']].to_csv("rf_xgb_gbc_bag_ensemble_logreg_final_opt_vars_addtl_rfs.csv", index = False)


