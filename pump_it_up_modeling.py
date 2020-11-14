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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


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


#%% clean up data

dta = pd.concat([train, test])

features = ['id', 
            'funder', 'gps_height',
               'installer', 'longitude', 'latitude', 'date_recorded',
               'basin',  'region_code', 'district_code', 'lga',
               'ward', 'population', 
               'scheme_management', 
               'management', 
               'permit', 
               'construction_year',
               'extraction_type_group', 
               'payment',
               'water_quality', 'quantity',
               'source_type','waterpoint_type']

dta = dta[features]


dta = dta.applymap(lambda col:col.lower() if type(col) == str else col)

dta[['permit']] = pd.to_numeric(dta['permit'])

dta.loc[dta['construction_year'] == 0, 'construction_year'] = np.nan
dta.loc[dta['funder'] == '0', 'funder'] = np.nan
dta.loc[dta['gps_height'] == '0', 'gps_height'] = np.nan
dta.loc[dta['population'] == '0', 'population'] = np.nan

# clean up non-word characters
numeric_cols = dta.select_dtypes(include = np.number)

text_cols = dta.select_dtypes('object')
text_cols = text_cols.apply(lambda col: col.str.replace('\\W+', '_'), axis = 1)

dta = pd.concat([numeric_cols, text_cols], axis = 1, ignore_index = False)

dta['msg_construction_year'] = dta.construction_year.isnull().astype(int)
dta['msg_population'] = dta.population.isnull().astype(int)


#%% separate data again
train = dta.loc[dta.id.isin(train.id)]
test = dta.loc[dta.id.isin(test.id)]

#%% categorical encoding and imputation

targ_enc = ce.TargetEncoder(cols = None)
targ_enc.fit(train, train_labels['status_group'])
train = targ_enc.transform(train)


imp = IterativeImputer(max_iter=10, max_value = 2013)
imp.fit(train)

train = pd.DataFrame(imp.transform(train), columns = train.columns)
train['construction_year'] = train['construction_year'].round(0)
train['permit'] = train['permit'].round(0)


#%% rf

rf = RandomForestClassifier(n_estimators = 250,
                            max_depth = 20, 
                            random_state = 23456)

cvs = cross_val_score(rf, train.drop('id', axis = 1), train_labels.status_group, scoring = 'accuracy', cv = 5, n_jobs = -1)
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

#%% feature importance
rf.fit(train.drop(['id'], axis = 1), train_labels.status_group)

feature_importances = pd.DataFrame(rf.feature_importances_,
                                    index = train.drop('id', axis = 1).columns,
                                    columns=['importance']).sort_values('importance',   
                                                                        ascending=False)
                                                                        
print(feature_importances)

#%% boosting

gbc = GradientBoostingClassifier(init = RandomForestClassifier(n_estimators=200,
                                                               max_samples = .85, 
                                                               max_features = 9, 
                                                               max_depth = 20, 
                                                               min_samples_split = 10,
                                                               random_state = 45678),
                                  subsample = .85,
                                  n_iter_no_change = 10,
                                  n_estimators=(1000),
                                  learning_rate=(.25),
                                  random_state=(23456))

cvs = cross_val_score(gbc, train.drop('id', axis = 1), train_labels.status_group, scoring = 'accuracy', cv = 5, n_jobs = -1)

print(cvs)
print('Mean Accuracy : {}'.format(np.mean(cvs).round(6)))
# [0.81254209 0.80858586 0.80867003 0.80589226 0.80858586] # md 10
# Mean Accuracy : 0.808855
# [0.81464646 0.80968013 0.81464646 0.80782828 0.80681818] # md 10 and subsample .85
# Mean Accuracy : 0.810724
# [0.81203704 0.80698653 0.80909091 0.80521886 0.80606061] # md 10, subsample .85 and 500 estimators: actually worse
# Mean Accuracy : 0.807879


#%% stack attempt 
# optimized params for xgb

xgb_params_upd = {'objective': 'multi:softmax',
                  'random_state': 23456,
                  'gamma': 0.001369048068388758,
                  'eta': 0.11,
                  'max_depth': 15,
                  'min_child_weight': 9,
                  'num_class' : 4,
                  'subsample' : .85,
                  'colsample_bytree' : .9}

 
# best model to date
estimators = [
    ('rf1', RandomForestClassifier(max_depth = 20, random_state = 23456)),
    ('rf2', RandomForestClassifier(max_depth = 20, random_state = 45678)),
    ('rf3', RandomForestClassifier(max_depth = 20, random_state = 56789)),
    ('rf4', RandomForestClassifier(max_depth = 20, random_state = 12345)),
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

clf = StackingClassifier(estimators, final_estimator = LogisticRegression(max_iter = 1000), cv = 10)

cvs = cross_val_score(clf, 
                      train.drop('id', axis = 1), 
                      train_labels.status_group, 
                      scoring = 'accuracy', 
                      cv = 5
                      )
print(cvs)
print('Mean Accuracy : {}'.format(np.mean(cvs).round(6)))
#Mean Accuracy : 0.818687
if np.mean(cvs).round(6) > 0.817946:
    clf.fit(train.drop('id', axis = 1), train_labels.status_group)
    print('model fitted')

 #%% generate submission

test = targ_enc.transform(test)
test = pd.DataFrame(imp.transform(test), columns = test.columns)
test['construction_year'] = test['construction_year'].round(0)
test['permit'] = test['permit'].round(0)
rf.fit(train.drop('id', axis = 1), train_labels.status_group)

# test['status_group'] = rf.predict(test.drop('id', axis = 1))
# test['status_group'] = test.status_group.map(cat_recode)
# test[['id', 'status_group']].to_csv("rf_with_min_samples_split.csv", index = False)
# # scored .8228 with pop


test['status_group'] = clf.predict(test.drop('id', axis = 1))
test['status_group'] = test.status_group.map(cat_recode)
test[['id', 'status_group']].to_csv("best_model_more_cv_dropped_low_imp_basic_rf.csv", index = False)


