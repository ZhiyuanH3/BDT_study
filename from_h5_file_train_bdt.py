#import init_data as id

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevent the plot from showing
import matplotlib.pyplot as plt
#import sys, optparse
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import metrics
from timeit import default_timer as timer
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import utils as U
import h5py


path0 = '/beegfs/desy/user/hezhiyua/pfc_test/raw_data/Skim/h/'
infname_train = 'vbf_qcd-train-v0_40cs.h5'
infname_test  = 'vbf_qcd-test-v0_40cs.h5'
infname_val   = 'vbf_qcd-val-v0_40cs.h5'
inffn_train = path0 + infname_train 
inffn_test  = path0 + infname_test
inffn_val   = path0 + infname_val

store_train = pd.HDFStore(inffn_train)
store_test  = pd.HDFStore(inffn_test)
store_val   = pd.HDFStore(inffn_val)

nr_train = store_train.get_storer('table').nrows
nr_test  = store_test.get_storer('table').nrows
nr_val   = store_val.get_storer('table').nrows
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#brs = ['E_0','C_0']
def datagen_h5(brs,store):
    size = store.get_storer('table').nrows
    foo = store.select('table',
                       columns = brs,
                       start   = 0,
                       stop    = size
                      )
    return foo



def colListGen(n,attr):
    l = []
    stra = attr + '_'
    for i in range(n):
        strt = stra + str(i)
        l.append(strt)
    return l

print colListGen(40,'E')
print colListGen(40,'C')

def chfGen(store_test):
    df_test    = store_test.select('table')
    

    #print df_test['E_0'][2]
    #for j in range(1000):
        #print df_test['E_0']   
        #print df_test['E_0'][j]


    #~~~~~~~~drop events of nan 
    #fname        = df_test.columns.values[0]
    #print fname
    #dropnan      = df_test[fname] !=                
    #df_test[w]   = df_test[:][dropnan]
    #~~~~~~~~drop events with values = -1


    df_test_Es = pd.DataFrame( df_test , df_test.index , colListGen(40,'E') )
    df_test_Cs = pd.DataFrame( df_test , df_test.index , colListGen(40,'C') )

    df_test_ECs = pd.DataFrame( df_test_Es.values * df_test_Cs.values,
                                index = df_test.index,
                                columns = colListGen(40,'EC')
                              )

    df_test_E_tot  = pd.DataFrame( df_test_Es.iloc[:,:].sum(axis=1)  , index = df_test.index , columns = ['E_tot'] )
    df_test_EC_tot = pd.DataFrame( df_test_ECs.iloc[:,:].sum(axis=1) , index = df_test.index , columns = ['EC_tot'] )
    df_test_E_tot_c = df_test_E_tot.replace(0,np.finfo(np.float32).eps)
    #df_test_E_tot_c = df_test_E_tot.replace(0.0,np.finfo(np.float32).eps)
    df_test_chf = pd.DataFrame( df_test_EC_tot.values / df_test_E_tot_c.values , index = df_test.index , columns = ['chf'] )
    
    df_test_w = pd.DataFrame( df_test , df_test.index , ['weight'] )
    df_test_l = pd.DataFrame( df_test , df_test.index , ['is_signal_new'] )

    return df_test_chf, df_test_w, df_test_l



df_chf_train, df_w_train, df_l_train = chfGen(store_train)
df_chf_test , df_w_test , df_l_test  = chfGen(store_test)
df_chf_val  , df_w_val  , df_l_val   = chfGen(store_val)


#print df_l_test
#print df_l_test.reset_index()
"""
df_test_orig = pd.concat([df_chf_test, df_w_test, df_l_test], ignore_index=True)
#print df_test_orig
df_val_orig = pd.concat([df_chf_val, df_w_val, df_l_val], ignore_index=True)
#print df_val_orig
df_train_orig = pd.concat([df_chf_train, df_w_train, df_l_train], ignore_index=True)
#print df_train_orig
"""

df_chf_test = df_chf_test.reset_index(drop=True)
df_w_test   = df_w_test.reset_index(drop=True)
df_l_test   = df_l_test.reset_index(drop=True)

df_test_orig = pd.DataFrame()
df_test_orig['chf'] = df_chf_test['chf']
df_test_orig['w']   = df_w_test['weight']
df_test_orig['l']   = df_l_test['is_signal_new']
print df_test_orig

df_chf_train = df_chf_train.reset_index(drop=True)
df_w_train   = df_w_train.reset_index(drop=True)
df_l_train   = df_l_train.reset_index(drop=True)

df_train_orig = pd.DataFrame()
df_train_orig['chf'] = df_chf_train['chf']
df_train_orig['w']   = df_w_train['weight']
df_train_orig['l']   = df_l_train['is_signal_new']
print df_train_orig

df_chf_val = df_chf_val.reset_index(drop=True)
df_w_val   = df_w_val.reset_index(drop=True)
df_l_val   = df_l_val.reset_index(drop=True)

df_val_orig = pd.DataFrame()
df_val_orig['chf'] = df_chf_val['chf']
df_val_orig['w']   = df_w_val['weight']
df_val_orig['l']   = df_l_val['is_signal_new']
print df_val_orig




print '~~~~~~~~~~~~~~~'
X_train = np.asarray(df_chf_train)
X_test  = np.asarray(df_chf_test)
X_val   = np.asarray(df_chf_val)
y_train = np.asarray(df_l_train)
y_test  = np.asarray(df_l_test)
y_val   = np.asarray(df_l_val)
w_train = np.asarray(df_w_train)
w_test  = np.asarray(df_w_test)
w_val   = np.asarray(df_w_val)

#print y_train[:222]

#"""
X_train = np.reshape(X_train,nr_train)
X_test  = np.reshape(X_test,nr_test)
X_val   = np.reshape(X_val,nr_val)
y_train = np.reshape(y_train,nr_train)
y_test  = np.reshape(y_test,nr_test)
y_val   = np.reshape(y_val,nr_val)
w_train = np.reshape(w_train,nr_train)
w_test  = np.reshape(w_test,nr_test)
w_val   = np.reshape(w_val,nr_val)
#"""
print '___________________________'



store_train.close()
store_test.close()
store_val.close()

















#====settings==================================================================================================
#test
attr_all  = 0 #0: taking only 2 attributes
weight_on = 1 #0: not weighting qcd
#data
train_test_ratio  = 0.6 + 0.2
bkg_multiple      = 10
bkg_test_multiple = 70#100
#general
roc_resolution = 0.0001
attrTwo = ['chf','chf']
xs   = { '50To100': 246300000 , '100To200': 28060000 , '200To300': 1710000 , '300To500': 347500 , 'sgn': 3.782 }
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~hyper_parameters
#-----------------------fcnn
#hypr_model
n_nodes = 30#30
#hypr_fit
validation_ratio = 0.2 / float(0.2 + 0.6) #0.3
n_batch_size     = 64
n_epochs         = 20#20
#-----------------------fcnn
#-----------------------------------bdt
maximum_depth_all_attr        = 4
number_of_estimators_all_attr = 140
rate_of_learning_all_attr     = 0.1
#-----------------------------------bdt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~hyper_parameters
#Fixing random state for reproducibility
random_seed = 4444
np.random.seed(random_seed)
#default strings
isSigL  = 'is_signal_new'
isSigL  = 'l'
weightL = 'weight'
#====settings==================================================================================================


print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


#\\\\\\\\\\\\\\\\\\\\\\\fcnn\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
Y_train = U.to_categorical(y_train) #1D-array to one-hot-encoding
Y_val   = U.to_categorical(y_val) 
#opt = optimizers.SGD(lr=0.01)
input_num = 1
print '#################################'
model = Sequential([
    Dense(n_nodes, activation='relu', input_shape=(input_num,)),
    Dense(n_nodes, activation='relu'),
    Dense(n_nodes, activation='relu'),
    Dense(2      , activation='softmax')
])
print '$$$$$$$$$$$$$$$$$$$$$$$$$$$'
model.compile(
    optimizer = 'rmsprop',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
model.summary()
print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
history = model.fit(
    X_train,                      #data
    Y_train,                      #labels
    batch_size = n_batch_size, 
    epochs = n_epochs, verbose = 2,
    validation_data = (X_val,Y_val,w_val),
    shuffle = True,
    sample_weight = w_train,
    initial_epoch = 0
)
#print history
y_pred_nn = model.predict_proba(X_test)
Y_test = U.to_categorical(y_test)   #1D-array to one-hot-encoding

##test#acc#function####################################################################      
##test#acc#function####################################################################

auc_nn = metrics.roc_auc_score(Y_test, y_pred_nn, sample_weight=w_test)
print 'AUC1:'
print auc_nn
y_pred_proba_nn = model.predict_proba(X_test)[:,1]
fpr_nn, tpr_nn, thresholds_nn = metrics.roc_curve(y_test, y_pred_proba_nn,sample_weight=w_test)

################for comparison bdt vs fcnn#####################################
#"""
all_probs_nn       = model.predict_proba(X_test)
#print all_probs_nn
dftt_nn            = df_test_orig.copy()
class_names     = {0: "background",
                   1: "signal"}
classes         = sorted(class_names.keys())
for cls in classes:
    #print all_probs_nn[:,cls]
    dftt_nn[class_names[cls]] = all_probs_nn[:,cls]
sig_nn = dftt_nn[isSigL] == 1
bkg_nn = dftt_nn[isSigL] == 0

probs_nn = dftt_nn["signal"][sig_nn].values
probb_nn = dftt_nn["signal"][bkg_nn].values

es_nn, eb_nn  = [], []
for c in np.arange(-1,1,roc_resolution):
    es_nn.append((float((probs_nn > c).sum())/probs_nn.size))
    eb_nn.append((float((probb_nn > c).sum())/probb_nn.size))
#"""
auc_nn = metrics.auc(es_nn, eb_nn)
################for comparison bdt vs fcnn#####################################
#\\\\\\\\\\\\\\\\\\\\\\\fcnn\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\






#||||||||||||||||||||||bdt||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#~~~~~~~~Create and fit an AdaBoosted decision tree
clf = DecisionTreeClassifier(max_depth = maximum_depth_all_attr)
bdt = AdaBoostClassifier(clf,
                         algorithm     = "SAMME",
                         n_estimators  = number_of_estimators_all_attr,
                         learning_rate = rate_of_learning_all_attr)
print 'fitting bdt...'
ti = timer()
bdt.fit( X_train , y_train , sample_weight = w_train )
clf.fit( X_train , y_train , sample_weight = w_train )
tf = timer()
print 'bdt fit completed>>>>>>>>>>>>>>>>>>>>>>>'
print 'time taken for bdt fit1: ' + str(tf-ti) + 'sec'
#joblib.dump(bdt,'bdt.pkl')
#bdt = joblib.load('bdt.pkl') 

#~~~~~~~~calculate the decision scores
#twoclass_output = bdt.decision_function(X_train)
all_probs       = bdt.predict_proba(X_test)
dftt            = df_test_orig.copy()
class_names     = {0: "background",
                   1: "signal"}
classes         = sorted(class_names.keys())
for cls in classes:
    dftt[class_names[cls]] = all_probs[:,cls]
sig = dftt[isSigL] == 1
bkg = dftt[isSigL] == 0

probs = dftt["signal"][sig].values
probb = dftt["signal"][bkg].values

es, eb  = [], []
for c in np.arange(-1,1,roc_resolution):
    es.append((float((probs > c).sum())/probs.size))
    eb.append((float((probb > c).sum())/probb.size))

y_pred_proba_bdt = bdt.predict_proba(X_test)[:,1]
fpr_bdt, tpr_bdt, thresholds_bdt = metrics.roc_curve(y_test, y_pred_proba_bdt, sample_weight=w_test)
auc_bdt = metrics.roc_auc_score(y_test, y_pred_proba_bdt, sample_weight=w_test)
#auc_bdt = metrics.auc(es, eb)
print 'AUC_bdt:'
print auc_bdt
