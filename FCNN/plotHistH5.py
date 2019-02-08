import pandas as pd
import numpy  as np
import sys
sys.path.append('/home/hezhiyua/desktop/BDT_study/')
from main_bdt         import bdt_train, bdt_val, bdt_test

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree     import DecisionTreeClassifier

from sklearn          import metrics
from keras.callbacks  import EarlyStopping, ModelCheckpoint, Callback
from keras.models     import Sequential
from keras.layers     import Dense
from keras            import utils

#====settings==================================================================================================
#pth          = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/pfc_400/large_sgn/with_chf/output/train/'
phase_point = '20_500'#'40_2000'#'40_500'
#pth         = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/pfc_400/large_sgn/with_chf/h5s/output/test/'+phase_point+'/'              
pth = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forBDT/h5s/output/test/'+phase_point+'/'

pth_out         = pth + '/' + 'test_out/test_early_stop_weights/'
pth_checkpoint  = pth_out + "/weights-latest.hdf5"

nan_replacement = 0 

#-------------------------------------- FCN
train_str    = 'train'#'val'#'train'#'test'
val_str      = 'val'#'test'#'val'
test_str     = 'test'#'train'#'test'#'val'#'test'

es_patience  = 8#4#8#5
n_nodes      = 4#2#8#4#30
in_shape     = (2,)#(1,)#(2,)
#Optimizer    = 'rmsprop'
#loss_f       = 'mean_squared_error'
Optimizer    = 'adam'
loss_f       = 'categorical_crossentropy'

n_batch_size = 32#64    #16-1024#512#128
n_epochs     = 44#22#2#88

#-------------------------------------- BDT
pss = {
      'calcROCon': 0,
      'path_result': './out/',
      'descr': 'testt',

      'max_depth': 4,
      'algorithm': 'SAMME',
      'n_estimators': 140,
      'learning_rate': 0.1,
      }
#====settings==================================================================================================

tvt_l     = ['train','val','test']

data_dict = {}
for i in tvt_l:    data_dict[i] = pth+'vbf_qcd-'+i+'-v0_40cs.h5'

tb_dict   = {}
for key, item in data_dict.iteritems():
    store           = pd.HDFStore(item)
    tb              = store.select('table')
    #tb.dropna() # did not solve the problem...
    #tb.fillna(nan_replacement, inplace=True) # solved the problem...

    if key == 'train':    tb = tb.iloc[:,:] # 1000/3750

    tb_dict[key]    = tb
    print tb
    #tb.isnull().any()
    print 'total events:' 
    print tb.shape[0]
    print 'num of signals:'
    print len(tb[ tb['is_signal_new'] == 1 ])


"""
tvt_str = 'train'
for i in range(len(tb_dict[tvt_str])):    print tb_dict[tvt_str]['CHF'][i]
"""
arr_dict = {}
for key, item in tb_dict.iteritems():    arr_dict[key] = np.asarray(item)

X_dict = {}
y_dict = {}
w_dict = {}
in_tpl = (0,1)#[0]#[1]#(0,1)
for key, item in arr_dict.iteritems():    X_dict[key], w_dict[key], y_dict[key] = item[:,in_tpl], item[:,2], item[:,3]

#for key, item in arr_dict.iteritems():
#    w_dict[key] = np.ones( X_dict[key].shape )



#\\\\\\\\\\\\\\\\\\\\\\\fcnn\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
Y_dict = {}
for key, item in y_dict.iteritems():
    Y_dict[key] = utils.to_categorical(item)
    #print Y_dict[key]

model = Sequential()
model.add(Dense(n_nodes, activation='relu', input_shape=in_shape))
model.add(Dense(n_nodes, activation='relu'))
model.add(Dense(n_nodes, activation='relu'))
model.add(Dense(2      , activation='softmax'))
model.summary()

model.compile(
    optimizer = Optimizer,
    loss      = loss_f,
    #metrics   =['accuracy']
)

#################################################
early_stop = EarlyStopping( monitor='val_loss',
		            patience=es_patience,
		            verbose=0,
		            mode='auto')
checkpoint = ModelCheckpoint(pth_checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#################################################

history = model.fit(

    X_dict[train_str],
    Y_dict[train_str],
    batch_size      = n_batch_size, 
    epochs          = n_epochs,
    verbose         = 2,     
    validation_data = (X_dict[val_str], Y_dict[val_str], w_dict[val_str]), 
    shuffle         = True,
    sample_weight   = w_dict[train_str],
    #initial_epoch   = 0,
    callbacks       = [checkpoint, early_stop]

)
print history

y_pred_nn = model.predict_proba(X_dict[test_str])
auc_nn    = metrics.roc_auc_score(Y_dict[test_str], y_pred_nn, sample_weight=w_dict[test_str])
print 'AUC(fcn):'
print auc_nn
#\\\\\\\\\\\\\\\\\\\\\\\fcnn\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\






##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BDT:
for key, item in X_dict.iteritems():    X_dict[key] = item.reshape(-1,1)

bdt_val(
                 X_Train      = X_dict[train_str],\
                 y_Train      = y_dict[train_str],\
                 W_train      = w_dict[train_str],\
                 df_Train     = tb_dict[train_str],\
                 X_Test       = X_dict[test_str],\
                 y_Test       = y_dict[test_str],\
                 W_test       = w_dict[test_str],\
                 df_Test_orig = tb_dict[test_str],\
                 ps           = pss
       )































exit()
#||||||||||||||||||||||bdt||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
clf = DecisionTreeClassifier( 
                              max_depth                = pss['max_depth'], 
                              min_weight_fraction_leaf = pss['min_weight_fraction_leaf'],
                            )
bdt = AdaBoostClassifier( 
                          clf,
                          algorithm     = pss['algorithm'],
                          n_estimators  = pss['n_estimators'],
                          learning_rate = pss['learning_rate'],
                        )
print '>>>>>>>>>>>>>>> fitting...'
bdt.fit(X_dict[train_str], y_dict[train_str], sample_weight=w_dict[train_str])
clf.fit(X_dict[train_str], y_dict[train_str], sample_weight=w_dict[train_str])
print '<<<<<<<<<<<<<<< fit completed'
y_pred_proba_bdt           = bdt.predict_proba(X_dict[test_str])[:,1]
auc_bdt                    = metrics.roc_auc_score(y_dict[test_str], y_pred_proba_bdt, sample_weight=w_dict[test_str])
print 'AUC(BDT): ', auc_bdt
#||||||||||||||||||||||bdt||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||



















