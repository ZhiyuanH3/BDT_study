#from sklearn.grid_search import RandomizedSearchCV
from   TrainC import mainF
import init_data         as id
import pandas            as pd
import numpy             as np
from sklearn.ensemble    import AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier
from sklearn             import metrics
from timeit              import default_timer    as timer 
import pickle        

num_of_random_epochs = 200
path_result          = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
multipl              = 200#1500
attrKin              = 1 # pt,mass,energy__on
version              = 11#10#9#7#6#2    1:testing
param_distributions  = {
                         'max_depth'         : range(1,300, 1),
                         'min_samples_split' : range(2,200,1),
                         'min_samples_leaf'  : range(1,100,1),   
                         'algorithm'     : ['SAMME','SAMME.R'],
                         'n_estimators'  : range(10,400,1),
                         'learning_rate' : [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                       }
attrPL = ['J1cHadEFrac','J1nHadEFrac','J1cMulti','J1DisplacedJetsTriggerBool']
plot                 = 0#1
nJ                   = 1#3 
M                    = 40
selectionOn          = 1#
#massL  = [15,20,25,30,35,40,45,50,55,60]
seedL  = [4444]

if   selectionOn == 0:
    path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/forbdtnew/brianSample/'
    selectedSamples = 0
    versionStr = str(version)+'_noSelection'
elif selectionOn == 1:
    path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromLisa_forBDT/with_triggerBool/withNSelectedTracks/'
    selectedSamples = 1
    versionStr = str(version)+'_withSelection'

def setParams(mass,load,seed):
    params = {
	       'loadOn'                        : load,
	       #'testOn'                        : 0,
	       'plotOn'                        : plot,
	       #'lolaOn'                        : 0,
	       #'fcnOn'                         : 0,

	       'attr_all'                      : 1,#0, #0: taking only 2 attributes
	       #'weight_on'                     : 1, #0: not weighting qcd
	       #data
	       'maxDataLoadCut'                : 1000000,
	       'dataUseThrsd'                  : 50000,
	       'num_of_jets'                   : nJ,
	       'selectedSample'                : selectedSamples,
	       'sgn_mass'                      : mass,#40,
	       #'sgn_ctauS'                     : 500, 
	       'outName'                       : 'roc_bdt',#'roc_fcnn_vs_bdt',
	       #'qcdPrefix'                     : 'QCD_HT', 
	       #'sgnPrefix'                     : 'VBFH_HToSSTobbbb_MH-125_MS-',        
	       #'versionN_b'                    : 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
	       #'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC',
	       #'train_test_ratio'              : 0.6 + 0.2,
	       'bkg_multiple'                  : multipl,#10,#15,#10,
	       'bkg_test_multiple'             : 100,#multipl,#70,#100
	       #general
	       #'xs'                            : { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 },
	       #'kinList'                       : ['pt','mass','energy'],
	       #'roc_resolution'                : 0.0001,
	       'attrAll'                       : attrPL,#attrL,
	       #'attrTwo'                       : ['',''],

	       'bmA1'                          : 'J1cHadEFrac',#'cHadEFrac',
	       'bmA2'                          : 'J1nHadEFrac',#'nHadEFrac',
	       'bmA1_th'                       : 0.38,#0.38,
	       'bmA2_th'                       : 1,#1000000,#0.38,
	       'attribute1'                    : 'J1cHadEFrac',
	       'attribute2'                    : 'J1nHadEFrac',
	       'attributeKin'                  : attrKin,#1,
	       #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~hyper_parameters
	       #-----------------------fcnn
	       #hypr_model
	       #'n_nodes'                       : 30,
	       #hypr_fit
	       #'validation_ratio'              : 0.2 / float(0.2 + 0.6), #0.3
	       #'n_batch_size'                  : 512,#64
	       #'n_epochs'                      : 20,
	       #-----------------------fcnn
	       #-----------------------------------bdt
	       #'maximum_depth_all_attr'        : 8,#4,
	       #'number_of_estimators_all_attr' : 140,
	       #'rate_of_learning_all_attr'     : 0.1,
	       #-----------------------------------bdt
	       #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~hyper_parameters
	       #Fixing random state for reproducibility
	       'random_seed' : seed,#4444,
	       #default strings
	       #'isSigL'  : 'is_signal',
	       #'weightL' : 'weight',
	       'path'      : path_data,
	       'path_out'  : '/beegfs/desy/user/hezhiyua/LLP/bdt_output/roc/',
	       #'path_lola' : '',
             }

    return params

#######################Load Datas################################################################################
sd                       = seedL[0]
vdpar                    = setParams(mass=M,load=1,seed=sd)
output_dict              = mainF(vdpar)
if 1:
    attrAll                   =  attrPL
    allAttrList               = id.CountAttrInList(ColumnLabelDict_sk,attrAll)
    #~~~~~~~~take in all the attributes info
    X_train, y_train, w_train = df_train[:,allAttrList], df_train[:,isSigPos], df_train[:,weightPos]
    X_test , y_test , w_test  = df_test[:,allAttrList] , df_test[:,isSigPos] , df_test[:,weightPos]
#######################Load Datas################################################################################


#||||||||||||||||||||||bdtRS|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def bdtRS(prm):
    clf = DecisionTreeClassifier(
                                  max_depth         = prm['max_depth']        ,
                                  min_samples_split = prm['min_samples_split'],
                                  min_samples_leaf  = prm['min_samples_leaf'],                              
                                )

    bdt = AdaBoostClassifier(
                              clf,
                              algorithm     = prm['algorithm']    ,#"SAMME",
                              n_estimators  = prm['n_estimators'] ,
                              learning_rate = prm['learning_rate'],
                            )
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Fitting bdt...'
    ti = timer()
    bdt.fit( X_train , y_train , sample_weight = w_train )
    clf.fit( X_train , y_train , sample_weight = w_train )
    tf = timer()
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BDT fit completed~'
    print 'time taken for 1 point: ' + str(tf-ti) + 'sec'
    y_pred_proba_bdt = bdt.predict_proba(X_test)[:,1]
    #fpr_bdt, tpr_bdt, thresholds_bdt = metrics.roc_curve(y_test, y_pred_proba_bdt, sample_weight=w_test)
    auc_bdt = metrics.roc_auc_score(y_test, y_pred_proba_bdt, sample_weight=w_test)
    print 'AUC_bdt: ', auc_bdt
    return auc_bdt
#||||||||||||||||||||||bdtRS||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#####################################################generate random parameter list
randomParamList = []
for i in range(num_of_random_epochs):
    prmTemp = {}
    for prmi, Li in param_distributions.iteritems():
        prmTemp[prmi] = np.random.choice(Li)
    randomParamList.append(prmTemp)
#####################################################generate random parameter list

#=====================================================run search
result     = {}
for i in range( len(randomParamList) ):
    result[i]           = {}
    result[i]['aoc']    = bdtRS( randomParamList[i] )
    result[i]['params'] = randomParamList[i] 
#=====================================================run search

###################################################################################Saving Results
pklN = 'result_kin'+str(attrKin)+'_v'+versionStr+'.pickle'
with open( path_result + pklN , 'wb' ) as f1:    pickle.dump(result,f1)
print '>>>>>>>>>>>>>>>>>>>>>> Results stored at: ' + path_result + pklN
###################################################################################Saving Results














################old lines:
"""  
param_distributions_bdt = {#'algorithm'    :  "SAMME",
                           'n_estimators'  : range(10,200,10),
                           'learning_rate' : np.linspace(0.0001,1,0.0001)
                          }


        #RandomizedSearchCV##################################################no weighting!!
        clf = DecisionTreeClassifier()
        bdt = AdaBoostClassifier(clf)
        rs  = RandomizedSearchCV(clf,#bdt,
                                 param_distributions,
                                 cv=4,
                                 n_iter=8,
                                 scoring='roc_auc',
                                 n_jobs=1,
                                 verbose=2)
        rs.fit(X_train, y_train) 
        best_estimator = rs.best_estimator_
        best_params    = rs.best_params_ 
        #print best_estimator
        print best_params
        print rs.best_score_
        #predict = rs.predict_proba(X_test)
        #print predict[:, 1]
        rs_bdt  = RandomizedSearchCV(bdt,
                                     param_distributions_bdt,
                                     cv=4,
                                     n_iter=8,
                                     scoring='roc_auc',
                                     n_jobs=1,
                                     verbose=2)
        rs_bdt.fit(X_train, y_train)
        best_estimator_bdt = rs_bdt.best_estimator_
        best_params_bdt    = rs_bdt.best_params_
        #print best_estimator_bdt
        print best_params_bdt
        print rs_bdt.best_score_
        #RandomizedSearchCV##################################################no weighting!!   
"""
                                                                           
