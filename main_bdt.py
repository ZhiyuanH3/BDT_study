import pandas            as pd
import numpy             as np
import multiprocessing   as mp
from sklearn.ensemble    import AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier
from sklearn.externals   import joblib
from sklearn             import metrics
from timeit              import default_timer    as timer
 
import os
import sys
sys.path.append('/home/hezhiyua/desktop/PyrootLearn/ROC/performance_test/')
from roc_gen import ROC_GEN


def DecisionScores(model,X_tests,df_test_origin,p):
    #twoclass_output = model.decision_function(X_trains)
    ##################################################################calculating the roc curve independently
    all_probs       = model.predict_proba(X_tests)
    dftt            = df_test_origin.copy()
    class_names     = {0: "background",
		       1: "signal"}
    classes         = sorted(class_names.keys())
    for cls in classes:
        dftt[class_names[cls]] = all_probs[:,cls]
    sig   = dftt[p['isSigL']] == 1
    bkg   = dftt[p['isSigL']] == 0
    
    smallerDF_s = dftt[['signal','weight']][sig].copy()
    smallerDF_b = dftt[['signal','weight']][bkg].copy()
    
    poisson     = 0
    dump_on     = 1 
    if dump_on == 1:
        pth = p['path_result'] 
        path_dump = pth + '/dumps/'
        if not os.path.isdir(path_dump):
            os.system('mkdir '+path_dump)
        dsc_str = '_'.join(p['descr'][1:])
        path_dump = path_dump + '/' + dsc_str + '/'
        if not os.path.isdir(path_dump):
            os.system('mkdir '+path_dump)
         
        joblib.dump(smallerDF_s   ,path_dump+'/s.pkl'  )
        joblib.dump(smallerDF_b   ,path_dump+'/b.pkl'  )
   
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Bayesian Uncertainty:   
    ROC = ROC_GEN(smallerDF_s, smallerDF_b)

    if poisson == 1:
        probs = dftt["signal"][sig].values.copy()
        probb = dftt["signal"][bkg].values.copy()
        #sortedProbList = sorted( list(probs) + list(probb) )[::-1]
        #sortedProbList_series = pd.Series(sortedProbList)
        #range_u = len( sortedProbList_series.loc[sortedProbList_series>0.57] ) 
        #sortedProbList_small = sortedProbList[:range_u]
        probS   = smallerDF_s['signal'].copy()
        probB   = smallerDF_b['signal'].copy()
        weightS = smallerDF_s['weight'].copy()
        weightB = smallerDF_b['weight'].copy()
        es, eb  = [], []
        unc     = [] 
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Calculating ROC..'
        tA = timer()
        weighted_n_sgn = weightS.sum()
        weighted_n_bkg = weightB.sum()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Poisson Uncertainty:
        resolution = 1000
        resolution = 1/float(resolution)
        for c in np.arange(0,1,resolution): #sortedProbList_small:
            ifSgnLike_sig = probS > c
            ifSgnLike_bkg = probB > c
            if len( weightB[ifSgnLike_bkg] ) == 0:
                delta = 0
            else:
                df_temp             = smallerDF_b[ifSgnLike_bkg].copy()
                df_temp_weight      = df_temp['weight'].copy()
                #weight_counts      = df_temp.weight.value_counts()
                df_temp_ns_weighted = df_temp_weight.groupby(df_temp_weight).sum().apply(np.square).sum().copy()
                #n_weighted         = len(smallerDF_b[ifSgnLike_bkg]) * smallerDF_b['weight'][ifSgnLike_bkg]        
                delta               = np.sqrt( df_temp_ns_weighted ) 
            passed_s = (weightS[ifSgnLike_sig]).sum()
            passed_b = (weightB[ifSgnLike_bkg]).sum()
            es.append(   float(passed_s)/weighted_n_sgn   ) #true positive
            eb.append(   float(passed_b)/weighted_n_bkg   ) #false positive
            unc.append( delta )
        tB = timer()
        print 'Time taken to calculate ROC: ', str(tB-tA), 'sec'
        roc_dict        = {}
        roc_dict['unc'] = unc     
        roc_dict['tpr'] = es
        roc_dict['fpr'] = eb
        joblib.dump(roc_dict, path_dump+'/roc_poisson.pkl')
    return ROC





def bdt_train(ps,X_Train,y_Train,W_train):
    #~~~~~~~~Create and fit an AdaBoosted decision tree
    clf = DecisionTreeClassifier(  max_depth = ps['max_depth']  )
    bdt = AdaBoostClassifier(  clf,
			       algorithm     = ps['algorithm'],
			       n_estimators  = ps['n_estimators'],
			       learning_rate = ps['learning_rate'],
			    )
    print '>>>>>>>>>>>>>>>>>>>>>>>>> Fitting BDT...'
    ti = timer()
    bdt.fit( X_Train , y_Train , sample_weight = W_train )
    clf.fit( X_Train , y_Train , sample_weight = W_train )
    tf = timer()
    print '>>>>>>>>>>>>>>>>>>>>>> BDT fit completed~'
    print 'time taken for bdt fit: ' + str(tf-ti) + 'sec'
 
    pkl_path = ps['path_result'] + '/' + 'save_model' 
    if not os.path.isdir(pkl_path):
        os.system('mkdir '+pkl_path)
  
    dsc_str = ps['descr'][0] + '_' + '_'.join(ps['descr'][2:5])
    pkl_path = pkl_path + '/' + dsc_str + '/'
    if not os.path.isdir(pkl_path):
        os.system('mkdir '+pkl_path)

    joblib.dump(bdt, pkl_path+'/'+'bdt.pkl')
 

def bdt_test(X_Test,y_Test,W_test,df_Test_orig,ps):
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading BDT-Model...'
    dsc_str = ps['descr'][0] + '_' + '_'.join(ps['descr'][2:5])
    pkl_path = ps['path_result'] + '/' + 'save_model' + '/' + dsc_str 
    pkl_path = pkl_path+'/'+'bdt.pkl'
    bdt = joblib.load(pkl_path)
    if ps['calcROCon'] == 1:
       roc_dict = DecisionScores(bdt,X_Test,df_Test_orig,ps)
    y_pred_proba_bdt                 = bdt.predict_proba(X_Test)[:,1]
    fpr_bdt, tpr_bdt, thresholds_bdt = metrics.roc_curve(y_Test, y_pred_proba_bdt, sample_weight=W_test)
    auc_bdt                          = metrics.roc_auc_score(y_Test, y_pred_proba_bdt, sample_weight=W_test)
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AUC_bdt: ', auc_bdt
    out_dict                   = {}
    out_dict['aoc']            = auc_bdt
    out_dict['fpr']            = fpr_bdt
    out_dict['tpr']            = tpr_bdt
    out_dict['thresholds_bdt'] = thresholds_bdt
    out_dict['roc']            = roc_dict
    return out_dict


def bdt_main(PS,X_Trains,X_Tests,y_Trains,y_Tests,W_trains,W_tests,df_Test_origs,pklNames):
    if   PS['trainOn'] == 1:
        bdt_train(PS,X_Trains,y_Trains,W_trains,pklNames)
    elif PS['trainOn'] == 0:
        out_dict = bdt_test(X_Tests,y_Tests,W_tests,df_Test_origs,PS)
    return out_dict











if __name__ == '__main__':
    pass










