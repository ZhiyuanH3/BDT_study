import pandas            as pd
import numpy             as np
import multiprocessing   as mp
from sklearn.ensemble    import AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier
from sklearn.externals   import joblib
from sklearn             import metrics
from timeit              import default_timer    as timer

import sys
sys.path.append('/home/hezhiyua/desktop/PyrootLearn/ROC/performance_test/')
from roc_gen import ROC_GEN



ES, EB      = [], []
#------------------for parallelization----------------------------------------------------------------------------------
"""
#turnning local objects to globals
SmallerDF_s = pd.DataFrame()
SmallerDF_b = pd.DataFrame()

ES_u, EB_u  = [], []
ES_l, EB_l  = [], []

weighted_n_sgn = None
weighted_n_bkg = None

def calcROC(c):     
    global ES, EB, SmallerDF_s, SmallerDF_b 
    ifSgnLike_sig = SmallerDF_s['signal'] > c
    ifSgnLike_bkg = SmallerDF_b['signal'] > c
    ES.append(   (float((SmallerDF_s['weight'][ifSgnLike_sig]).sum())/SmallerDF_s['weight'].sum())   ) #true positive
    EB.append(   (float((SmallerDF_b['weight'][ifSgnLike_bkg]).sum())/SmallerDF_b['weight'].sum())   ) #false positive

def calcROC_new(c):
    global ES, EB, SmallerDF_s, SmallerDF_b, weighted_n_sgn, weighted_n_bkg, ES_u, EB_u, ES_l, EB_l
    ifSgnLike_sig = SmallerDF_s['signal'] > c
    ifSgnLike_bkg = SmallerDF_b['signal'] > c
    if len(SmallerDF_b['weight'][ifSgnLike_bkg]) == 0:
        delta = 0 
    else:
        delta           = np.sqrt(  len(SmallerDF_b[ifSgnLike_bkg])  ) * np.average( SmallerDF_b['weight'][ifSgnLike_bkg] )
    ES.append(   float((SmallerDF_s['weight'][ifSgnLike_sig]).sum())/weighted_n_sgn   ) #true positive
    EB.append(   float((SmallerDF_b['weight'][ifSgnLike_bkg]).sum())/weighted_n_bkg   ) #false positive

    EB_u.append(   float( (SmallerDF_b['weight'][ifSgnLike_bkg]).sum() + delta )/weighted_n_bkg   ) #false positive
    EB_l.append(   float( (SmallerDF_b['weight'][ifSgnLike_bkg]).sum() - delta )/weighted_n_bkg   ) #false positive
"""
#------------------for parallelization----------------------------------------------------------------------------------


def DecisionScores(model,X_tests,df_test_origin,isSigLs,calcROCon=0):
    #global ES, EB, SmallerDF_s, SmallerDF_b, weighted_n_sgn, weighted_n_bkg, ES_u, EB_u, ES_l, EB_l
    #twoclass_output = model.decision_function(X_trains)
    ##################################################################calculating the roc curve independently
    all_probs       = model.predict_proba(X_tests)
    dftt            = df_test_origin.copy()
    class_names     = {0: "background",
		       1: "signal"}
    classes         = sorted(class_names.keys())
    for cls in classes:
        dftt[class_names[cls]] = all_probs[:,cls]
    sig   = dftt[isSigLs] == 1
    bkg   = dftt[isSigLs] == 0
    

    smallerDF_s = dftt[['signal','weight']][sig].copy()
    smallerDF_b = dftt[['signal','weight']][bkg].copy()
    
    poisson     = 0
    dump_on     = 1 
    if dump_on == 1:
        #pth = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
        pth = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/bdt_overview/'
        joblib.dump(smallerDF_s   ,pth+'/dumps/s.pkl'  )
        joblib.dump(smallerDF_b   ,pth+'/dumps/b.pkl'  )
        #joblib.dump(sortedProbList,pth+'/dumps/spl.pkl')  

   
    # Bayesian Uncertainty:   
    ROC = ROC_GEN(smallerDF_s, smallerDF_b)

    # Change the outpust path!!
    if poisson == 1:
        probs = dftt["signal"][sig].values.copy()
        probb = dftt["signal"][bkg].values.copy()
        #sortedProbList = sorted( list(probs) + list(probb) )[::-1]
        #sortedProbList_series = pd.Series(sortedProbList)
        #range_u = len( sortedProbList_series.loc[sortedProbList_series>0.57] ) #0.55
        #sortedProbList_small = sortedProbList[:range_u]
        #print 'sortedProbList-length: ', str( len(sortedProbList) ) 
        #print 'sortedProbList_small-length: ', str( len(sortedProbList_small) )
        probS   = smallerDF_s['signal'].copy()
        probB   = smallerDF_b['signal'].copy()
        weightS = smallerDF_s['weight'].copy()
        weightB = smallerDF_b['weight'].copy()
        es, eb  = [], []
        unc     = [] 
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Calculating ROC..'
        tA = timer()
        """
        for c in sortedProbList: 
            ifSgnLike_sig = probS > c
            ifSgnLike_bkg = probB > c
            es.append(   (float((smallerDF_s['weight'][ifSgnLike_sig]).sum())/weightS.sum())   ) #true positive
            eb.append(   (float((smallerDF_b['weight'][ifSgnLike_bkg]).sum())/weightB.sum())   ) #faule positive 
        """ 
        #SmallerDF_s = smallerDF_s.copy() 
        #SmallerDF_b = smallerDF_b.copy()    
    
        weighted_n_sgn = weightS.sum()
        weighted_n_bkg = weightB.sum()
      
        #pool = mp.Pool(processes=80)
        #pool.map(calcROC_new, sortedProbList_small) #sortedProbList)
        #sortedProbList_small = sortedProbList_small[:20]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Poisson Uncertainty:
        resolution = 1000
        resolution = 1/float(resolution)
        for c in np.arange(0,1,resolution):#sortedProbList_small:
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
        #pool.map(calcROC, sortedProbList_small)
        #pool.map(calcROC, sortedProbList)
        ##################################################################calculating the roc curve independently
        tB = timer()
        print 'Time taken to calculate ROC: ', str(tB-tA), 'sec'
        
        #pth = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
        #path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/'
        roc_dict = {}
        roc_dict['unc'] = unc     
        roc_dict['tpr'] = es
        roc_dict['fpr'] = eb
        joblib.dump(roc_dict, pth+'/dumps/roc_poisson.pkl')
    return ROC





def bdt_train(ps,X_Train,y_Train,W_train,pklName):
    if 1:
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
        joblib.dump(bdt,pklName)
 

def bdt_test(X_Test,y_Test,W_test,df_Test_orig,pklName,isSig_L='is_signal',calcROC=0):
    global ES, EB     
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading BDT-Model...'
    bdt = joblib.load(pklName)  
    if calcROC == 1:
       roc_dict = DecisionScores(bdt,X_Test,df_Test_orig,isSig_L,calcROC)
    y_pred_proba_bdt                 = bdt.predict_proba(X_Test)[:,1]
    fpr_bdt, tpr_bdt, thresholds_bdt = metrics.roc_curve(y_Test, y_pred_proba_bdt, sample_weight=W_test)
    auc_bdt                          = metrics.roc_auc_score(y_Test, y_pred_proba_bdt, sample_weight=W_test)
    #auc_bdt = metrics.auc(es, eb)
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AUC_bdt: ', auc_bdt
    out_dict                   = {}
    out_dict['aoc']            = auc_bdt
    out_dict['fpr']            = fpr_bdt
    out_dict['tpr']            = tpr_bdt
    out_dict['thresholds_bdt'] = thresholds_bdt
    out_dict['roc']            = roc_dict
    #out_dict['es'] = ES
    #out_dict['eb'] = EB
    return out_dict




def bdt_main(PS,X_Trains,X_Tests,y_Trains,y_Tests,W_trains,W_tests,df_Test_origs,pklNames,trainOn=1,isSig_Ls='is_signal',calcROCs=0):
    global ES, EB
    if   trainOn == 1:
        bdt_train(PS,X_Trains,y_Trains,W_trains,pklNames)
    elif trainOn == 0:
        out_dict = bdt_test(X_Tests,y_Tests,W_tests,df_Test_origs,pklNames,isSig_L=isSig_Ls,calcROC=calcROCs)
    #if trainOn == 0:
    return out_dict








if __name__ == '__main__':
    pass
