import pandas            as pd
import numpy             as np
import multiprocessing   as mp
from sklearn.ensemble    import AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier
from sklearn.externals   import joblib
from sklearn             import metrics
from timeit              import default_timer    as timer
import heapq
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
        for c in np.arange(0,1,resolution): 
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
        print 'Time to calculate ROC: ', str(tB-tA), 'sec'
        roc_dict        = {}
        roc_dict['unc'] = unc     
        roc_dict['tpr'] = es
        roc_dict['fpr'] = eb
        joblib.dump(roc_dict, path_dump+'/roc_poisson.pkl')
    return ROC





def bdt_train(ps,X_Train,y_Train,W_train):
    #~~~~~~~~Create and fit an AdaBoosted decision tree
    clf = DecisionTreeClassifier(  max_depth                = ps['max_depth'],\
                                   min_weight_fraction_leaf = ps['min_weight_fraction_leaf'],\
                                )


    bdt = AdaBoostClassifier(  
                              clf                                ,\
			      algorithm     = ps['algorithm']    ,\
			      n_estimators  = ps['n_estimators'] ,\
			      learning_rate = ps['learning_rate'],\
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
    #pkl_path = pkl_path + '/' + dsc_str + '/'
    #if not os.path.isdir(pkl_path):
    #    os.system('mkdir '+pkl_path)
    #if ps['dump']:
    if 1:
        joblib.dump(bdt, pkl_path+'/'+'bdt_'+dsc_str+'.pkl')
        joblib.dump(clf, pkl_path+'/'+'clf_'+dsc_str+'.pkl')
   

def bdt_test(X_Test,y_Test,W_test,df_Test_orig,ps):
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading BDT-Model...'
    dsc_str  = ps['descr'][0] + '_' + '_'.join(ps['descr'][2:5])
    pkl_pth  = ps['path_result'] + '/' + 'save_model' #+ '/' + dsc_str 
    pkl_path = pkl_pth+'/'+'bdt_'+dsc_str+'.pkl'
    bdt      = joblib.load(pkl_path)
    y_pred_proba_bdt           = bdt.predict_proba(X_Test)[:,1]
    auc_bdt                    = metrics.roc_auc_score(y_Test, y_pred_proba_bdt, sample_weight=W_test)
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AUC_bdt: ', auc_bdt
    out_dict                   = {}
    out_dict['aoc']            = auc_bdt
    if not ps['val_on']:
        if ps['calcROCon'] == 1:    roc_dict = DecisionScores(bdt,X_Test,df_Test_orig,ps)
        fpr_bdt, tpr_bdt, thresholds_bdt     = metrics.roc_curve(y_Test, y_pred_proba_bdt, sample_weight=W_test)
        out_dict['fpr']            = fpr_bdt
        out_dict['tpr']            = tpr_bdt
        out_dict['thresholds_bdt'] = thresholds_bdt
        out_dict['roc']            = roc_dict


    if 0:
        clf_pth  = pkl_pth+'/'+'clf_'+dsc_str+'.pkl'
        clf      = joblib.load(clf_pth)
        from sklearn.externals.six import StringIO
        from sklearn.tree import export_graphviz
        import pydotplus
        from sklearn import tree
    
        features = ["nst","nst"]
        classes  = ["background","signal"]
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        feature_names=features,  
                        class_names=classes,  
                        filled=True, rounded=True,  
                        special_characters=True) 
    
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("/home/hezhiyua/classifier_tree.pdf")

    

    return out_dict


def bdt_main(PS,X_Trains,X_Tests,y_Trains,y_Tests,W_trains,W_tests,df_Test_origs):
    if   PS['trainOn'] == 1:    bdt_train(PS,X_Trains,y_Trains,W_trains)
    elif PS['trainOn'] == 0:    out_dict = bdt_test(X_Tests,y_Tests,W_tests,df_Test_origs,PS)
    return out_dict





def bdt_train_score(X_Train,y_Train,W_train,df_Train,ps):
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading BDT-Model...'
    dsc_str  = ps['descr'][0] + '_' + '_'.join(ps['descr'][2:5])
    pkl_path = ps['path_result'] + '/' + 'save_model' #+ '/' + dsc_str 
    pkl_path = pkl_path+'/'+'bdt_'+dsc_str+'.pkl'
    bdt      = joblib.load(pkl_path)
    if ps['calcROCon'] == 1:    roc_dict = DecisionScores(bdt,X_Train,df_Train,ps)
    y_pred_proba_bdt                     = bdt.predict_proba(X_Train)[:,1]
    fpr_bdt, tpr_bdt, thresholds_bdt     = metrics.roc_curve(y_Train, y_pred_proba_bdt, sample_weight=W_train)
    auc_bdt                              = metrics.roc_auc_score(y_Train, y_pred_proba_bdt, sample_weight=W_train)
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training score -- AUC_bdt: ', auc_bdt
    out_dict                   = {}
    out_dict['aoc']            = auc_bdt
    out_dict['fpr']            = fpr_bdt
    out_dict['tpr']            = tpr_bdt
    out_dict['thresholds_bdt'] = thresholds_bdt
    out_dict['roc']            = roc_dict   

    return out_dict



def bdt_val(X_Train,y_Train,W_train,df_Train,X_Test,y_Test,W_test,df_Test_orig,ps):
    PS           = ps
    PS['val_on'] = 1

    par_str      = 'min_weight_fraction_leaf'
    min_par_val  = 0.
    max_par_val  = 0.5
    n_per_dpth   = 4#5#4#7

    steps        = 0
    goon         = 1  
    depth        = 0
    max_depth    = 4#5#4#3
   
    A            = min_par_val
    a            = A
    B            = max_par_val
    b            = B
    C            = (B-A)/2.    
    A_score      = 0.5
    B_score      = 0.5
    C_score      = 0.5

    # Approch: Zoom In
    while goon:
        par_list        = []
        test_score_list = []
        incrmt          = (b - a)/float(n_per_dpth+1)
        for i in np.arange(a+incrmt, b, incrmt):
            PS[par_str] = i
            bdt_train(PS,X_Train,y_Train,W_train)
            steps      += 1
            test_dict   = bdt_test(X_Test,y_Test,W_test,df_Test_orig,ps)
            test_score  = test_dict['aoc']
            test_score_list.append(test_score)
            par_list.append(i)
        par_list.append(A)
        par_list.append(B)
        par_list.append(C)
        test_score_list.append(A_score)
        test_score_list.append(B_score)   
        test_score_list.append(C_score)

        best_test_scores = heapq.nlargest(3, test_score_list) 
        test_score_best0 = best_test_scores[0]
        test_score_best1 = best_test_scores[1]
        test_score_best2 = best_test_scores[2]
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> Best Scores:', best_test_scores
        best0            = par_list[ test_score_list.index(test_score_best0) ]
        best1            = par_list[ test_score_list.index(test_score_best1) ]
        best2            = par_list[ test_score_list.index(test_score_best2) ]        
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> Best Param.:', best0
        if test_score_best0 > C_score: 
            C                = best0
            C_score          = test_score_best0

        if ((best0 > best1) and (best0 < best2)) or ((best0 < best1) and (best0 > best2)):
            if   best1 == best2: break
            elif best1 > best2 :
                A = best2
                a = A
                B = best1
                b = B
                A_score = test_score_best2
                B_score = test_score_best1      
            else:
                B = best2
                b = B
                A = best1
                a = A
                B_score = test_score_best2
                A_score = test_score_best1
        else:  
            if   best1 == best0: break
            elif best1 > best0 :
                A = best0 
                a = A - incrmt
                B = best1
                b = B
                A_score = test_score_best0
                B_score = test_score_best1   
            else:
                B = best0
                b = B + incrmt
                A = best1
                a = A
                B_score = test_score_best0
                A_score = test_score_best1
        depth += 1
        if depth == max_depth: goon = 0 


    # Approch: Random Search
    """
    if 1:
        n_epoch         = 20
        n_bins          = 100
        par_list        = []
        test_score_list = []
        incrmt          = (B - A)/float(n_bins+1)
        for i in range(n_epoch):
            PS[par_str] = np.random.choice( np.arange(A+incrmt, B, incrmt) )
            bdt_train(PS,X_Train,y_Train,W_train)
            steps      += 1
            test_dict   = bdt_test(X_Test,y_Test,W_test,df_Test_orig,ps)
            test_score  = test_dict['aoc']
            test_score_list.append(test_score)
            par_list.append(i)
        par_list.append(A)
        par_list.append(B)
        test_score_list.append(A_score)
        test_score_list.append(B_score)   
        best_test_scores = heapq.nlargest(2, test_score_list)
        test_score_best1 = best_test_scores[0]
        test_score_best2 = best_test_scores[1]
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> 2 Best:', best_test_scores
        best1            = par_list[ test_score_list.index(test_score_best1) ]
        best2            = par_list[ test_score_list.index(test_score_best2) ]
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> Best Param.:', best1
    """

    # Approch: Grid Search
    """
    if 1:
        n_per_dpth   = 20
        par_list        = []
        test_score_list = []
        incrmt          = (B - A)/float(n_per_dpth+1)
        for i in np.arange(A+incrmt, B, incrmt):
            PS[par_str] = i
            bdt_train(PS,X_Train,y_Train,W_train)
            steps      += 1
            test_dict   = bdt_test(X_Test,y_Test,W_test,df_Test_orig,ps)
            test_score  = test_dict['aoc']
            test_score_list.append(test_score)
            par_list.append(i)
        par_list.append(A)
        par_list.append(B)
        test_score_list.append(A_score)
        test_score_list.append(B_score)   
        best_test_scores = heapq.nlargest(2, test_score_list)
        test_score_best1 = best_test_scores[0]
        test_score_best2 = best_test_scores[1]
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> 2 Best:', best_test_scores
        best1            = par_list[ test_score_list.index(test_score_best1) ]
        best2            = par_list[ test_score_list.index(test_score_best2) ]
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> Best Param.:', best1
    """


    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Steps: ', steps
    PS['val_on']  = 0
    PS[par_str]   = best0
    bdt_train(PS,X_Train,y_Train,W_train)
  










if __name__ == '__main__':
    pass










