import init_data         as id
import pandas            as pd
import numpy             as np
from sklearn.externals   import joblib
from timeit              import default_timer    as timer
from rocPlot             import plotROC_main
from main_load_datas     import LoadData_main
from main_bdt            import bdt_main, bdt_train, bdt_test


def mainF(kwargs):
    p = kwargs    
  
    """
    params = {}
    for param in default_params.keys():
        if param in kwargs.keys():
            cls = default_params[param].__class__
            value = cls(kwargs[param])
            params[param] = value
        else:
            params[param] = default_params[param]
    """
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Parameters are:")
    for k,v in p.items():
        print("{0} = {1}".format(k,v))
    

    if p['attributeKin'] == 1:
        p['attrAll'] += id.nKinListGen(p['num_of_jets'], p['kinList'])
    RefA    = 'J1pt'



    if p['load_from_root'] == 1:
        LoadData_main( p )
    if 1:
        timeA = timer()
        pkls  = joblib.load(p['path']+'/loadedDatas/'+'preloaded_data'+'_ctauS'+str(p['sgn_ctauS'])+'_M'+str(p['sgn_mass'])+'.pkl') 
        df_train     = pkls['df_train'] 
        df_test      = pkls['df_test'] 
        df_test_orig = pkls['df_test_orig'] 
        in_dict      = pkls['out_dict'] 
  

        timeB = timer()
        print 'Time taken for loading datas from pkl: ', str(timeB-timeA), 'sec'
        ColumnLabelDict    = in_dict['ColumnLabelDict']  
        ColumnLabelDict_sk = in_dict['ColumnLabelDict_sk']
        JetPrfx            = in_dict['JetPrfx'] 
        isSigPos           = in_dict['isSigPos']
        weightPos          = in_dict['weightPos']    
        JetPrfx            = in_dict['JetPrfx']         

        #--------------------------benchmark--------------------------------
        LCL = {
                'cHadEFrac'   :['<',0.2],
                'cMulti'      :['<',10],
                'nEmEFrac'    :['<',0.15],
                'nHadEFrac'   :['>',0.8],
                'photonEFrac' :['<',0.1],
              }

        HCL = {
                'cHadEFrac'   :['<',0.08],
                'cMulti'      :['<',8],
                'nEmEFrac'    :['<',0.08],
                'nHadEFrac'   :['>',0.9],
                'photonEFrac' :['<',0.08],
                'ecalE'       :['<',10]
              }

        
        cutBaseDict               = {}
        cutBaseDict['Loose Cut']  = id.CutBaseBenchmarkNew(df_test_orig, LCL, JetPrfx, RefA, p['isSigL'])
        cutBaseDict['Hard Cut']   = id.CutBaseBenchmarkNew(df_test_orig, HCL, JetPrfx, RefA, p['isSigL'])
        #--------------------------benchmark--------------------------------        



  
        allAttrList    = id.CountAttrInList(ColumnLabelDict_sk, p['attrAll'])
        #~~~~~~~~take in all the attributes info
        X_train, y_train, w_train = df_train[:,allAttrList], df_train[:,isSigPos], df_train[:,weightPos]
        X_test , y_test , w_test  = df_test[:,allAttrList] , df_test[:,isSigPos] , df_test[:,weightPos]
        if p['weight_on'] == 0:
            w_train = None
            w_test  = None
        #print w_train
        #print ColumnLabelDict
        #print ColumnLabelDict_sk
     
    if   p['bdtTrainOn'] == 1:
        bdt_train(ps=p, X_Train=X_train, y_Train=y_train, W_train=w_train, pklName=p['pkl_name'])
      
        output_dict = bdt_test(
                                X_Test       = X_test,\
                                y_Test       = y_test,\
                                W_test       = w_test,\
                                df_Test_orig = df_test_orig,\
                                pklName      = p['pkl_name'],\
                                isSig_L      = p['isSigL'],\
                                calcROC      = p['calcROCon']
                              )


    elif p['bdtTrainOn'] == 0:
        output_dict = bdt_test(                                  
			        X_Test       = X_test,\
			        y_Test       = y_test,\
			        W_test       = w_test,\
			        df_Test_orig = df_test_orig,\
			        pklName      = p['pkl_name'],\
			        isSig_L      = p['isSigL'],\
			        calcROC      = p['calcROCon'] 
			      )

        output_dict['cut_based'] = cutBaseDict        
    #output_dict = bdt_main(PS=params, X_Trains=X_train, X_Tests=X_test, y_Trains=y_train, y_Tests=y_test, W_trains=w_train, W_tests=w_test, df_Test_origs=df_test_orig, pklNames='save_models/'+'bdt.pkl', trainOn=bdtTrainOn, isSig_Ls=isSigL, calcROCs=calcROCon)


    #~~~~~~~~plot roc-curve
    #########################################################################################
    if p['plotOn']:
        pass
        """  
        out_name     = 'roc'
        cutBaseDict  = {
                         'Loose Cut': [sgn_eff    , fls_eff   ],
                         'Hard Cut' : [sgn_eff_HC , fls_eff_HC]
                       }

        pklDict      = {
                         'BDT'+'(descriptions)': {'aoc':auc_bdt, 'fpr':fpr_bdt, 'tpr':tpr_bdt}
                       }
        plotROC_main(path_out, out_name, cutBaseDict, pklDict)
        """ 
    #########################################################################################

    return output_dict
