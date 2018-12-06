import os 
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
    #params = {}
    #for param in default_params.keys():
    #    if param in kwargs.keys():
    #        cls           = default_params[param].__class__
    #        value         = cls(kwargs[param])
    #        params[param] = value
    #    else:
    #        params[param] = default_params[param]  
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Parameters are:'
    for k,v in p.items():    print("{0} = {1}".format(k,v))
    
    RefA    = 'J1pt'

    if p['attributeKin']   == 1:    p['attrAll'] += id.nKinListGen(p['num_of_jets'], p['kinList'])
    if p['load_from_root'] == 1:    pkls          = LoadData_main( p )
    
    df_train     = pkls['df_train'] 
    df_test      = pkls['df_test'] 
    df_test_orig = pkls['df_test_orig'] 
    in_dict      = pkls['out_dict'] 
    """
    timeA        = timer()
    descrStr     = '_'.join(p['descr'])
    load_pth     = p['path']+'/'+p['loadedDatas_dir']+'/'+'preload_'+descrStr+'_forTrain'+str(p['bdtTrainOn'])+'.pkl'
    pkls         = joblib.load( load_pth ) 
    df_train     = pkls['df_train'] 
    df_test      = pkls['df_test'] 
    df_test_orig = pkls['df_test_orig'] 
    in_dict      = pkls['out_dict']   
    timeB        = timer()
    print 'Time for loading datas: ', str(timeB-timeA), 'sec'
    """

    ColumnLabelDict    = in_dict['ColumnLabelDict']  
    ColumnLabelDict_sk = in_dict['ColumnLabelDict_sk']
    JetPrfx            = in_dict['JetPrfx'] 
    isSigPos           = in_dict['isSigPos']
    weightPos          = in_dict['weightPos']    
    JetPrfx            = in_dict['JetPrfx']         
    #--------------------------Benchmark--------------------------------
    LCL                       = p['cut_base']['LC'] 
    HCL                       = p['cut_base']['HC']
    cutBaseDict               = {}
    cutBaseDict['loose_cut']  = id.CutBaseBenchmarkNew(df_test_orig, LCL, JetPrfx, RefA, p['isSigL'])
    cutBaseDict['hard_cut' ]  = id.CutBaseBenchmarkNew(df_test_orig, HCL, JetPrfx, RefA, p['isSigL'])
    #--------------------------Benchmark--------------------------------        
    allAttrList    = id.CountAttrInList(ColumnLabelDict_sk, p['attrAll'])
    # ~~~~~~~~Take in all the attributes info
    X_train, y_train, w_train = df_train[:,allAttrList], df_train[:,isSigPos], df_train[:,weightPos]
    X_test , y_test , w_test  = df_test[:,allAttrList] , df_test[:,isSigPos] , df_test[:,weightPos]
    if p['weight_on'] == 0:
        w_train = None
        w_test  = None
    #print w_train
    #print ColumnLabelDict
    #print ColumnLabelDict_sk
     
    if   p['bdtTrainOn'] == 1:    bdt_train(ps=p, X_Train=X_train, y_Train=y_train, W_train=w_train)
    elif p['bdtTrainOn'] == 0:    pass
    output_dict = bdt_test(                                  
		            X_Test       = X_test,\
		            y_Test       = y_test,\
		            W_test       = w_test,\
		            df_Test_orig = df_test_orig,\
                            ps           = p 
                          )
    output_dict['cut_based'] = cutBaseDict        

    #os.system('rm '+load_pth)
    return output_dict
