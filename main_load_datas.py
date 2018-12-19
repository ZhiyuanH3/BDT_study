import init_data         as id
import pandas            as pd
import numpy             as np
import math
from sklearn.externals   import joblib
from timeit              import default_timer    as timer
from collections         import OrderedDict

def LoadData_main(kwargs):
    p = kwargs
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Parameters are:")
    #for k,v in p.items():
    #    print("{0} = {1}".format(k,v))
    bkg_name_dict = {}
    sgn_name_dict = {}
    HTL1          = ['50to100' ,'100to200' ,'200to300'  ,'300to500']
    HTL2          = ['500to700','700to1000','1000to1500','1500to2000','2000toInf']
    HTL1          = HTL1[1:] # drop 50to100GeV sample 

    tail          = 'j_skimed.root'

    def root_name_bkg_gen(x): return p['qcdPrefix']+x+'_'+p['versionN_b']+'_'+str(p['num_of_jets'])+tail

    for stri in HTL1:    bkg_name_dict[stri]  = id.DataFile(root_name_bkg_gen(stri), p['path'])
    if p['selectedSample'] == 1:
        for stri in HTL2:    bkg_name_dict[stri]  = id.DataFile(root_name_bkg_gen(stri), p['path'])

    key_str_sgn                = 'MS'+str(p['sgn_mass'])+'ct'+str(p['sgn_ctauS'])
    root_name_sgn = p['sgnPrefix']+str(p['sgn_mass'])+'_ctauS-'+ str(p['sgn_ctauS'])+'_'+p['versionN_s']+'_'+str(p['num_of_jets'])+tail
    sgn_name_dict[key_str_sgn] = id.DataFile(root_name_sgn, p['path'])


    timeA = timer()
    id.SetCrossSection(bkg_name_dict, p['xs'])
    #df_bkg_dict, N_bkg_dict, N_available_bkg = id.LoadData(bkg_name_dict,setWeight=True ,thrsd=p['maxDataLoadCut'])
    #df_sgn_dict, N_sgn_dict, N_available_sgn = id.LoadData(sgn_name_dict,setWeight=False,thrsd=p['maxDataLoadCut'])
    df_bkg_dict, N_bkg_dict, N_available_bkg = id.LoadData(p,bkg_name_dict,setWeight=True ,thrsd=p['maxDataLoadCut'])
    df_sgn_dict, N_sgn_dict, N_available_sgn = id.LoadData(p,sgn_name_dict,setWeight=False,thrsd=p['maxDataLoadCut'])

    print '#available background: ',    N_available_bkg
    print '#available signal: '    ,    N_available_sgn
    
    for j in df_sgn_dict:    df_sig = df_sgn_dict[j]
    
    #df_bkg, df_sig, df_test_bkg, df_test_sig = id.SplitData(df_bkg_dict,df_sig,N_bkg_dict,bkg_name_dict,xs,bkg_test_multiple,bkg_multiple,N_available_bkg,N_available_sgn,train_test_ratio,random_seed,dataUseThrsd,weightL)
    
    #df_bkg, df_sig, df_test_bkg, df_test_sig = id.SplitDataNew(df_bkg_dict,df_sig,N_bkg_dict,bkg_name_dict,xs,bkg_test_multiple,bkg_multiple,N_available_bkg,N_available_sgn,train_test_ratio,random_seed,dataUseThrsd,weightL,N_sgn)
    
    df_bkg, df_sig, df_test_bkg, df_test_sig = id.SplitDataNew(df_bkg_dict,df_sig,N_bkg_dict,bkg_name_dict,p['xs'],p['N_bkg_to_test'],p['N_bkg_to_train'],N_available_bkg,N_available_sgn,p['train_test_ratio'],p['random_seed'],p['dataUseThrsd'],p['weightL'],p['test_mode'])
    
    print df_bkg[:8]
    print '>>>>>>>>>>>>>>>> Number of background used for training: ', len(df_bkg)
    print df_sig[:8]
    df       = pd.concat([df_sig, df_bkg], ignore_index=True)
    
    np.random.seed(p['random_seed'])
    #df      = utils.shuffle(df)
    df       = df.iloc[np.random.permutation(len(df))]
    #df      = df.iloc[np.random.RandomState(seed=random_seed).permutation(len(df))]
    df_train = df.copy()
    df_ts    = pd.concat([df_test_sig, df_test_bkg], ignore_index=True)
    
    np.random.seed(p['random_seed'])
    #df_test_orig = utils.shuffle(df)  
    df_test_orig  = df_ts.iloc[np.random.permutation(len(df_ts))]
    #df_test_orig = df_ts.iloc[np.random.RandomState(seed=random_seed).permutation(len(df_ts))]
    
    ColumnLabelDict, ColumnLabelDict_sk, JetPrfx = id.getColumnLabel(df_train)
    print ColumnLabelDict_sk
    ColumnLabelDict, ColumnLabelDict_sk, JetPrfx = id.getColumnLabel(df_test_orig)
    print ColumnLabelDict_sk

    pkls                   = {}
    pkls['df_train_o']     = df_train
    pkls['df_test_orig_o'] = df_test_orig
    
    pkls['df_train_orig']  = df_train
    pkls['df_test_orig']   = df_test_orig

    df_train       = np.asarray(df_train)
    df_test        = np.asarray(df_test_orig)
    isSigPos       = ColumnLabelDict[p['isSigL']]
    weightPos      = ColumnLabelDict[p['weightL']]

    timeB        = timer()

    out_dict = {
                 'ColumnLabelDict'   : ColumnLabelDict,
                 'ColumnLabelDict_sk': ColumnLabelDict_sk,
                 'JetPrfx'           : JetPrfx,
                 'JetPrfx'           : JetPrfx, 
                 'isSigPos'          : isSigPos,
                 'weightPos'         : weightPos,
               }
    
    #pkls                 = {}
    pkls['df_train']     = df_train
    pkls['df_test']      = df_test
    #pkls['df_test_orig'] = df_test_orig
    pkls['out_dict']     = out_dict
 

    """
    descrStr       = '_'.join(p['descr'])
    dump_pth = p['path']+'/'+p['loadedDatas_dir']+'/'+'preload_'+descrStr+'_forTrain'+str(p['bdtTrainOn'])+'.pkl'
    joblib.dump(pkls, dump_pth)
    timeC = timer()
    """
    print 'Time for loading datas from ROOT: ', str(timeB-timeA), 'sec'
    #print 'Time for dumping datas to pkl: '   , str(timeC-timeB), 'sec'

    return pkls



