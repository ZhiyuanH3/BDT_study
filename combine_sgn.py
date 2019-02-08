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

    return df_sig, df_test_sig    


    """
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

    """
    descrStr       = '_'.join(p['descr'])
    dump_pth = p['path']+'/'+p['loadedDatas_dir']+'/'+'preload_'+descrStr+'_forTrain'+str(p['bdtTrainOn'])+'.pkl'
    joblib.dump(pkls, dump_pth)
    timeC = timer()
    """
    #print 'Time for loading datas from ROOT: ', str(timeB-timeA), 'sec'
    #print 'Time for dumping datas to pkl: '   , str(timeC-timeB), 'sec'

    #return pkls

















kin      = '1'
train_on = '1'
#"""
trnm     = 50
tstm     = 50
trnl     = 500
tstl     = 500
bdt_modelL     = '2best'
version        = 0
selectionOn    = 1
###################################################################################################
if   bdt_modelL == 'full':
    attr_str = bdt_modelL
    attrA    = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE','J1VBF_DisplacedJet40_VTightID_Hadronic_match']
elif bdt_modelL == '2best':
    attr_str = bdt_modelL
    attrA    = ['J1nSelectedTracks','J1eleMulti']#'J1VBF_DisplacedJet40_VTightID_Hadronic_match']#'J1photonMulti']
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attributes: ', attrA
###################################################################################################

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Do not change the order!
dsc            = []
dsc.append('trn_'   + str(trnm)        + 'GeV_' + str(trnl) + 'mm')
dsc.append('tst_'   + str(tstm)        + 'GeV_' + str(tstl) + 'mm')
dsc.append('slct'   + str(selectionOn)                            )
dsc.append('attr_'  + attr_str                                    )
dsc.append('kin'    + str(kin)                                    )
dsc.append('v'      + str(version)                                )
descrStr       = '_'.join(dsc)
path_result    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD/test/'


p                     = {}
p['train_test_ratio'] = 0.6
p['N_bkg_to_train']   = 3750
p['N_bkg_to_test']    = 21250
p['maxDataLoadCut']   = 888888888

CL       = {}
CL['LC'] = {
             'cHadEFrac'   :['<',0.2],
             'cMulti'      :['<',10],
             'nEmEFrac'    :['<',0.15],
             'nHadEFrac'   :['>',0.8],
             'photonEFrac' :['<',0.1],
           }
CL['HC'] = {
             'cHadEFrac'   :['<',0.08],
             'cMulti'      :['<',8],
             'nEmEFrac'    :['<',0.08],
             'nHadEFrac'   :['>',0.9],
             'photonEFrac' :['<',0.08],
             'ecalE'       :['<',10]
           }

rd_seed   = 4444
bee_pth   ='/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/'


path_data = bee_pth+'/fromBrian_forBDT/bkg_sgn/'

def setParams(**pp):
    keyList = pp.keys()
    if not 'maxDataLoadCut'    in keyList:    maxDataLoadCut    = 3000000
    else                                 :    maxDataLoadCut    = pp['maxDataLoadCut']
    if not 'num_of_jets'       in keyList:    num_of_jets       = 1
    else                                 :    num_of_jets       = pp['num_of_jets']


    params = {
               'load_from_root'                : pp['load_from_root'],
               'bdtTrainOn'                    : pp['bdtTrainOn'],
               'testOn'                        : 0,
               'calcROCon'                     : 1,
               #'lolaOn'                        : 0,
               #'fcnOn'                         : 0,
               'weight_on'                     : 1, #0: not weighting qcd
               #--------------------------------------------------------------Data
               #'N_sgn'                         : n_sgn,
               'descr'                         : dsc,
               'cut_base'                      : CL,
               'test_mode'                     : pp['test_mode'],
               'maxDataLoadCut'                : maxDataLoadCut,
               'dataUseThrsd'                  : 50000,
               'num_of_jets'                   : num_of_jets,
               'selectedSample'                : selectionOn,
               'sgn_mass'                      : pp['mass'],
               'sgn_ctauS'                     : pp['ct'],
               'qcdPrefix'                     : 'QCD_HT',
               'sgnPrefix'                     : 'VBFH_HToSSTobbbb_MH-125_MS-',
               'versionN_b'                    : 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
               #'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC',
               'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC',
               'train_test_ratio'              : pp['train_test_ratio'],
               'N_bkg_to_test'                 : pp['N_bkg_to_test'],
               'N_bkg_to_train'                : pp['N_bkg_to_train'],
               #'bkg_multiple'                  : bkg_multiple,#10,#15,#10,
               #'bkg_test_multiple'             : bkg_test_multiple,#50000,#multipl,#100,#multipl,#70,#100
               #--------------------------------------------------------------General
               'xs'                            : { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 },
               'kinList'                       : ['pt','mass','energy'],
               'attrAll'                       : pp['attrAll'],
               'attributeKin'                  : int(kin),
               #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Hyper_Parameters
               #-----------------------fcnn
               #hypr_model
               #'n_nodes'                       : 30,
               #hypr_fit
               #'validation_ratio'              : 0.2 / float(0.2 + 0.6), #0.3
               #'n_batch_size'                  : 512,#64
               #'n_epochs'                      : 20,
               #-----------------------fcnn
               #-----------------------------------BDT
               'max_depth'                     : 4,
               'algorithm'                     : 'SAMME',
               'n_estimators'                  : 140,
               'learning_rate'                 : 0.1,
               #-----------------------------------BDT
               #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~hyper_parameters
               #Fixing random state for reproducibility

               'random_seed'                   : pp['random_seed'],
               #-----------------------------------------------------Default_Strings
               'isSigL'                        : 'is_signal',
               'weightL'                       : 'weight',
               'path'                          : path_data,
               'path_result'                   : path_result,
               'loadedDatas_dir'               : 'loadedDatas',
              }

    return params




mass_list = [20,30,40,50]
ctau_list = [500,1000,2000,5000]

df_sgn_list      = []
df_sgn_test_list = []

ld_root       = 1
load_root     = ld_root
bdt_train_on  = int(train_on)


if   int(train_on) == 1:    tst_mode     = 0
elif int(train_on) == 0:    tst_mode     = 1 # Combine train_on = 0 and tst_mode = 0 --> tst&trn same test mode!!

testMode = tst_mode

def run_i(tstl,tstm):
    vdpar = setParams(
                       attrAll          = attrA                ,\
                       load_from_root   = load_root            ,\
                       bdtTrainOn       = bdt_train_on         ,\
                       ct               = tstl                 ,\
                       mass             = tstm                 ,\
                       random_seed      = rd_seed              ,\
                       train_test_ratio = p['train_test_ratio'],\
                       N_bkg_to_test    = p['N_bkg_to_test']   ,\
                       N_bkg_to_train   = p['N_bkg_to_train']  ,\
                       maxDataLoadCut   = p['maxDataLoadCut']  ,\
                       test_mode        = testMode
                     )
    return LoadData_main(vdpar)


for m_ii in mass_list:
    for l_ii in ctau_list:
        df_sgn_i, df_test_sgn_i = run_i(l_ii,m_ii)
        df_sgn_list.append(df_sgn_i)
    
df_sgn_stacked = pd.concat(df_sgn_list)
print df_sgn_stacked
print 'df_length: ', len(df_sgn_stacked)

