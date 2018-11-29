import numpy  as     np
from   TrainC import mainF

path_result = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
#path_result = '/nfs/dust/cms/user/hezhiyua/LLP/outputs/bdt/result/'
selectionOn = 1#
multipl     = 200#1500#1000#500#200#2000#1000#100
attrKin     = 0#pt,mass,energy__on
allAttr     = 1
version     = 9#7#6#2    1:testing
plot        = 0#1
nJ          = 1#3 
M           = 40
attrL  = ['cHadEFrac'] 
#attrPL  = ['J1cHadE','J1cHadEFrac','J1cHadMulti','J1nHadEFrac','J1cMulti','J1DisplacedJetsTriggerBool']
attrPL  = ['J1cHadEFrac','J1nHadEFrac','J1cMulti','J1DisplacedJetsTriggerBool']

#attrL  = ['J1cHadEFrac'] 
#attrPL = ['J1cHadEFrac', 'J1nHadEFrac', 'J1cMulti', 'J1DisplacedJetsTriggerBool', 'J2cHadEFrac', 'J2nHadEFrac', 'J2cMulti', 'J2DisplacedJetsTriggerBool', 'J3cHadEFrac', 'J3nHadEFrac', 'J3cMulti', 'J3DisplacedJetsTriggerBool']
#attrPL = ['J1cHadEFrac', 'J1nHadEFrac', 'J1cMulti', 'J1DisplacedJetsTriggerBool']
#attrPL = ['J2cHadEFrac', 'J2nHadEFrac', 'J2cMulti', 'J2DisplacedJetsTriggerBool', 'J3cHadEFrac', 'J3nHadEFrac', 'J3cMulti', 'J3DisplacedJetsTriggerBool']
#attrPL = ['J2cHadEFrac', 'J2nHadEFrac', 'J2cMulti', 'J2DisplacedJetsTriggerBool']
#attrPL = ['J3cHadEFrac', 'J3nHadEFrac', 'J3cMulti', 'J3DisplacedJetsTriggerBool']



lrL = [0.1]
neL = [140]
#lrL = [0.001,0.01,0.1,0.2,0.4,0.6,0.8]
#neL = [10,20,40,60,80,100,120,140,160,180,200]



massL  = [15,20,25,30,35,40,45,50,55,60]
seedL  = [4444]#[1111,2222,3333,4444]



if   selectionOn == 0:
    #path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/forBDT/with_energy/'
    path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/forbdtnew/brianSample/'
    selectedSamples = 0
    versionStr = str(version)+'_noSelection'
elif selectionOn == 1:
    #path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromLisa_forBDT/withNonLeadingJets/'
    #path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromLisa_forBDT/with_triggerBool/withNonFrac/'
    path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromLisa_forBDT/with_triggerBool/withNSelectedTracks/'
    selectedSamples = 1
    versionStr = str(version)+'_withSelection'

def setParams(lr,ne,mass,load,seed):
    params = {   
                               'loadOn'                        : load,
                               #'testOn'                        : 0,
                               'plotOn'                        : plot,
                               #'lolaOn'                        : 0,
                               #'fcnOn'                         : 0,
    
                               'attr_all'                      : allAttr,#0, #0: taking only 2 attributes
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
                               'versionN_b'                    : 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
                               'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC',
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
                               'n_nodes'                       : 30,
                               #hypr_fit
                               #'validation_ratio'              : 0.2 / float(0.2 + 0.6), #0.3
                               'n_batch_size'                  : 512,#64
                               'n_epochs'                      : 20,
                               #-----------------------fcnn
                               #-----------------------------------bdt
                               'maximum_depth_all_attr'        : 8,#4,
                               'number_of_estimators_all_attr' : ne,#140,
                               'rate_of_learning_all_attr'     : lr,#0.1,
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


tempResult = {}
result     = {}

sd    = seedL[0] 
vdpar = setParams(lrL[0],neL[0],mass=M,load=1,seed=sd) 
df_train, df_test, df_test_orig, ColumnLabelDict, ColumnLabelDict_sk, JetPrfx, sgn_eff, fls_eff, sgn_eff_HC, fls_eff_HC, isSigPos, weightPos = mainF(vdpar)
    
for lri in lrL:
    result[lri] = {}
    for nei in neL:
        result[lri][nei] = {}  
        pars                               = setParams(lri,nei,mass=M,load=0,seed=sd)
        aoc                                = mainF(pars)
        result[lri][nei]['aoc']   = aoc
        result[lri][nei]['params'] = pars  


import pickle
if   attrKin == 0:    pklN = 'result_v'                     + versionStr + '.pickle'
elif attrKin == 1:    pklN = 'result_with_pt_mass_energy_v' + versionStr + '.pickle'
f1 = open( path_result + pklN , 'wb' )
pickle.dump(result,f1)
f1.close()
print 'results stored at: ' + path_result + pklN



