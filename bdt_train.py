import os
import numpy                         as np
import multiprocessing               as mp
import argparse                      as agp
from   TrainC_new                import mainF
from   sklearn.externals         import joblib

pars = agp.ArgumentParser()
pars.add_argument('--kin'   ,action='store',type=str,help='kinematics'                )
pars.add_argument('--inputs',action='store',type=str,help='model input'               )
pars.add_argument('--train' ,action='store',type=str,help='train model'               )
pars.add_argument('--trnm'  ,action='store',type=str,help='train mass'                )
pars.add_argument('--tstm'  ,action='store',type=str,help='test mass'                 )
pars.add_argument('--trnl'  ,action='store',type=str,help='train life time'           )
pars.add_argument('--tstl'  ,action='store',type=str,help='test life time'            )
pars.add_argument('--attr1' ,action='store',type=str,help='attribute1'                )
pars.add_argument('--attr2' ,action='store',type=str,help='attribute2'                )
#pars.add_argument('--tst'   ,action='store',type=str,help='test when tst&trn are same')
args     = pars.parse_args()
kin      = args.kin
inputs   = args.inputs
train_on = args.train
trnm     = args.trnm
tstm     = args.tstm
trnl     = args.trnl
tstl     = args.tstl
attr1    = args.attr1
attr2    = args.attr2
#tst_mode = args.tst
print 'train: ', trnm, trnl
print 'test:  ', tstm, tstl
trnm           = int(trnm)
trnl           = int(trnl)
tstm           = int(tstm)
tstl           = int(tstl)
bdt_modelL     = inputs
version        = 0 
selectionOn    = 1
#multipl        = 8000#1500
###################################################################################################
if   bdt_modelL == 'full':
    attr_str = bdt_modelL  
    attrA    = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE','J1VBF_DisplacedJet40_VTightID_Hadronic_match']
elif bdt_modelL == '2best':
    attr_str = bdt_modelL
    attrA    = ['J1nHadEFrac','J1cHadMulti']#['J1cHadEFrac','J1cHadEFrac']#['J1cHadEFrac','J1cHadEFrac']#['J1nSelectedTracks','J1eleMulti']#'J1VBF_DisplacedJet40_VTightID_Hadronic_match']#'J1photonMulti'] elif bdt_modelL == 'find2b':
elif bdt_modelL == 'find2b':
    attr_str = attr1+'_'+attr2
    attrA    = [attr1,attr2]
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
#path_result    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/generalization_bdt/'
#path_result    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD/test/'
path_result    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG/'


p                     = {}
#p['train_test_ratio'] = 0.6#0.99#0.6
p['train_test_ratio'] = 0.6#0.8#0.6#0.99#0.6

#p['N_bkg_to_train']   = 2000000
#p['N_bkg_to_test']    = 11400000
"""
p['N_bkg_to_train']   = 3750
p['N_bkg_to_test']    = 21250
p['maxDataLoadCut']   = 888888888
"""
"""
p['N_bkg_to_train']   = 15203#3798
p['N_bkg_to_test']    = 10135#21533
p['maxDataLoadCut']   = 888888888
"""

#p['N_bkg_to_train']   = 20271#15203#3798
#p['N_bkg_to_test']    = 5068#10135#21533
p['N_bkg_to_train']   = 3798
p['N_bkg_to_test']    = 21533
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
#bee_pth   = '/'

if   selectionOn == 0:    path_data = bee_pth+'/forbdtnew/brianSample/'
#elif selectionOn == 1:    path_data = bee_pth+'/fromLisa_forBDT/with_triggerBool/withNSelectedTracks/'
elif selectionOn == 1:    path_data = bee_pth+'/fromBrian_forBDT/bkg_sgn/'

def setParams(**pp):
    keyList = pp.keys()
    if not 'maxDataLoadCut'    in keyList:    maxDataLoadCut    = 3000000 
    else                                 :    maxDataLoadCut    = pp['maxDataLoadCut'] 
    if not 'num_of_jets'       in keyList:    num_of_jets       = 1 
    else                                 :    num_of_jets       = pp['num_of_jets']
    #if not 'train_test_ratio'  in keyList:    train_test_ratio  = 0.5+0.1
    #else                                 :    train_test_ratio  = pp['train_test_ratio']
    #if not 'bkg_test_multiple' in keyList:    bkg_test_multiple = 8000
    #else                                 :    bkg_test_multiple = pp['bkg_test_multiple']
    #if not 'bkg_multiple'      in keyList:    bkg_multiple      = 8000
    #else                                 :    bkg_multiple      = pp['bkg_multiple']

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


result = {}

def train_test_model(bdt_train_on, load_root=1, testMode=1):
    global result
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
    result['data'] = mainF(vdpar)                  
    result['info'] = vdpar 


#preload_pth = path_data+'/loadedDatas/'+'preload_'+descrStr+'_forTrain'+str(train_on)+'.pkl'
#if os.path.isfile(preload_pth):    ld_root = 0
#else                          :    ld_root = 1
ld_root = 1 
  
if   int(train_on) == 1:    tst_mode     = 0 
elif int(train_on) == 0:    tst_mode     = 1 # Combine train_on = 0 and tst_mode = 0 --> tst&trn same test mode!!

train_test_model( int(train_on), load_root=ld_root, testMode=tst_mode ) 

##################################dumping############################################
pklN     = 'RS_'+descrStr+'.pkl'
pth_dump = path_result + pklN
joblib.dump(result, pth_dump)
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Results stored at: ' + pth_dump
##################################dumping############################################


























# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Old Lines:
"""
def lifeTime_compare(ctauL_l,load_root=1):
    global result
    for cti in ctauL_l:
        vdpar = setParams(
                           attrAll           = attrA                ,\
                           load_from_root    = load_root            ,\
                           bdtTrainOn        = 0                    ,\
                           ct                = cti                  ,\
                           mass              = M_test_fix           ,\
                           random_seed       = sd                   ,\
                           train_test_ratio  = p['train_test_ratio'],\
                           N_bkg_to_test     = p['N_bkg_to_test']   ,\
                           N_bkg_to_train    = p['N_bkg_to_train']  ,\
                           maxDataLoadCut    = p['maxDataLoadCut']  ,\
                           test_mode=1
                         )
        result['ctau'][cti] = mainF(vdpar)
"""


"""
subd_N_list           = []
subd_N_list.append(dsc[2])
subd_N_list.append(dsc[3])
subd_N_list.append(dsc[4])
subdir         = '_'.join(subd_N_list)
path_result    = pth_out + '/' + subdir + '/'
if not os.path.isdir(path_result):    os.system('mkdir '+path_result)
"""














"""
nCores = 80
P      = {}
LL     = []
aa     = 0
bb     = 0
length = int( len(massL) / nCores )
for i in xrange(nCores):
    bb = aa + length
    LL.append(massL[aa:bb])
for i in xrange(nCores):    P[i] = mp.Process(target= mass_compare, args=(,massLi))
for i in xrange(nCores):    P[i].start()
for i in xrange(nCores):    P[i].join()
"""

"""
#################################best 2 variables##################################################
if findBestTwo ==1:
    tempResult[str(sd)] = {}
    keyedL              = []
    goon                = 1
    for a1 in attrL:
        for a2 in attrL:
            for kk in keyedL:
                k1 = a1+','+a2 
                k2 = a2+','+a1
                if k1 == kk or k2 == kk:    goon = 0
                else                   :    goon = 1
            if goon == 1:
                attrTemp                                      = [a1,a2]
                pars                                          = setParams(attrTemp, mass=M, random_seed=sd)
                aoc, Fpr_bdt, Tpr_bdt, Thresholds_bdt, Es, Eb = mainF(pars)
                key                                           = a1+','+a2
                keyedL.append(key)        
                tempResult[str(sd)][key]                      = {}
                tempResult[str(sd)][key]['aoc']               = aoc
                tempResult[str(sd)][key]['params']            = pars
    for key in keyedL:
        result[key]           = {}
        result[key]['params'] = tempResult['4444'][key]['params']      
        aoc = np.zeros( len(seedL) )
        for sd in enumerate(seedL):
            aoc[sd[0]] = tempResult[str(sd[1])][key]['aoc']
        result[key]['aoc']        = np.average(aoc)
        result[key]['std_of_aoc'] = np.std(aoc)
#################################best 2 variables##################################################
"""





