import os
import numpy                         as np
from   TrainC_new      import mainF
import pickle
import multiprocessing               as mp
import argparse                      as agp

pars = agp.ArgumentParser()
pars.add_argument('--kin'   ,action='store',type=int,help='kinematics'     )
pars.add_argument('--inputs',action='store',type=str,help='model input'    )
pars.add_argument('--train' ,action='store',type=int,help='train model'    )
pars.add_argument('--trnm'  ,action='store',type=int,help='train mass'     )
pars.add_argument('--tstm'  ,action='store',type=int,help='test mass'      )
pars.add_argument('--trnl'  ,action='store',type=int,help='train life time')
pars.add_argument('--tstl'  ,action='store',type=int,help='test life time' )
pars.add_argument('--attr'  ,action='store',type=str,help='attribute'      )

args   = pars.parse_args()
kin    = args.kin
inputs = args.inputs
train  = args.train
trnm   = args.trnm
tstm   = args.tstm
trnl   = args.trnl
tstl   = args.tstl
attr   = args.attr
print 'train: ', trnm
print 'test: ' , tstm

ctauSL = []
massL  = []
if tstm  :    massL.append(int(tstm))
#else     :    massL.append(50)
if tstl  :    ctauSL.append(int(tstl))
else     :    ctauSL.append(500)

if inputs:    bdt_modelL = inputs
else     :    bdt_modelL = '2best'#'full'
if kin   :    attrKin    = kin
else     :    attrKin    = 0 # pt,mass,energy__on

if train :    train_on   = train
else     :    train_on   = 0

version        = 0 # 0:testing
selectionOn    = 1
#findBestTwo    = 0
#multipl        = 8000#1500
if trnm  :    M_train    = int(trnm)
#else     :    M_train    = 40
if trnl  :    CT_train   = int(trnl)
else     :    CT_train   = 500
M_test_fix     = M_train       
CT_test_fix    = CT_train

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Do not change the order!
dsc = []
dsc.append('trn_'     + str(M_train)    + 'GeV_' + str(CT_train)    + 'mm')
dsc.append('tst_'     + str(massL[0])   + 'GeV_' + str(CT_test_fix) + 'mm')
dsc.append('Selected' + str(selectionOn)                                  )
dsc.append(bdt_modelL                                                     )
dsc.append('kin'      + str(attrKin)                                      )
dsc.append('v'        + str(version)                                      )
#dsc.append(attr compare_ctau compare_masses)
descrStr       = '_'.join(dsc) 

pth_out        = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/bdt_overview/'
###################################################################################################
if   bdt_modelL == 'full':
    attrA = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE']
elif bdt_modelL == '2best':
    attrA = ['J1nSelectedTracks','J1photonMulti']#['J1cHadEFrac']#['J1nSelectedTracks']
if attr:
    attrA.append(attr)
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attributes: ', attrA
###################################################################################################

subd_N_list    = []
subd_N_list.append(dsc[2])
subd_N_list.append(dsc[3])
subd_N_list.append(dsc[4])
#subd_N_list.append()
subdir         = '_'.join(subd_N_list)
path_result    = pth_out + '/' + subdir + '/'
if not os.path.isdir(path_result):    os.system('mkdir '+path_result)
#n_sgn                = 397#448
p                     = {}
p['train_test_ratio'] = 0.6
p['N_bkg_to_train']   = 2000000
p['N_bkg_to_test']    = 11400000
p['maxDataLoadCut']   = 888888888

CL        = {}
CL['LCL'] = {
              'cHadEFrac'   :['<',0.2],
              'cMulti'      :['<',10],
              'nEmEFrac'    :['<',0.15],
              'nHadEFrac'   :['>',0.8],
              'photonEFrac' :['<',0.1],
            }
CL['HCL'] = {
              'cHadEFrac'   :['<',0.08],
              'cMulti'      :['<',8],
              'nEmEFrac'    :['<',0.08],
              'nHadEFrac'   :['>',0.9],
              'photonEFrac' :['<',0.08],
              'ecalE'       :['<',10]
            }

seedL     = [4444]
bee_pth   ='/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/'
if   selectionOn == 0:    path_data = bee_pth+'/forbdtnew/brianSample/'
elif selectionOn == 1:    path_data = bee_pth+'/fromLisa_forBDT/with_triggerBool/withNSelectedTracks/'


def setParams(**pp):
    keyList = pp.keys()
    if not 'maxDataLoadCut'    in keyList:    maxDataLoadCut    = 3000000 
    else                                 :    maxDataLoadCut    = pp['maxDataLoadCut'] 
    if not 'train_test_ratio'  in keyList:    train_test_ratio  = 0.5+0.1
    else                                 :    train_test_ratio  = pp['train_test_ratio']
    if not 'bkg_test_multiple' in keyList:    bkg_test_multiple = 8000
    else                                 :    bkg_test_multiple = pp['bkg_test_multiple']
    if not 'bkg_multiple'      in keyList:    bkg_multiple      = 8000
    else                                 :    bkg_multiple      = pp['bkg_multiple']
    if not 'num_of_jets'       in keyList:    num_of_jets       = 1 
    else                                 :    num_of_jets       = pp['num_of_jets']

    params = {   
	       'load_from_root'                : pp['load_from_root'],
	       'bdtTrainOn'                    : pp['bdtTrainOn'],
	       'testOn'                        : 0,
	       'plotOn'                        : 0,
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
	       'out_roc_name'                  : 'roc_bdt',
	       'qcdPrefix'                     : 'QCD_HT', 
	       'sgnPrefix'                     : 'VBFH_HToSSTobbbb_MH-125_MS-',        
	       'versionN_b'                    : 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
	       'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC',
	       'train_test_ratio'              : train_test_ratio,#0.6+0.2,
	       'N_bkg_to_test'                 : pp['N_bkg_to_test'], 
	       'N_bkg_to_train'                : pp['N_bkg_to_train'],
	       #'bkg_multiple'                  : bkg_multiple,#10,#15,#10,
	       #'bkg_test_multiple'             : bkg_test_multiple,#50000,#multipl,#100,#multipl,#70,#100
	       #--------------------------------------------------------------General
	       'xs'                            : { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 },
	       'kinList'                       : ['pt','mass','energy'],
	       'attrAll'                       : pp['attrAll'],
	       'attributeKin'                  : attrKin,
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
	       #'path_roc'                      : '/beegfs/desy/user/hezhiyua/LLP/bdt_output/roc/',
	       'path_result'                   : path_result,
	       #'path_lola'                     : '',
	      }

    return params


tempResult       = {}
result           = {}
result['masses'] = {}
result['ctau']   = {}
sd               = seedL[0]

def train_model(): 
    global result
    vdpar = setParams( 
                       attrAll          = attrA                ,\
		       load_from_root   = 1                    ,\
		       bdtTrainOn       = 1                    ,\
		       ct               = CT_train             ,\
		       mass             = M_train              ,\
		       random_seed      = sd                   ,\
		       train_test_ratio = p['train_test_ratio'],\
		       N_bkg_to_test    = p['N_bkg_to_test']   ,\
		       N_bkg_to_train   = p['N_bkg_to_train']  ,\
		       maxDataLoadCut   = p['maxDataLoadCut']  ,\
		       test_mode=0 
                     )
    result['masses'][M_train] = mainF(vdpar)
    result['info']            = vdpar

       
# To make sure the qcd data for test and training don't overlap, the config should be same as the one for the training
def mass_compare(massL_l,load_root=1):
    global result
    for mi in massL_l:
        vdpar = setParams(
                           attrAll          = attrA                ,\
                           load_from_root   = load_root            ,\
                           bdtTrainOn       = 0                    ,\
                           ct               = CT_test_fix          ,\
                           mass             = mi                   ,\
                           random_seed      = sd                   ,\
                           train_test_ratio = p['train_test_ratio'],\
                           N_bkg_to_test    = p['N_bkg_to_test']   ,\
                           N_bkg_to_train   = p['N_bkg_to_train']  ,\
                           maxDataLoadCut   = p['maxDataLoadCut']  ,\
                           test_mode=1 
                         )
        result['masses'][mi] = mainF(vdpar)                  
 

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
                           bkg_test_multiple = 60000                ,\
                           bkg_multiple      = 8000                 ,\
                           maxDataLoadCut    = p['maxDataLoadCut']  ,\
                           N_sgn             = n_sgn 
                         )
        result['ctau'][cti] = mainF(vdpar)


if train_on == 1:    train_model()
if train_on == 0:
    mass_compare(massL, load_root=1) 
    #lifeTime_compare(ctauSL, load_root=1)

##################################dumping############################################
pklN     = 'res'+'_'+descrStr+'.pickle'
pth_dump = path_result + pklN
with open( pth_dump , 'wb' ) as f1:    pickle.dump(result,f1)
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Results stored at: ' + pth_dump
##################################dumping############################################











# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Old Lines:
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





