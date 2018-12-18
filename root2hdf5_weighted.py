#######################################
# Imports
#######################################
print "Imports: Starting..."
import sys
if len(sys.argv) != 4:
    print "Enter signal_file background_file n_constituents as arguments"
    sys.exit()
    
import os
import pickle
import pdb

print "Imported basics"
import numpy  as np
import pandas as pd
import h5py
import time

import init_data         as id
from sklearn.externals   import joblib
from main_load_datas     import LoadData_main
print "Imports: Done..."

##doesn't work if input files large - too much memory

########################################
# Configuration
########################################
"""
infname_sig = sys.argv[1]
infname_bkg = sys.argv[2]
n_cands = int(sys.argv[3])
"""
"""
infname_sig = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'
infname_bkg = 'QCD_HT100To200_pfc_1j_skimed.root'#'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'#'QCD_HT100To200_pfc_1j_skimed.root'
"""
n_cands = 40
mass    = 40
ct      = 500  

dsc     = 'testing'

path_data = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forLola/'
path_out  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forLola/h5/'

p                     = {}
p['train_test_ratio'] = 0.6
p['N_bkg_to_train']   = 20000#2000000
p['N_bkg_to_test']    = 5330#11400000
p['maxDataLoadCut']   = 888888888


version = "v0_{0}cs".format(n_cands)

if 1:
    params = {
               'load_from_root'                : 1,#p['load_from_root'],
               'weight_on'                     : 1, #0: not weighting qcd
               #--------------------------------------------------------------Data
               #'N_sgn'                         : n_sgn,
               'descr'                         : dsc,
               'test_mode'                     : 0,#p['test_mode'],                
               'maxDataLoadCut'                : p['maxDataLoadCut'],
               'dataUseThrsd'                  : 50000,
               'num_of_jets'                   : 1,#num_of_jets,
               'selectedSample'                : 1,#selectionOn,
               'sgn_mass'                      : mass,
               'sgn_ctauS'                     : ct,
               'qcdPrefix'                     : 'QCD_HT',
               'sgnPrefix'                     : 'VBFH_HToSSTobbbb_MH-125_MS-',
               'versionN_b'                    : 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
               'versionN_s'                    : 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC',
               'train_test_ratio'              : p['train_test_ratio'],
               'N_bkg_to_test'                 : p['N_bkg_to_test'],
               'N_bkg_to_train'                : p['N_bkg_to_train'],
               #'bkg_multiple'                  : bkg_multiple,#10,#15,#10,
               #'bkg_test_multiple'             : bkg_test_multiple,#50000,#multipl,#100,#multipl,#70,#100
               #--------------------------------------------------------------General
               'xs'                            : { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 },
               #Fixing random state for reproducibility
               'random_seed'                   : 4444,
               #-----------------------------------------------------Default_Strings
               'isSigL'                        : 'is_signal',
               'weightL'                       : 'weight',
               'path'                          : path_data,
               'loadedDatas_dir'               : 'loadedDatas',
              }




## load data
start = time.time()

#xs          = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }
train_ratio = 0.6
test_ratio  = 0.2
val_ratio   = 0.2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pkls         = LoadData_main( params )
df_train     = pkls['df_train_o']
df_test_orig = pkls['df_test_orig_o']
#in_dict      = pkls['out_dict']

df_all       = df_train.copy()
"""
ColumnLabelDict    = in_dict['ColumnLabelDict']
ColumnLabelDict_sk = in_dict['ColumnLabelDict_sk']
JetPrfx            = in_dict['JetPrfx']
isSigPos           = in_dict['isSigPos']
weightPos          = in_dict['weightPos']
JetPrfx            = in_dict['JetPrfx']
"""    
open_time = time.time()

df       = pd.DataFrame()
df_t     = pd.DataFrame()

#dfTest         = pd.DataFrame()
#dfTest_t       = pd.DataFrame()
#dfTest         = df_all
#dfTest_t       = df_test_orig
#dfTest['tt']   = 1
#dfTest_t['tt'] = 1
print df_all

attr_trans       = {}
attr_trans['E']  = 'energy'
attr_trans['PX'] = 'px'
attr_trans['PY'] = 'py'
attr_trans['PZ'] = 'pz'
attr_trans['C']  = 'ifTrack'

for i in range(n_cands):     
    for j in ['E','PX','PY','PZ','C']:
        lbl_str_out       = j+'_'+str(i)
        lbl_str_in        = 'Jet'+str(1)+'s_pfc'+str(i+1)+'_'+ attr_trans[j]
        df[lbl_str_out]   = df_all[lbl_str_in]
        df_t[lbl_str_out] = df_test_orig[lbl_str_in]

    """
    df["E_{0}".format(i) ]   = df_all["Jet{0}s_pfc{1}_energy".format(1, i+1) ]
    df["PX_{0}".format(i)]   = df_all["Jet{0}s_pfc{1}_px".format(1, i+1) ]
    df["PY_{0}".format(i)]   = df_all["Jet{0}s_pfc{1}_py".format(1, i+1) ]
    df["PZ_{0}".format(i)]   = df_all["Jet{0}s_pfc{1}_pz".format(1, i+1) ]
    df["C_{0}".format(i)]    = df_all["Jet{0}s_pfc{1}_ifTrack".format(1, i+1) ]
 
    df_t["E_{0}".format(i) ] = df_test_orig["Jet{0}s_pfc{1}_energy".format(1, i+1) ]
    df_t["PX_{0}".format(i)] = df_test_orig["Jet{0}s_pfc{1}_px".format(1, i+1) ]
    df_t["PY_{0}".format(i)] = df_test_orig["Jet{0}s_pfc{1}_py".format(1, i+1) ]
    df_t["PZ_{0}".format(i)] = df_test_orig["Jet{0}s_pfc{1}_pz".format(1, i+1) ]
    df_t["C_{0}".format(i)]  = df_test_orig["Jet{0}s_pfc{1}_ifTrack".format(1, i+1) ]
    """
    """
    df["E_{0}".format(i) ] = dfTest['tt']
    df["PX_{0}".format(i)] = dfTest['tt']
    df["PY_{0}".format(i)] = dfTest['tt']
    df["PZ_{0}".format(i)] = dfTest['tt']
    df["C_{0}".format(i) ] = dfTest['tt']#df_all["Jet{0}s_pfc{1}_ifTrack".format(1, i+1) ]
    df_t["E_{0}".format(i) ] = dfTest_t['tt']
    df_t["PX_{0}".format(i)] = dfTest_t['tt']
    df_t["PY_{0}".format(i)] = dfTest_t['tt']
    df_t["PZ_{0}".format(i)] = dfTest_t['tt']
    df_t["C_{0}".format(i) ] = dfTest_t['tt']#df_test_orig["Jet{0}s_pfc{1}_ifTrack".format(1, i+1) ]
    """

df["is_signal_new"]   = df_all[ params['isSigL'] ]#["is_signal"]
df["weight"]          = df_all[ params['weightL'] ]#["weight"]
df_t["is_signal_new"] = df_test_orig[ params['isSigL'] ]#["is_signal"]
df_t["weight"]        = df_test_orig[ params['weightL'] ]#["weight"]


# Train / Test / Validate
# ttv==0: 60% Train
# ttv==1: 20% Test
# ttv==2: 20% Final Validation
df["ttv"] = np.random.choice([0,2], df.shape[0],p=[ train_ratio/float(train_ratio+val_ratio) , val_ratio/float(train_ratio+val_ratio) ])
train = df[ df["ttv"]==0 ]
val   = df[ df["ttv"]==2 ]
test  = df_t
        
print len(df), len(train), len(test), len(val)

train.to_hdf( path_out + 'vbf_qcd-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf(  path_out + 'vbf_qcd-test-{0}.h5'.format(version) ,'table',append=True)
val.to_hdf(   path_out + 'vbf_qcd-val-{0}.h5'.format(version)  ,'table',append=True)



close_time = time.time()
print "Time for the lot: ", (close_time - open_time)



