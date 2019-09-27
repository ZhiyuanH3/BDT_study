import os, sys
sys.path.append('./Tools/')
sys.path.append('./Templates/')
import multiprocessing as mp
import copy
import math
import argparse
from array import array
from time import time as tm
#import root_numpy     as rnp
import numpy          as np
import pandas         as pd
#from tools import ptrank, padd, showTimeLeft
from templates import *
from ROOT import ROOT, gROOT, TDirectory, TFile, gFile, TBranch, TLeaf, TTree
#from ROOT import AddressOf
#pwd = os.popen('pwd').read()
#pwd = pwd.split('\n')[0]
#pl = '.L ' + pwd + '/Objects' + '/Objects_m1.h+'
#gROOT.ProcessLine(pl)
#from ROOT import JetType, JetTypeSmall, JetTypePFC_fourVect, JetTypePFC_fiveVect, JetTypePFC_sixVect
#Js = JetType()



#################
# settings      #
#################
path               = '/beegfs/desy/user/hezhiyua/backed/fromLisa/fromBrianLLP/'
#path               = '/beegfs/desy/user/hezhiyua/backed/fromLisa/fromLisaLLP//'
#path              = '/beegfs/desy/user/hezhiyua/backed/dustData/'+'crab_folder_v2/'#'/home/brian/datas/roottest/'
num_of_jets           = 1#3#4
testOn                = 0
nonLeadingJetsOn      = 0#1
nLimit                = 10000000000000
numOfEntriesToScan    = 100 #only when testOn = 1
Npfc                  = 400
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< adjusted for different oldfile location
#path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/skim_output/'
#path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forBDT/' 
#path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forLola/large_sgn/'
path_out    = path+'plots/'

versionN_b  = 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1'
#versionN_s  = 'TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC'
versionN_s  = 'TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC'
HT          = '50'
lola_on     = 0 # 1: prepared for lola
ct_dep      = 0 # 1: for ct dependence comparison
cut_on      = 1
life_time   = [] 
sgn_ct      = '500'
treeName    = 'tree44'
JetName     = 'Jet1s'
fn          = ''
newFileName = ''#fn.replace('.root','_skimed.root')

jet_idx     = 0

len_of_lt = len(life_time)

pars = argparse.ArgumentParser()
pars.add_argument('--ht'    ,action='store',type=int,help='specify HT of the QCD file')
pars.add_argument('-s'      ,action='store',type=int,help='specify if for VBF file')
pars.add_argument('--model' ,action='store',type=str,help='specify model')
pars.add_argument('-t'      ,action='store',type=int,help='specify if to do test')
pars.add_argument('--ct'    ,action='store',type=int,help='specify signal life time')
pars.add_argument('--mass'  ,action='store',type=int,help='specify signal mass')
pars.add_argument('--nj'    ,action='store',type=int,help='specify number of jets')
args = pars.parse_args()
if args.ht  :    HT = str( args.ht ) 
if args.s   :    ct_dep = args.s
if args.t   :    testOn = 1
if args.ct  :    sgn_ct      = args.ct
if args.mass:    sgn_mass    = args.mass
if args.nj  :    num_of_jets = args.nj

if   args.model == 'bdt':
    lola_on    = 0
elif args.model == 'lola5':
    lola_on    = 1
    NumOfVecEl = 5
elif args.model == 'lola6':
    lola_on    = 1
    NumOfVecEl = 6

life_time.append( str(sgn_ct) )

#"""
if lola_on:
    forLola   = structure(model='lola', nConstit=Npfc       , dimOfVect=NumOfVecEl)
    forLola.Objh()
else      :
    forBDT    = structure(model='bdt' , nConstit=1          , preStr='J'          ) # nConstit=num_of_jets
    forBDT.Objh()
#"""


if   ct_dep == 0:
    matchOn = 0
    if   '50'  == HT: channel = {'QCD':'QCD_HT50to100_'  + versionN_b + '.root'}
    elif '100' == HT: channel = {'QCD':'QCD_HT100to200_' + versionN_b + '.root'}
    elif '200' == HT: channel = {'QCD':'QCD_HT200to300_' + versionN_b + '.root'}
    elif '300' == HT: channel = {'QCD':'QCD_HT300to500_' + versionN_b + '.root'}
    elif '500'  == HT: channel = {'QCD':'QCD_HT500to700_'   + versionN_b + '.root'}
    elif '700'  == HT: channel = {'QCD':'QCD_HT700to1000_'  + versionN_b + '.root'}
    elif '1000' == HT: channel = {'QCD':'QCD_HT1000to1500_' + versionN_b + '.root'}
    elif '1500' == HT: channel = {'QCD':'QCD_HT1500to2000_' + versionN_b + '.root'}
    elif '2000' == HT: channel = {'QCD':'QCD_HT2000toInf_'  + versionN_b + '.root'}
elif ct_dep == 1:
    matchOn = 1
    channel = {}
    for lt in life_time:
        channel['ct' + lt] = 'VBFH_HToSSTobbbb_MH-125_MS-' + str(sgn_mass) + '_ctauS-' + lt + '_' + versionN_s + '.root'

# Struct
#if   lola_on == 0:
#    Jets1 = JetTypeSmall() #for bdt: JetTypeSmall; for lola: JetTypePFC_fourVect
#elif lola_on == 1:
#    if   NumOfVecEl == 5:    Jets1 = JetTypePFC_fiveVect() 
#    elif NumOfVecEl == 6:    Jets1 = JetTypePFC_sixVect()

#-------------------------------------
cs            = {}
cs['pt_L']    = 'pt'  + '>' + '15'
cs['eta_L']   = 'eta' + '>' + '-2.4' 
cs['eta_U']   = 'eta' + '<' + '2.4'
cs['matched'] = 'isGenMatched' + '==' + '1'
#-------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#prs = 'Jet_old_dict[' +str(j+1) + '].'
prs = 'oldTree.Jets[k].' 
a   = ' and '
o   = ' or '
condition_str = '(' + prs + cs['pt_L']  + ')' +\
                a +\
                '(' + prs + cs['eta_L'] + ')' +\
                a +\
                '(' + prs + cs['eta_U'] + ')'
if matchOn == 1:
    condition_str = condition_str +\
                    a +\
                    '(' + prs + cs['matched']  + ')'
if   cut_on == 1: print '\nCuts:\n',condition_str
elif cut_on == 0: print 'no cut applied~'
#-------------------------------------------------------------------------------------------------------------------------
#######################################
if   nonLeadingJetsOn == 0:    whichJetStr = 'k==0' # >>>>>>>>>>>>>>>>>>>>> 0 or 1????
elif nonLeadingJetsOn == 1:    whichJetStr = 'k>=0'
#######################################
#DisplacedJets_Trigger_str = 'oldTree.HLT_VBF_DisplacedJet40_DisplacedTrack_v or oldTree.HLT_VBF_DisplacedJet40_DisplacedTrack_2TrackIP2DSig5_v or oldTree.HLT_HT350_DisplacedDijet40_DisplacedTrack_v or oldTree.HLT_HT350_DisplacedDijet80_DisplacedTrack_v or oldTree.HLT_VBF_DisplacedJet40_VTightID_DisplacedTrack_v or oldTree.HLT_VBF_DisplacedJet40_VVTightID_DisplacedTrack_v or oldTree.HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v or oldTree.HLT_VBF_DisplacedJet40_VTightID_Hadronic_v or oldTree.HLT_VBF_DisplacedJet40_VVTightID_Hadronic_v or oldTree.HLT_HT650_DisplacedDijet80_Inclusive_v or oldTree.HLT_HT750_DisplacedDijet80_Inclusive_v'
#######################################





import root_numpy     as rnp

def pick_ele(x):
    if len(x) == 0:    return None
    else          :    return x[0]


def plting(inName,n_pixHit):

    if 'VBF' in inName:
        xs_in    = xs['sgn']
    else:    
        xs_in    = xs[inName]
        print inName
        inName   = 'QCD_HT'+str(inName)+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1.root'



    attr_lst     = ['jetIndex','pt','energy','px','py','pz','isTrack','pdgId',     'nPixelHits']
    pre_str      = 'PFCandidates'
    cut_pre_str  = 'Jets['+str(jet_idx)+']'
    cut_attr_lst = ['pt','eta','isGenMatched'    ,'cHadEFrac','mass']#'nHadEFrac','cHadMulti','nHadMulti','npr']

    if 'QCD' in inName:    is_sgn = 0
    else              :    is_sgn = 1
    f1           = TFile(path + inName, 'r')
    tin          = f1.Get('ntuple/tree')
    NumE         = tin.GetEntriesFast()
    print '\nEntries: ', NumE
    s_cut        = None

    df_s    = {}
    cut_lst = []
    ftr_lst = []
    for i in cut_attr_lst:
        tmp_str        = cut_pre_str+'.'+i
        cut_lst.append(tmp_str)
        df_s[tmp_str] = pd.DataFrame( rnp.tree2array(tin, [tmp_str], stop=s_cut) )
    for i in attr_lst:
        tmp_str        = pre_str+'.'+i
        ftr_lst.append(tmp_str)
        df_s[tmp_str]        = pd.DataFrame( rnp.tree2array(tin, [tmp_str], stop=s_cut) )
    f1.Close()

    df = pd.DataFrame()
    for i in cut_lst:    df[i] = df_s[i]
    for i in ftr_lst:    df[i] = df_s[i]
    for i in cut_lst:    df[i] = df[i].apply(pick_ele)

    #if drop_nan:    df = df.dropna()
    mask_pt  = df[cut_pre_str+'.'+'pt'] > 15
    mask_eta = (df[cut_pre_str+'.'+'eta'] > -2.4) & (df[cut_pre_str+'.'+'eta'] < 2.4)
    mask_mtc = df[cut_pre_str+'.'+'isGenMatched'] == 1
    mask_bkg = mask_pt & mask_eta
    mask_sgn = mask_bkg & mask_mtc

    print df.shape

    if is_sgn: msk = mask_sgn
    else     : msk = mask_bkg

    if cut_on: df  = df[msk]
    print df.shape
    #df.drop(cut_lst,axis=1)








    df            = df[ df[pre_str+'.'+'energy'].apply(len) != 0 ]
    attr_list0    = ['jetIndex','energy','px','py','pz','isTrack','pdgId',     'nPixelHits']
    #attr_list0    = ['jetInd','energy','px','py','pz','c','pdgId','nPixHit']
    df_dict       = {}
    for a in attr_list0:
        a          = pre_str+'.'+a  
        df_dict[a] = pd.DataFrame.from_records( df[a].values.tolist() )
        jet_one    = df_dict[pre_str+'.'+'jetIndex']     == jet_idx#0 #1
        df_dict[a] = df_dict[a][jet_one]


    for a in attr_list0:
        a               = pre_str + '.' + a
        #df_dict[a] = pd.DataFrame.from_records( df[a].values.tolist() )
        is_charged      = df_dict[pre_str+'.'+'isTrack']     == 1
        #at_least_n_pHit = df_dict['nPixHit'] > n_pixHit 
        at_most_n_pHit  = df_dict[pre_str+'.'+'nPixelHits'] > n_pixHit
        #at_most_n_pHit  = df_dict[pre_str+'.'+'nPixelHits'] < n_pixHit
        df_dict[a]      = df_dict[a][is_charged]
        #df_dict[a]      = df_dict[a][at_least_n_pHit]
        df_dict[a]      = df_dict[a][at_most_n_pHit]    

        #df_dict[a]      = df_dict[a].apply(lambda x: ,axis=1)  

        if a == pre_str+'.'+'nPixelHits':
            #print df_dict[a].shape[1]
            df_count =  df_dict[a].count(axis='columns')
            #print df_dict[a] 
            #print df_count
            nt_list  = df_count.values.tolist()
            n_events = len(df_count)
            print n_events
            
            Weight   = xs_in / float(n_events * tot_xs) 
            
            weight_list = [Weight] * len(nt_list)


    return nt_list, weight_list#, n_pfc_list


#name_qcd = 'QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1.root'
name_sgn = 'VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-1000_TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC.root'
#name_sgn = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-1000_TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC.root'

#qcd_list, qcd_n_pfc_list = plting(name_qcd)
#sgn_list, sgn_n_pfc_list = plting(name_sgn)
#qcd_list = plting(name_qcd,n_pixHit_i)
#sgn_list = plting(name_sgn,n_pixHit_i)


str_qcd_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']


from os import system as act
import matplotlib.pyplot as plt
#num_bins = 40#50
num_bins = range(0,40,2)

to_normalize = True
to_cumulate  = False#-1#True
to_log       = True

n_range      = 10#3#1
xs           = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }

tot_xs       = 0
for qcd_i in str_qcd_list:
    tot_xs  += xs[qcd_i]


def run_i(name_qcd):
    name_qcd = 'QCD_HT'+str(name_qcd)+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1.root'    
    print name_qcd

name_qcd_list = []


for n_pixHit_i in range(n_range):
    titles  = 'num of charged PFC with nPixelHits > '+str(n_pixHit_i)
    #titles  = 'num of charged PFC with nPixelHits < '+str(n_pixHit_i)
    xlabels = titles
    
    #qcd_list = plting(name_qcd,n_pixHit_i)
    #sgn_list = plting(name_sgn,n_pixHit_i)
    qcd_tot_list = []
    sgn_tot_list = []
    qcd_tot_w_list = []
    sgn_tot_w_list = []

    for i in str_qcd_list:
        #name_qcd_i = 'QCD_HT'+str(i)+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1.root'
        #name_qcd_list.append(str_i)
        print i
        qcd_list, qcd_w = plting(i,n_pixHit_i)
        #sgn_list, sgn_w = plting(name_sgn,n_pixHit_i)        
        qcd_tot_list   += qcd_list
        #sgn_tot_list   += sgn_list
        qcd_tot_w_list += qcd_w
        #sgn_tot_w_list += sgn_w

    sgn_list, sgn_w = plting(name_sgn,n_pixHit_i)
    sgn_tot_list   += sgn_list
    #sgn_tot_w_list += sgn_w

    n_array_qcd, bin_array_qcd, _ = plt.hist(qcd_tot_list, num_bins, weights=qcd_tot_w_list, normed=to_normalize, facecolor='blue', alpha=0.5, label='QCD', log=to_log, cumulative=to_cumulate)
    n_array_sgn, bin_array_sgn, _ = plt.hist(sgn_tot_list, num_bins, normed=to_normalize, facecolor='red', alpha=0.5, label='VBF', log=to_log, cumulative=to_cumulate)
    plt.legend(loc='upper right')
    plt.xlabel(xlabels)  
    plt.ylabel('au.')  
    plt.title(titles)  
    plt.subplots_adjust(left=0.15)  
    plt.savefig(path_out+'nPixHit_'+str(n_pixHit_i)+'.png')
    #plt.show()
    plt.close()








    
