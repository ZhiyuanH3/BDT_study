from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hm import *

path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/generalization_bdt/rs/'


def read_pkl(pth):
    pkls = joblib.load(pth)
    return pkls['data']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

trn_m = 40
trn_l = 5000

val = 'val'
val = 'err'

mass_list    = [20,30,40,50,60]
ctau_list    = [500,1000,2000,5000]

cut_type     = ['hard_cut']#['loose_cut','hard_cut']
inputs       = ['full']#['2best','full']
kin_var      = ['kin0']#['kin0','kin1']

combi_dict   = {}
cc           = 0
for i in inputs:
    for j in kin_var:
        tmp_list = []
        tmp_list.append(i)
        tmp_list.append(j)
        combi_dict[cc] = '_'.join(tmp_list)
        cc            += 1

empty_log = []
out_dict  = {}

if 1:
    for ci in cut_type:
        out_dict[ci] = {}
        for key, item in combi_dict.iteritems():
            out_dict[ci][item] = {}
            for m_trn in [trn_m]:
                out_dict[ci][item][m_trn]            = {}
                for l_trn in [trn_l]:
                    out_dict[ci][item][m_trn][l_trn] = {}
                    for m_tst in mass_list:
                        out_dict[ci][item][m_trn][l_trn][m_tst] = {}
                        for l_tst in ctau_list:
                            out_dict[ci][item][m_trn][l_trn][m_tst][l_tst] = {}
                            trn_part = str(m_trn)+'GeV_'+str(l_trn)+'mm_'
                            tst_part = 'tst'+'_'+str(m_tst)+'GeV_'+str(l_tst)
                            file_to_look = 'RS_'+'trn'+'_'+trn_part+tst_part+'mm_slct1_attr_'+item+'_v0.pkl'
                            print file_to_look
                            path_tot     = path + file_to_look
                            in_dict      = read_pkl(path_tot)
                            if in_dict:
                                roc_dict = in_dict['roc']
                                fpr_bdt  = roc_dict['fpr']
                                tpr_bdt  = roc_dict['tpr']
                                #e_tpr_l  = roc_dict['e_tpr_l']
                                e_fpr_l  = roc_dict['e_fpr_l']
                                #e_tpr_h  = roc_dict['e_tpr_h']
                                e_fpr_h  = roc_dict['e_fpr_h']

                                cut_dict = in_dict['cut_based']
                                dicti    = cut_dict[ci]
                                sgn_eff  = dicti['tpr']
                                fls_eff  = dicti['fpr']
                                #tpr_e_l  = dicti['tpr_e_l']
                                fpr_e_l  = dicti['fpr_e_l']
                                #tpr_e_h  = dicti['tpr_e_h']
                                fpr_e_h  = dicti['fpr_e_h']

                                if sgn_eff != 0:
                                    tmp_tpr, indx = find_nearest(tpr_bdt, sgn_eff)
                                    tmp_fpr       = fpr_bdt[indx]



                                    count     = 1
                                    if_done   = False
                                    print 'sgn_eff', sgn_eff
                                    while not if_done:
                                        if   tmp_tpr > sgn_eff:
                                            idx_h     = indx
                                            idx_l     = indx+count
                                            tmp_tpr_h = tmp_tpr
                                            tmp_tpr_l = tpr_bdt[idx_l]
                                            if tmp_tpr_l < sgn_eff:    if_done = True
                                        elif tmp_tpr < sgn_eff:
                                            idx_l     = indx
                                            idx_h     = indx-count
                                            tmp_tpr_l = tmp_tpr
                                            tmp_tpr_h = tpr_bdt[idx_h]
                                            idx_l     = indx
                                            if tmp_tpr_h > sgn_eff:    if_done = True
                                        elif tmp_tpr == sgn_eff:
                                            idx_h     = indx
                                            idx_l     = indx
                                            tmp_tpr_l = tmp_tpr
                                            tmp_tpr_h = tmp_tpr
                                            if_done   = True
                                        print 'tmp_tpr_l', tmp_tpr_l
                                        print 'tmp_tpr_h', tmp_tpr_h
                                        count += 1

                                    delta_fpr_l   = np.abs( tmp_fpr-(fpr_bdt[idx_l]-e_fpr_l[idx_l]) )
                                    delta_fpr_h   = np.abs( (fpr_bdt[idx_h]+e_fpr_h[idx_h])-tmp_fpr )
                                    err_fpr       = np.max([delta_fpr_l,delta_fpr_h,e_fpr_h[indx],e_fpr_l[indx]])
                                    fpr_err       = np.maximum(fpr_e_l,fpr_e_h)

                                    if tmp_fpr != 0:
                                        inv_fpr     = fls_eff/float(tmp_fpr)
                                        inv_fpr_err = np.sqrt(np.square(fpr_err/fls_eff) + np.square(err_fpr/tmp_fpr))

                                    else           :
                                        print '>>>>>>>>>>>>>>>>>>>> Zero devision!'
                                        inv_fpr     = 0
                                        inv_fpr_err = 0
                                else:    pass # !!!!!!!!!!!!!!!!!!!!!!!!!!!! hard coded cut TPR!!!!!

                                out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR']      = inv_fpr
                                out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR--err'] = inv_fpr_err

                            else:
                                print '>>>>>>>>>>>>>>>>>>>> no in_dict!!'
                                out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR']      = 0
                                out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR--err'] = 0

                                empty_log.append(file_to_look)


    path_out = './plot/'

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Empty_log:', empty_log

    #print out_dict
    ####################################################################
    ####################################################################

