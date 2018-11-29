from sklearn.externals import joblib
import pickle
import matplotlib
matplotlib.use('Agg') #prevent the plot from showing
from   matplotlib import pyplot as plt
import numpy as np

def plotROC(inDict):

    Colors      = ['red','green','blue','black','lime','goldenrod','slateblue','yellowgreen','navy','yellow']
    colors      = ['pink','lightgreen','paleturquoise','grey','greenyellow','wheat','plum','y','gold']
    i           = 0
    orderedList = []
    for dsci, dicti  in inDict.iteritems():
        orderedList.append(dsci)
    orderedList = sorted(orderedList)
    for dsci in orderedList:

        auc_bdt = inDict[dsci]['aoc']#dicti['aoc']


        self_cut_base = 1
        use_indp_roc  = 0
        if   use_indp_roc == 1:
            fpr_bdt = inDict[dsci]['eb']#dicti['eb']
            tpr_bdt = inDict[dsci]['es']#dicti['es']
        elif use_indp_roc == 0:
            fpr_bdt = inDict[dsci]['fpr']#dicti['fpr']
            tpr_bdt = inDict[dsci]['tpr']#dicti['tpr']
        plt.plot(tpr_bdt, fpr_bdt, label=dsci+", AOC=%.4f"%auc_bdt, color=colors[i])
        #unc = inDict[dsci]['unc']
        #plt.fill_between(tpr_bdt, fpr_bdt-unc[:18610], fpr_bdt+unc[:18610])  
        if self_cut_base == 1:
            CBdict = inDict[dsci]['cut_base']#dicti['cut_base']
            for dsci, dicti  in CBdict.iteritems():
                if 'H' in dsci: Marker = 'v'
                else          : Marker = '.'     
                sgn_eff = dicti[0]
                fls_eff = dicti[1] 
                plt.plot(sgn_eff, fls_eff, 'or', label=dsci+': (TPR=%.3f,FPR=%.5f)'%(sgn_eff,fls_eff), color=colors[i], marker=Marker)
        i += 1

    fpr = np.asarray( inDict['50']['Fpr'] )
    tpr = np.asarray( inDict['50']['Tpr'] )
    unc = np.asarray( inDict['50']['unc'] )
    #print unc
    plt.plot(tpr, fpr)
    plt.fill_between(tpr, fpr-unc, fpr+unc, alpha=0.4)



def plotCuts(inDict):
    colors = ['red','green','blue']
    i = 0
    for dsci, dicti  in inDict.iteritems():
        sgn_eff = dicti[0]
        fls_eff = dicti[1]
        plt.plot(sgn_eff, fls_eff, 'or', label=dsci+': (TPR=%.3f,FPR=%.5f)'%(sgn_eff,fls_eff), color=colors[i])
        i += 1

def plotROC_main(pathOut,outName,cutBase_dict,pkl_dict):
    if 1:
        plt.subplots_adjust(hspace=0.4)
        fig = plt.figure(num=None, figsize=(21, 9), dpi=120, facecolor='w', edgecolor='k')

        ax = plt.subplot(121)
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        axes = plt.gca()
        axes.set_xlim([0.02,0.54])#0.02,0.44 
        axes.set_ylim([0.000001,0.1])#0.0001
        plotROC(pkl_dict)
        #plotCuts(cutBase_dict)
        plt.grid(True)
        #plt.legend( loc=4 , prop={'size': 15} )
        plt.title('ROC (zoomed in)', fontsize=24)
        plt.ylabel('False Positive Rate', fontsize=20)
        plt.xlabel('True Positive Rate', fontsize=20)

        plt.subplot(122)
        plotROC(pkl_dict)
        #plotCuts(cutBase_dict)
        #plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.grid(True)
        plt.legend( loc=2 , prop={'size': 15} ) #9#12#15
        plt.title('ROC', fontsize=24)
        plt.ylabel('False Positive Rate', fontsize=20)
        plt.xlabel('True Positive Rate', fontsize=20)

        #plt.ioff()
        plt.close()
        #plt.show(block=False)

        fig.savefig(pathOut + outName + '.png', bbox_inches='tight')
#########################################################################################









###########
#         #
# testing #
#         #
###########
if __name__ == '__main__':
 
    plot_on      =    1#0 
    out_name     =    'test'
    path         =    '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/bdt_overview/'
    #path         =    '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
    #path         =    '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/forCompare/'
    path_out     =    '/beegfs/desy/user/hezhiyua/LLP/bdt_output/roc/'

    fileNameDict = {
                     'BDT()'       : 'result_with_pt_mass_energy_v1_withSelection.pickle',#'result_v1_withSelection.pickle',
                     'bdt'         : 'result_v1_withSelection.pickle' 
                   }
   
    """
    fileNameDict = {
                      'BDT(nSelectedTracks+DisplacedJetsTrigger)'     :'result_v12_withSelection.pickle',
                      'BDT(nSelectedTracks+DisplacedJetsTrigger+kin.)':'result_with_pt_mass_energy_v13_withSelection.pickle',
                      'BDT(best 5 variables)'    :'result_v15_withSelection.pickle',
                      'BDT(best 5 with kin.)'    :'result_with_pt_mass_energy_v16_withSelection.pickle',
                      'BDT(all variables)'       :'result_with_pt_mass_energy_v14_withSelection.pickle',
                    }
    """
    #"""
    cutBaseDict  = {
                      'Loose Cut': [0,0],#[0.1955307262569827 , 0.003643577798535045],
                      'Hard Cut' : [0,0]#[0.15642458100558612, 0.0006177645800627758]
                   }
    #"""

    """
    pklDict = {}
    for lbi, fileNamei in fileNameDict.iteritems():
        with open(path+fileNamei,'read') as f1:
            pklDict[lbi] = pickle.load(f1)
    """
    pklDict = {}
    fn = 'res_kin_v0_withSelection_train_on_40_500mm_test_on_40_500mm_2best.pickle'
    #fn = 'result_kin_v4_withSelection_compare_masses.pickle'
    #fn = 'result_v4_withSelection_compare_masses.pickle'
    #fn = 'result_kin_v2_withSelection_compare_ctau.pickle'
    #fn = 'result_v2_withSelection_compare_ctau.pickle'
    #fn = 'result_kin_v1_withSelection_debug.pickle'
    #fn = 'result_kin_v2_withSelection_debug.pickle'

    with open(path+fn,'read') as ff:    
        pkls    = pickle.load(ff)
        #feature = pkls['ctau']
        feature = pkls['masses'] 
        for mm, dicti in feature.iteritems(): 
            pklDict[str(mm)] = dicti

    #print pkls['masses'][40]
    #exit()

    ####################################################################
    pth = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
    roc_delta = joblib.load(pth+'/dumps/roc.pkl')
    pklDict['50']['unc']=roc_delta['unc']
    pklDict['50']['Tpr']=roc_delta['tpr']
    pklDict['50']['Fpr']=roc_delta['fpr']
    #print roc_delta['upper']['tpr']
    #print roc_delta['upper']['fpr']   
    """
    pklDict['50_u']={}
    pklDict['50_l']={}
    pklDict['50_u']['tpr']=roc_delta['upper']['tpr']
    pklDict['50_u']['fpr']=roc_delta['upper']['fpr']
    pklDict['50_u']['aoc']=0#roc_delta['upper']['aoc']
    pklDict['50_u']['cut_base']=cutBaseDict

    pklDict['50_l']['tpr']=roc_delta['lower']['tpr']
    pklDict['50_l']['fpr']=roc_delta['lower']['fpr']
    pklDict['50_l']['aoc']=0#roc_delta['lower']['aoc']
    pklDict['50_l']['cut_base']=cutBaseDict 
    """
    ####################################################################

    length = len(pkls['masses'][50]['tpr'] )
    #print len(pkls['thresholds_bdt'])
    #for i in xrange(length):
    #    if pkls['masses'][50]['tpr'][i] >= 0.14:#0.2: 
    #        print i 
    #        exit()
    
    for i in range(6):

        print pkls['masses'][50]['tpr'][641+i]
        print pkls['masses'][50]['fpr'][641+i]
 
    print pkls['masses'][50]['threshold'][641]
    print pkls['masses'][50]['threshold'][642]

    #length = len(pkls['tpr'] )
    #print len(pkls['thresholds_bdt'])
    #for i in xrange(length):
    #    if pkls['tpr'][i] >= 0.2: 
    #        print i 
    #        exit()

    #print pkls['thresholds_bdt'][306]
    #print pkls    
    if plot_on:
        plotROC_main(path_out, out_name, cutBaseDict, pklDict)

