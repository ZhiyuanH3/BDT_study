import os
import func                as fc
import pandas              as pd
import numpy               as np
import root_numpy
from ROOT              import TFile
from sklearn           import utils
from sklearn.externals import joblib
from ROOT              import TGraphAsymmErrors as GAE
from ROOT              import TH1F
from ROOT              import Double

###############
#             #
###############
class DataFile:
    numOfParameters = 0 ,
    prefix          = '',
    jetPrefix       = '',
    fileName        = '',
    fullName        = '',
    fileType        = '',
    path            = '',
    dataType        = '',
    dictName        = '',
    ifTest          = False,
    ifSkimed        = False,
    forLoLa         = False,
    forBDT          = False,
    fixPart         = [],
    crossSection    = -1,
    HTmin           = -1,
    HTmax           = -1,
    MS              = -1,
    ctau            = -1,    
    
    def __init__(self,fn,pth,*fp):
        self.fileName  = fn
        self.fullName  = pth+fn
        self.path      = pth  
        self.fixPart   = fp
        
        dotPos         = self.fileName.find('.')
        self.fileType  = self.fileName[dotPos+1:]
        if 'test' in fn:    self.ifTest   = True
        else           :    self.ifTest   = False  

        if 'skim' in fn:    self.ifSkimed = True
        else           :    self.ifSkimed = False

        if 'LoLa' in fn:    self.forLoLa  = True
        else           :    self.forLola  = False   

        if   'QCD' in fn:
            self.dataType = 'QCD'
            self.HTmin    = fc.FindNum(fn,'HT')[0]
            if 'Inf' in fn:
                self.HTmax    = 88888888#fc.FindNum(fn,'to')[0] #'To'       
                self.dictName = str(self.HTmin) + 'to' + 'Inf'
            else:
                self.HTmax    = fc.FindNum(fn,'to')[0] #'To'
                self.dictName = str(self.HTmin) + 'to' + str(self.HTmax) #'To'
        elif 'ctauS' in fn:
            self.dataType = 'SGN'    
            self.MS       = fc.FindNum(fn,'MS')[0]
            self.ctau     = fc.FindNum(fn,'ctauS')[0]            

    def setXS(self,cs):
        self.crossSection = cs
    def getName(self):
        return self.fileName
    def getDictName(self):
        return self.dictName
    def getDataType(self):
        return self.dataType
    
    def setNumOfPar(self,nop):
        self.numOfParameters = nop
    def setFileName(self,fn):
        self.fileName = fn
    def setFileType(self,ft):
        self.fileType = ft
    

###############
#             #
###############
def SetCrossSection(inst,xs):
    for w in inst:
        inst[w].setXS(  xs[ inst[w].getDictName() ]  )
              
###############
#             #
###############
def LoadData(ps, name_dict, setWeight=False, mainTree='tree44', thrsd=1000000):
    df_dict     = {}
    N_dict      = {}
    N_available = 0
    for w in name_dict:    
        descrStr    = ps['descr'][2] + '_' + ps['descr'][5]
        preload_pth = ps['path']+'/'+ps['loadedDatas_dir']+'/'+'preload_'+descrStr+'_'+w+'.pkl'
        if os.path.isfile(preload_pth):    
            in_dict   = joblib.load(preload_pth)
            nevents   = in_dict['N']
            N_dict[w] = nevents
            if nevents > thrsd:    N_dict[w] = thrsd 
            print "number of events: " + str(nevents)
            print str(N_dict[w]) + ' loaded..'
            print '................................loading data'
            df_dict[w] = in_dict['df']
        else                          :    
            out_dict      = {}
            fn            = name_dict[w].fullName
            fb            = TFile(fn,"r")
            print '................................loading tree'
            tree          = fb.Get(mainTree) 
            nevents       = tree.GetEntries()
            N_dict[w]     = nevents
            out_dict['N'] = nevents
            if nevents > thrsd:    N_dict[w] = thrsd 
            print "number of events: " + str(nevents)
            print str(N_dict[w]) + ' loaded..'
            print '................................tree loaded'
            #~~~~~~~~set up DataFrames
            print '................................loading data'
            df_dict[w]     = pd.DataFrame( root_numpy.root2array(fn, treename=mainTree, start=0, stop=N_dict[w]) ) 
            out_dict['df'] = df_dict[w]
            joblib.dump(out_dict, preload_pth)
        print '................................data completely loaded'
        if   name_dict[w].dataType == 'QCD':    df_dict[w]['is_signal'] = 0 
        elif name_dict[w].dataType == 'SGN':    df_dict[w]['is_signal'] = 1
        #~~~~~~~~drop events with values = -1 
        print df_dict[w].columns.values[:]
        firstColName = df_dict[w].columns.values[0]
        dropone      = df_dict[w][firstColName] != -1 
        df_dict[w]   = df_dict[w][:][dropone]
        #~~~~~~~~get features
        N_available  = N_available + nevents
  
    return df_dict, N_dict, N_available


###############
#             #
###############
def getColumnLabel(DF,refLab='_'):
    colLab_dict    = {}
    colLab_dict_sk = {}
    numOfCol       = len( DF.columns )
    i              = 0
    jetPrfx        = fc.getPrefix(DF.columns.values[0],refLab)
    jetPrfxLen     = len( jetPrfx )
    while i < numOfCol:       
        colLab_dict[DF.columns.values[i]]      = i
        if jetPrfx in DF.columns.values[i]:    colLab_dict_sk[DF.columns.values[i][jetPrfxLen:]] = i
        else                              :    colLab_dict_sk[DF.columns.values[i]]              = i    
        i += 1
    return colLab_dict, colLab_dict_sk, jetPrfx


###############
#   carefull  #
###############

# specify a number to train for qcd:
def SplitDataNew(df_bkg_dict         ,\
                 df_sig              ,\
                 N_bkg_dict          ,\
                 bkg_name_dict       ,\
                 xs                  ,\
                 N_bkg_to_test       ,\
                 N_bkg_to_train      ,\
                 N_available_bkg     ,\
                 N_available_sgn     ,\
                 train_test_ratio    ,\
                 random_seed         ,\
                 Thrsd     = 300000  ,\
                 weightL   = 'weight',\
                 test_mode = 0         ):

    N_bkg_to_test_dict  = {}
    N_bkg_to_train_dict = {}
    df_bkg_test_list    = []
    df_bkg_train_list   = []
    tot_xs              = 0 

    train_test_cut_ratio  = 0.15#0.5

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Calculate the total cross section:
    for w in bkg_name_dict:    tot_xs += xs[w]
    #print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>total cross section: ', str(tot_xs)
    
    thrd   = N_bkg_to_test + N_bkg_to_train 
 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Check if number of background events is enough 
    if N_available_bkg >= thrd:

        N_sgn_to_train        = int( N_available_sgn * train_test_ratio )
        N_sgn_to_test         = N_available_sgn - N_sgn_to_train                 

        df_bkg_train_dict     = {}
        df_bkg_test_dict      = {}  
        N_bkg_train_dict      = {} 
        N_bkg_test_dict       = {} 

        for w in bkg_name_dict:
            train_test_cut_position  = int( N_bkg_dict[w] * train_test_cut_ratio ) 
            
            df_bkg_train_dict[w]     = df_bkg_dict[w].copy()[:train_test_cut_position]
            df_bkg_test_dict[w]      = df_bkg_dict[w].copy()[train_test_cut_position:]  

        N_taken_train        = 0       
        N_taken_test         = 0
        for w in bkg_name_dict:   
            # If #background from a specific HT bin is smaller than Thrsd, take all the events in that bin 
            if N_bkg_dict[w] < Thrsd: 
                N_taken_train            += int(N_bkg_dict[w]*train_test_cut_ratio)
                N_taken_test             += int(N_bkg_dict[w]*(1-train_test_cut_ratio))  
            
                N_bkg_to_test_dict[w]     = int(N_bkg_dict[w]*(1-train_test_cut_ratio))
                N_bkg_to_train_dict[w]    = int(N_bkg_dict[w]*train_test_cut_ratio)

        bkg_needed_test       = N_bkg_to_test
        bkg_needed_train      = N_bkg_to_train
 
        N_to_take_test        = bkg_needed_test  - N_taken_test
        N_to_take_train       = bkg_needed_train - N_taken_train

        if (N_to_take_test<=0) or (N_to_take_train<=0): 
            print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Negative number: Please change Thrsd~'

        N_available_test_bkg_left   = int(N_available_bkg*(1-train_test_cut_ratio)) - N_taken_test
        N_available_train_bkg_left  = int(N_available_bkg*train_test_cut_ratio)     - N_taken_train

        for w in bkg_name_dict:
            if N_bkg_dict[w] >= Thrsd:
                # Decide how many events should be used(train and test): proportional to the relative portion of a specificHT bin out of the total #available background events
                N_bkg_to_test_dict[w]     = int(  int(N_bkg_dict[w]*(1-train_test_cut_ratio)) * ( N_to_take_test/float(N_available_test_bkg_left) )  )         
                N_bkg_to_train_dict[w]    = int(  int(N_bkg_dict[w]*train_test_cut_ratio) * ( N_to_take_train/float(N_available_train_bkg_left) )  )


            if N_bkg_to_train_dict[w] == 0:
                # Error prompt if the relative portion of the HT bin is too small
                print '>>>>>>Zero division>>>>Please check if number of higher HT QCD-samples are too large?'
                N_bkg_to_train_dict[w]    = int( N_bkg_dict[w] * train_test_ratio )
            #if N_bkg_to_use_dict[w]   == 0:
            #    N_bkg_to_use_dict[w]      = N_bkg_dict[w]

            # Add a 'weight' column to the dataframe
            df_bkg_test_dict[w][weightL]    = 0   # Add weight column
            df_bkg_train_dict[w][weightL]   = 0   # Add weight column 
            # Add weights to cope with the relative cross section and to compensate the number of events being used for training
            df_bkg_train_dict[w].loc[:N_bkg_to_train_dict[w],[weightL]]   = xs[w] / float( N_bkg_to_train_dict[w] * tot_xs  )   # Add to weight column
            df_bkg_test_dict[w].loc[:N_bkg_to_test_dict[w],[weightL]]     = xs[w] / float( N_bkg_to_test_dict[w] * tot_xs  )   # Add to weight column
            
            df_bkg_train_list.append( df_bkg_train_dict[w][ : N_bkg_to_train_dict[w] ] )
            df_bkg_test_list.append(  df_bkg_test_dict[w][  : N_bkg_to_test_dict[w]  ] )
  
        # Combine all HT bins
        df_bkg              = pd.concat( df_bkg_train_list , ignore_index=True )
        df_test_bkg         = pd.concat( df_bkg_test_list , ignore_index=True )
        # ~~~~~~~~ Fixing random state for reproducibility
        # Shuffle the events
        np.random.seed(random_seed)                     
        df_bkg              = df_bkg.iloc[np.random.permutation(len(df_bkg))]
        df_test_bkg         = df_test_bkg.iloc[np.random.permutation(len(df_test_bkg))]
        # ~~~~~~~~ Set up data split
        if   test_mode == 1:
            #if in test mode: use all available signal events
            df_sig[weightL]                  = 1 / float(N_available_sgn)#( N_sgn_av )
            df_test_sig                      = df_sig[:]
            df_sig                           = df_sig[:] # Only there for the return: should not be used!!
        elif test_mode == 0:
            #if in train mode: split up the data for training and testing
            df_sig[weightL]                      = 0
            df_sig.loc[:N_sgn_to_test, weightL]  = 1 / float( N_sgn_to_test )
            df_sig.loc[N_sgn_to_test:, weightL]  = 1 / float( N_sgn_to_train )

            df_test_sig                      = df_sig[:N_sgn_to_test]
            df_sig                           = df_sig[len(df_test_sig):]  

    else:
        print '>>>>>>>>>>>>>>>>>>>> Background data not enough: Please change config~'    
        exit()
    return df_bkg, df_sig, df_test_bkg, df_test_sig
















###############
#   carefull  #
###############
def SplitData_New(
                   df_bkg_dict       ,\
                   df_sig            ,\
                   N_bkg_dict        ,\
                   bkg_name_dict     ,\
                   xs                ,\
                   bkg_test_multiple ,\
                   bkg_multiple      ,\
                   N_available_bkg   ,\
                   N_available_sgn   ,\
                   train_test_ratio  ,\
                   random_seed       ,\
                   #Thrsd   = 50000   ,\
                   Thrsd   = 300000  ,\
                   weightL = 'weight',\
                   N_available_sgn_model = 0):
    N_bkg_to_use_dict   = {}
    N_bkg_to_train_dict = {}
    df_bkg_test_list    = []
    df_bkg_train_list   = []
    tot_xs              = 0 

    if N_available_sgn_model != 0:
        #check if it's testing mode: if yes use the parameters for training
        N_sgn_av        = N_available_sgn
        N_available_sgn = N_available_sgn_model

    #calculate the total cross section
    for w in bkg_name_dict:
        tot_xs += xs[w]
    #print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>total cross section: ', str(tot_xs)
    
    thrd                = (bkg_test_multiple * int( N_available_sgn * (1-train_test_ratio) )) + int(N_available_sgn * train_test_ratio) * bkg_multiple 
 
    #check if number of background events is enough 
    if N_available_bkg >= thrd:
        N_sgn_to_test       = int( N_available_sgn * (1-train_test_ratio) )
        N_sgn_to_train      = N_available_sgn - N_sgn_to_test                
        N_bkg_to_test       = int( N_sgn_to_test * bkg_test_multiple )
        N_bkg_to_train      = int( N_sgn_to_train * bkg_multiple )          

        N_taken             = 0
        for w in bkg_name_dict:   
            #if #background from a specific HT bin is smaller than Thrsd, take all the events in that bin 
            if N_bkg_dict[w] < Thrsd:
                N_taken                   += N_bkg_dict[w]   
                N_bkg_to_use_dict[w]      = N_bkg_dict[w]
                N_bkg_to_train_dict[w]    = int( N_bkg_dict[w] * train_test_ratio )
            else: 
                pass

        bkg_needed            = N_bkg_to_train + N_bkg_to_test
        bkg_train_ratio       = N_bkg_to_train/float(bkg_needed) 
        N_to_take             = bkg_needed - N_taken
        N_available_bkg_left  = N_available_bkg - N_taken
        for w in bkg_name_dict:
            if N_bkg_dict[w] < Thrsd:
                pass 
            else:
                #decide how many events should be used(train and test): proportional to the relative portion of a specificHT bin out of the total #available background events
                N_bkg_to_use_dict[w]      = int(  N_bkg_dict[w] * ( N_to_take/float(N_available_bkg_left) )  )         
                N_bkg_to_train_dict[w]    = int(  N_bkg_dict[w] * ( int(bkg_train_ratio*N_to_take)/float(N_available_bkg_left) )  )


            if N_bkg_to_train_dict[w] == 0:
                #error prompt if the relative portion of the HT bin is too small
                print '>>>>>>Zero division>>>>Please check if number of higher HT QCD-samples are too large?'
                N_bkg_to_train_dict[w]    = int( N_bkg_dict[w] * train_test_ratio )
            if N_bkg_to_use_dict[w]   == 0:
                N_bkg_to_use_dict[w]      = N_bkg_dict[w]

            #add a 'weight' column to the dataframe
            df_bkg_dict[w][weightL]   = 0   #add weight column
            #add weights to cope with the relative cross section and to compensate the number of events being used for training
            df_bkg_dict[w].loc[:N_bkg_to_train_dict[w],[weightL]]   = xs[w] / float( N_bkg_to_train_dict[w] * tot_xs  )   #add to weight column
            df_bkg_dict[w].loc[N_bkg_to_train_dict[w]:N_bkg_to_use_dict[w],[weightL]]   = xs[w] / float( (N_bkg_to_use_dict[w]-N_bkg_to_train_dict[w]) * tot_xs  )   #add to weight column
            #"""
            df_bkg_train_list.append( df_bkg_dict[w][ : N_bkg_to_train_dict[w] ] )
            df_bkg_test_list.append( df_bkg_dict[w][ N_bkg_to_train_dict[w] : N_bkg_to_use_dict[w] ] )
        #df_sig[weightL]     = 1 / float( N_sgn_to_train )                                   #add weight column
        #combine all HT bins
        df_bkg              = pd.concat( df_bkg_train_list , ignore_index=True )
        df_test_bkg         = pd.concat( df_bkg_test_list , ignore_index=True )
        #~~~~~~~~Fixing random state for reproducibility
        #shuffle the events
        np.random.seed(random_seed)                     
        df_bkg              = df_bkg.iloc[np.random.permutation(len(df_bkg))]
        df_test_bkg         = df_test_bkg.iloc[np.random.permutation(len(df_test_bkg))]
        #~~~~~~~~set up data split
        if   N_available_sgn_model != 0:
            #if in test mode: use all available signal events
            df_sig[weightL]                  = 1 / float( N_sgn_av )
            df_test_sig                      = df_sig[:]
            df_sig                           = df_sig[:] #only there for the return: should not be used!!
        elif N_available_sgn_model == 0:
            #if in train mode: split up the data for training and testing
            df_sig[weightL]                  = 0
            #df_sig[weightL][:N_sgn_to_test]  = 1 / float( N_sgn_to_test )
            #df_sig[weightL][N_sgn_to_test:]  = 1 / float( N_sgn_to_train )
            df_sig.loc[:N_sgn_to_test, weightL]  = 1 / float( N_sgn_to_test )
            df_sig.loc[N_sgn_to_test:, weightL]  = 1 / float( N_sgn_to_train )

            df_test_sig                      = df_sig[:N_sgn_to_test]
            df_sig                           = df_sig[len(df_test_sig):]  

    else:
        print '>>>>>>>>>>>>>>>>>>>>Background data not enough!'    
        exit()
    return df_bkg, df_sig, df_test_bkg, df_test_sig










###############
#             #
###############
def CountAttrInList(ColumnLabelDictSkimed,AttrToUse):
    outputArray = []
    for cc in ColumnLabelDictSkimed:
        if cc in AttrToUse:    outputArray.append(ColumnLabelDictSkimed[cc])
    outputArray.sort()
    return outputArray


###############
#             #
###############
def nKinListGen(numOfJets,oldList,prefix='J'):
    nL = []
    for i in range(numOfJets):
        for stri in oldList:
            nL.append( prefix+str(i+1)+stri )
    return nL



###############
#             #
###############
def CutBaseBenchmark(df_test_orig,inDict,JetPrfx_bkg,refAttr='pt',isSigAttrStr='is_signal',weightAttrStr='weight'):
    refAttrLabel = JetPrfx_bkg + refAttr
    tt           = df_test_orig.copy()
    sg           = tt[isSigAttrStr]==1
    bg           = tt[isSigAttrStr]==0

    BA_l = {}
    #pick out events that satisfiy the cut
    for iAttr, iList in inDict.iteritems():
        iAttr = 'J1'+iAttr
        if   iList[0] == '<':    BA_l[iAttr] = tt[JetPrfx_bkg+iAttr] < iList[1]
        elif iList[0] == '>':    BA_l[iAttr] = tt[JetPrfx_bkg+iAttr] > iList[1] 

    pos       = tt[refAttrLabel]
    pos_sgn   = tt[weightAttrStr]
    pos_bkg   = tt[weightAttrStr]
    n_pos     = tt[weightAttrStr]
    for iAttr, iList in inDict.iteritems():
        iAttr     = 'J1'+iAttr
        pos       = pos[ BA_l[iAttr] ]          #events that pass the selection(all the cuts)
        pos_sgn   = pos_sgn[ BA_l[iAttr] ]      #signal events that pass the selection(all the cuts) 
        pos_bkg   = pos_bkg[ BA_l[iAttr] ]      #background events that pass the selection(all the cuts)
        n_pos     = n_pos[ BA_l[iAttr] ]        #see below
    pos_sgn   = pos_sgn[ sg ]
    pos_bkg   = pos_bkg[ bg ]
    n_pos     = float( n_pos.sum() )            #sum up the weights

    n_sgn        = float( tt[weightAttrStr][sg].sum() )    #sum of weights from all signal
    n_bkg        = float( tt[weightAttrStr][bg].sum() )    #sum of weights from all background
    n_pos_sgn    = float( pos_sgn.sum() )                  #sum of weights of signal events that pass the selection
    n_pos_bkg    = float( pos_bkg.sum() )                  #sum of weights of background events that pass the selection
    sgn_eff      = np.divide( n_pos_sgn , n_sgn )          
    fls_eff      = np.divide( n_pos_bkg , n_bkg )
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Benchmark:'
    print 'num of total test events: ',tt[refAttrLabel].count()
    print "num of signals          : ",n_sgn
    print 'num of background       : ',n_bkg
    print "num of pos events       : ",n_pos
    print "num of pos bkg          : ",n_pos_bkg
    print "num of pos sgn          : ",n_pos_sgn
    print "true positive rate      : ",sgn_eff
    print "false positive rate     : ",fls_eff
    return sgn_eff, fls_eff



###################
# Cut Base Points #
###################
def CutBaseBenchmarkNew(df_test_orig,inDict,JetPrfx_bkg,refAttr='pt',isSigAttrStr='is_signal',weightAttrStr='weight'):
    refAttrLabel = JetPrfx_bkg + refAttr
    tt           = df_test_orig.copy()
    sg           = tt[isSigAttrStr]==1
    bg           = tt[isSigAttrStr]==0

    BA_l = {}
    #pick out events that satisfiy the cut
    for iAttr, iList in inDict.iteritems():
        iAttr = 'J1'+iAttr
        if   iList[0] == '<':    BA_l[iAttr] = tt[JetPrfx_bkg+iAttr] < iList[1]
        elif iList[0] == '>':    BA_l[iAttr] = tt[JetPrfx_bkg+iAttr] > iList[1]

    pos       = tt[refAttrLabel]
    pos_sgn   = tt[weightAttrStr]
    pos_bkg   = tt[weightAttrStr]
    n_pos     = tt[weightAttrStr]
    for iAttr, iList in inDict.iteritems():
        iAttr     = 'J1'+iAttr
        pos       = pos[ BA_l[iAttr] ]          #events that pass the selection(all the cuts)
        pos_sgn   = pos_sgn[ BA_l[iAttr] ]      #signal events that pass the selection(all the cuts) 
        pos_bkg   = pos_bkg[ BA_l[iAttr] ]      #background events that pass the selection(all the cuts)
        n_pos     = n_pos[ BA_l[iAttr] ]        #see below
    pos_sgn   = pos_sgn[ sg ]
    pos_bkg   = pos_bkg[ bg ]
    n_pos     = float( n_pos.sum() )            #sum up the weights

    n_sgn        = float( tt[weightAttrStr][sg].sum() )    #sum of weights from all signal
    n_bkg        = float( tt[weightAttrStr][bg].sum() )    #sum of weights from all background
    n_pos_sgn    = float( pos_sgn.sum() )                  #sum of weights of signal events that pass the selection
    n_pos_bkg    = float( pos_bkg.sum() )                  #sum of weights of background events that pass the selection
    sgn_eff      = np.divide( n_pos_sgn , n_sgn )
    fls_eff      = np.divide( n_pos_bkg , n_bkg )
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Benchmark:'
    print 'num of total test events: ',tt[refAttrLabel].count()
    print "num of signals          : ",n_sgn
    print 'num of background       : ',n_bkg
    print "num of pos events       : ",n_pos
    print "num of pos bkg          : ",n_pos_bkg
    print "num of pos sgn          : ",n_pos_sgn
    print "true positive rate      : ",sgn_eff
    print "false positive rate     : ",fls_eff
    #return tt[weightAttrStr][sg], tt[weightAttrStr][bg], pos_sgn, pos_bkg#sgn_eff, fls_eff

    df_sg = tt[weightAttrStr][sg]
    df_bg = tt[weightAttrStr][bg]
    df_pos_sgn = pos_sgn
    df_pos_bkg = pos_bkg  
    bin_i = 0
    bin_f = 1
#def CutBaseROC(df_sg, df_bg, df_pos_sgn, df_pos_bkg):
    h_cut_pre_tpr = TH1F('h_cut_pre_tpr' , 'hist_cut_pre_tpr'  , 1, bin_i, bin_f)
    h_cut_pos_tpr = TH1F('h_cut_pos_tpr' , 'hist_cut_pos_tpr'  , 1, bin_i, bin_f)

    h_cut_pre_fpr = TH1F('h_cut_pre_fpr' , 'hist_cut_pre_fpr'  , 1, bin_i, bin_f)
    h_cut_pos_fpr = TH1F('h_cut_pos_fpr' , 'hist_cut_pos_fpr'  , 1, bin_i, bin_f)

    root_numpy.fill_hist(h_cut_pre_tpr   , df_sg      , df_sg)
    root_numpy.fill_hist(h_cut_pos_tpr   , df_pos_sgn , df_pos_sgn)

    root_numpy.fill_hist(h_cut_pre_fpr   , df_bg      , df_bg)
    root_numpy.fill_hist(h_cut_pos_fpr   , df_pos_bkg , df_pos_bkg)

    g_cut_tpr = GAE()
    g_cut_fpr = GAE()
    g_cut_tpr.Divide(h_cut_pos_tpr, h_cut_pre_tpr, "cl=0.683 b(1,1) mode")
    g_cut_fpr.Divide(h_cut_pos_fpr, h_cut_pre_fpr, "cl=0.683 b(1,1) mode")

    g_size_cut  = 1

    x        = Double()
    y        = Double()
    x_s      = Double()
    y_s      = Double()

    arr_x    = np.zeros(g_size_cut)
    arr_y    = np.zeros(g_size_cut)
    arr_x_s  = np.zeros(g_size_cut)
    arr_y_s  = np.zeros(g_size_cut)

    for i in xrange( g_size_cut ):
        g_cut_fpr.GetPoint(i,x,y)
        arr_x[i]   = x
        arr_y[i]   = y

        g_cut_tpr.GetPoint(i,x_s,y_s)
        arr_x_s[i] = x_s
        arr_y_s[i] = y_s

    buffer_l   = g_cut_fpr.GetEYlow()
    buffer_l.SetSize(g_size_cut)
    arr_l      = np.array(buffer_l, copy=True)

    buffer_h   = g_cut_fpr.GetEYhigh()
    buffer_h.SetSize(g_size_cut)
    arr_h      = np.array(buffer_h, copy=True)

    buffer_l_s   = g_cut_tpr.GetEYlow()
    buffer_l_s.SetSize(g_size_cut)
    arr_l_s      = np.array(buffer_l_s, copy=True)

    buffer_h_s   = g_cut_tpr.GetEYhigh()
    buffer_h_s.SetSize(g_size_cut)
    arr_h_s      = np.array(buffer_h_s, copy=True)
    print len(arr_h)
    print len(arr_l)

    print 'TPR: ', arr_y_s
    print 'FPR: ', arr_y
    print arr_l_s
    print arr_l
    print arr_h_s
    print arr_h
    out_dict            = {}
    out_dict['tpr']     = arr_y_s[0]
    out_dict['fpr']     = arr_y[0]
    out_dict['tpr_e_l'] = arr_l_s[0]
    out_dict['fpr_e_l'] = arr_l[0]
    out_dict['tpr_e_h'] = arr_h_s[0]
    out_dict['fpr_e_h'] = arr_h[0]

    return out_dict















#if __name__ == '__main__':
#  print nKinListGen(3, ['pt','mass','energy'])






















# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Old Lines:


"""
###############
#             #
###############
def SplitData(df_bkg_dict       ,\
              df_sig            ,\
              N_bkg_dict        ,\
              bkg_name_dict     ,\
              xs                ,\
              bkg_test_multiple ,\
              bkg_multiple      ,\
              N_available_bkg   ,\
              N_available_sgn   ,\
              train_test_ratio  ,\
              random_seed       ,\
              #Thrsd   = 50000   ,\
              Thrsd   = 900000,\
              weightL = 'weight'):
    N_bkg_to_use_dict   = {}
    N_bkg_to_train_dict = {}
    df_bkg_test_list    = []
    df_bkg_train_list   = []
    tot_xs              = 0 
    for w in bkg_name_dict:
        tot_xs += xs[w]
    thrd                = (bkg_test_multiple * int( N_available_sgn * (1-train_test_ratio) )) + int(N_available_sgn * train_test_ratio) * bkg_multiple 

    if N_available_bkg >= thrd:
        N_sgn_to_test       = int( N_available_sgn * (1-train_test_ratio) )
        N_sgn_to_train      = N_available_sgn - N_sgn_to_test                
        N_bkg_to_test       = int( N_sgn_to_test * bkg_test_multiple )
        N_bkg_to_train      = int( N_sgn_to_train * bkg_multiple )          
        for w in bkg_name_dict:   
            if N_bkg_dict[w] < Thrsd:
                N_bkg_to_use_dict[w]      = N_bkg_dict[w]
                N_bkg_to_train_dict[w]    = int( N_bkg_dict[w] * train_test_ratio )
            else:
                N_bkg_to_use_dict[w]      = int(  N_bkg_dict[w] * ( (N_bkg_to_train + N_bkg_to_test)/float(N_available_bkg) )  )         
                N_bkg_to_train_dict[w]    = int(  N_bkg_dict[w] * ( (N_bkg_to_train)/float(N_available_bkg) )  )

            if N_bkg_to_train_dict[w] == 0:
                print '>>>>>>Zero division>>>>Please check if number of higher HT QCD-samples are too large?'
                N_bkg_to_train_dict[w]    = int( N_bkg_dict[w] * train_test_ratio )
            if N_bkg_to_use_dict[w]   == 0:
                N_bkg_to_use_dict[w]      = N_bkg_dict[w]

            df_bkg_dict[w][weightL]   = xs[w] / float( N_bkg_to_train_dict[w] * tot_xs  )   #add weight column
            df_bkg_train_list.append( df_bkg_dict[w][ : N_bkg_to_train_dict[w] ] )
            df_bkg_test_list.append( df_bkg_dict[w][ N_bkg_to_train_dict[w] : N_bkg_to_use_dict[w] ] )
        df_sig[weightL]     = 1 / float( N_sgn_to_train )                                   #add weight column
        df_bkg              = pd.concat( df_bkg_train_list , ignore_index=True )
        df_test_bkg         = pd.concat( df_bkg_test_list , ignore_index=True )
        #~~~~~~~~Fixing random state for reproducibility
        np.random.seed(random_seed)                     
        df_bkg              = df_bkg.iloc[np.random.permutation(len(df_bkg))]
        df_test_bkg         = df_test_bkg.iloc[np.random.permutation(len(df_test_bkg))]
        #~~~~~~~~set up data split
        df_test_sig         = df_sig[:N_sgn_to_test]
        df_sig              = df_sig[len(df_test_sig):]  

    else:
        print '>>>>>>>>>>>>>>>>>>>>Background data not enough!'    
        N_bkg_to_train      = int( N_available_bkg * train_test_ratio )     #for training 
        N_bkg_to_test       = N_available_bkg - N_bkg_to_train
        N_sgn_to_train      = int( N_bkg_to_train / float(bkg_multiple) )   #for training
        N_sgn_to_test       = int( N_bkg_to_test / float(bkg_test_multiple) )
        for w in bkg_name_dict:
            N_bkg_to_use_dict[w]      = int(  N_bkg_dict[w] * ( (N_bkg_to_train + N_bkg_to_test)/float(N_available_bkg) )  ) #only there for symmetry
            N_bkg_to_train_dict[w]    = int(  N_bkg_dict[w] * ( (N_bkg_to_train)/float(N_available_bkg) )  )
            df_bkg_dict[w][weightL]   = xs[w] / float( N_bkg_to_train_dict[w] * tot_xs  )   #add weight column
            df_bkg_list.append( df_bkg_dict[w][:N_bkg_to_use_dict[w]] )
        df_sig[weightL]     = 1 / float( N_sgn_to_train )                                   #add weight column
        #print 1 / float( N_sgn_to_train )
        df_bkg              = pd.concat( df_bkg_list , ignore_index=True ) 
        #~~~~~~~~Fixing random state for reproducibility
        np.random.seed(random_seed)                     
        df_bkg              = df_bkg.iloc[np.random.permutation(len(df_bkg))]
        #~~~~~~~~set up data split
        df_test_bkg         = df_bkg[:N_bkg_to_test]
        df_bkg              = df_bkg[len(df_test_bkg):]
        df_test_sig         = df_sig[:N_sgn_to_test]
        df_sig              = df_sig[len(df_test_sig):(  len(df_test_sig)+1  +  N_sgn_to_train  )]
        print '>>>>>>>>>>>>>>>Signal sample not fully used!'    
    return df_bkg, df_sig, df_test_bkg, df_test_sig









###############
#             #
###############
def CutBaseBenchmarkTwo(df_test_orig,benchmarkAttr1,benchmarkAttr2,benchmarkThrd1,benchmarkThrd2,JetPrfx_bkg,refAttr='pt',isSigAttrStr='is_signal',weightAttrStr='weight'):
    refAttrLabel = JetPrfx_bkg + refAttr
    tt           = df_test_orig.copy()         #copy the dataframe for testing
    sg           = tt[isSigAttrStr]==1         #pick out signal events
    bg           = tt[isSigAttrStr]==0         #pick out background events
    BA1_l        = tt[JetPrfx_bkg+benchmarkAttr1] < benchmarkThrd1    
    BA2_l        = tt[JetPrfx_bkg+benchmarkAttr2] < benchmarkThrd2
    pos          = tt[refAttrLabel][BA1_l][BA2_l]
    pos_sgn      = tt[weightAttrStr][BA1_l][BA2_l][sg]
    pos_bkg      = tt[weightAttrStr][BA1_l][BA2_l][bg]
    n_sgn        = float( tt[weightAttrStr][sg].sum() )
    n_bkg        = float( tt[weightAttrStr][bg].sum() )
    n_pos        = float( tt[weightAttrStr][BA1_l][BA2_l].sum() )
    n_pos_sgn    = float( pos_sgn.sum() )
    n_pos_bkg    = float( pos_bkg.sum() )
    sgn_eff      = np.divide( n_pos_sgn , n_sgn )
    fls_eff      = np.divide( n_pos_bkg , n_bkg )
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Benchmark:'
    print 'num of total test events: ',tt[refAttrLabel].count()
    print "num of signals          : ",n_sgn
    print 'num of background       : ',n_bkg
    print "num of pos events       : ",n_pos
    print "num of pos bkg          : ",n_pos_bkg
    print "num of pos sgn          : ",n_pos_sgn
    print "true positive rate      : ",sgn_eff
    print "false positive rate     : ",fls_eff
    return sgn_eff, fls_eff
"""




