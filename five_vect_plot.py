from __future__ import division
from ROOT import ROOT, gROOT, TDirectory, TFile, gFile, TBranch, TLeaf, TTree
from ROOT import TText, TPaveLabel, TLatex, TGraphErrors, TLine, gPad
from ROOT import TH1, TH1F, TH2F, TChain, TCanvas, TLegend, gStyle
from array import array
import math
from timeit import default_timer as timer

start= timer()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gStyle.SetOptStat(0)
#path0 = "/beegfs/desy/user/hezhiyua/pfc_test/raw_data/Skim/data/data_lola/fiveVect/"
#path1 = "/beegfs/desy/user/hezhiyua/pfc_test/plots/"
path0 = '/home/brian/datas/pfc_test/temp/llp_test/Skim/'
path1 = '/home/brian/'    

ct_dep = 0 #1 for ct dependence comparison
cut_on = 0 #1 to apply cuts
twoD = 0 # 2D plot option: 0 --> 1D
CHS = 0 # CHS jet option: 0 --> off
critical_line = 0 #draw red line 
number_of_bin = 100
num_of_jets = 1
#life_time = ['0','0p05','0p1','1','5','10','25','50','100','500','1000','2000','5000','10000']
#life_time_float = [0.001,0.05,0.1,1,5,10,25,50,100,500,1000,2000,5000,10000]
#len_of_lt = len(life_time)

if ct_dep == 0:
    channel = {
           
           #'t#bar{t}':'TT_TuneCUETP8M2T4_13TeV-powheg-pythia8.root',
           #'QCD':'QCD_HT50To100_1j_skimed.root',
           'QCD' : 'QCD_HT300To500_pfc_1j_skimed.root',
           'VBF-500mm-40GeV' : 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root' 
           #'QCD':'QCD_HT100To200_pfc_1j_skimed.root',
           #'VBF-500mm-40GeV':'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root' 
           #'VBF-500mm-40GeV':'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_4mj_skimed.root',
           #'VBF-0mm-40GeV':'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-0_TuneCUETP8M1_13TeV-powheg-pythia8.root',
           #'ZH-0mm-40GeV':'ZH_HToSSTobbbb_ZToLL_MH-125_MS-40_ctauS-0_TuneCUETP8M1_13TeV-powheg-pythia8.root',
   	   #'H#rightarrowb#bar{b}':'VBFHToBB_M-125_13TeV_powheg_pythia8.root'
          }
elif ct_dep == 1:
    channel = {}
    for lt in life_time:
        channel['ct' + lt] = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-' + lt + '_TuneCUETP8M1_13TeV-powheg-pythia8'+'_4j_skimed'+'.root'
    channel['QCD'] = 'QCD_HT100To200_1j_skimed.root'
    #channel['QCD'] = '/VBFHToBB_M-125_13TeV_powheg_pythia8.root'
    legends = 'SGN(VBF)'
    legendb = 'BKG(QCD)'

#attr = ['pfc1_energy']
#attr = ['pfc1_ifTrack']
attr = ['energy']

#attr = ['dR_q1','dR_q2','dR_q3','dR_q4']
#attr = ['chf','nhf', 'phf', 'elf', 'muf', 'chm', 'cm', 'nm']
#attr = ['pt', 'eta', 'phi', 'CSV', 'chf', 'nhf', 'phf', 'elf', 'muf', 'chm', 'cm', 'nm']
#attr = ['nhf','chf']
#attr = ['chm']
#attr_dict = {'pt':'p_{T}', 'eta':'#eta', 'phi':'#phi', 'CSV':'Combined Secondary Vertex(CSV)', 'chf':'Charged Hadron Fraction', 'nhf':'Neutral Hadron Fraction', 'phf':'Photon Fraction', 'elf':'Electron Fraction', 'muf':'Muon Fraction', 'chm':'Charged Hadron Multiplicity', 'cm':'Charged Multiplicity', 'nm':'Neutral Multiplicity'}
attr_dict = {'pfc1_energy':'pfc1_energy','pfc1_ifTrack':'pfc1_ifTrack'}
####################################generating list with 10 Jets
def jet_list_gen(n):
    jl = []
    for ii in range(1,n+1): 
        if CHS == 0:
            #jl.append('Jet' + "%s" %ii)
            jl.append('Jet' + "%s" %ii + 's')

        elif CHS == 1:
            jl.append('CHSJet' + "%s" %ii)
    return jl
####################################generating list with 10 Jets
jet = jet_list_gen(num_of_jets)

#++++++++++++++++++++++++++++cuts+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
def cut_dict_gen(cut_name):
    cut_dict = {}
    cut_dict['pt'] =  '(' + cut_name + 'pt' + '>' + '15' + ')' # + '&&' + '(' + cut_name + 'pt' + '>' + '70' + ')' + '&&' + '(' + cut_name + 'pt' + '<' + '75' + ')'  
       
    cut_dict['eta'] = '(' + cut_name + 'eta' + '<' + '2.4' + ')' + '&&' + '(' + cut_name + 'eta' + '>' + '-2.4' + ')'
    cut_dict['dR'] = '(' + cut_name + 'dR_q1' + '<' + '0.4' + ')' + '&&' + '(' + cut_name + 'dR_q2'+ '>=' + '0.4' + ')' + '&&' + '(' + cut_name + 'dR_q3'+ '>=' + '0.4' + ')' + '&&' + '(' + cut_name + 'dR_q4'+ '>=' + '0.4' + ')'
    cut_dict['chf'] = ''
    #cut_dict['GenBquark'] = '(GenBquark1.pt>15)&&(GenBquark1.eta<2.4)&&(GenBquark1.eta>-2.4)' 
    return cut_dict
########################################################################
###################################################################################################
def cutting_gen(pref):
    for i in jet:
        cuts = cut_dict_gen( pref + i + '.'  )
    cuttings_sgn = cuts['pt'] + '&&' + cuts['eta'] + '&&' + cuts['dR']  #+ '&&' + cut_dict['GenBquark']
    cuttings_bkg = cuts['pt'] + '&&' + cuts['eta'] 
    return cuttings_bkg, cuttings_sgn
###################################################################################################
cutting_bkg, cutting_sgn = cutting_gen('')
if cut_on == 0:
    cutting_bkg, cutting_sgn = '', ''

###################################################################################################
def cut_tex_gen(cut):
    inf = float(88888888)
    cut_str = cut.replace('(','').replace(')','').replace('&&','&').replace('<=','#leq').replace('phi','#phi').replace('eta','#eta').replace('>=','#geq') 
    cut_str = cut_str.replace('Jet1.','').replace('GenBquark','Gbq').replace('pt','p_{T}').replace('dR_','#DeltaR-')
    cut_str_list = cut_str.split('&')
    bound = {}
    for ct in cut_str_list:       
        if '#leq' in ct:             
            at, n = ct.split('#leq')
        elif '#geq' in ct:          
            at, n = ct.split('#geq')
        elif '<' in ct:            
            at, n = ct.split('<')
        elif '>' in ct:            
            at, n = ct.split('>')  
        bound[at] = {}
        bound[at]['LB'] = []
        bound[at]['UB'] = []
        bound[at]['syb'] = []
        bound[at]['B'] = []
    for ct in cut_str_list:    
        if '#leq' in ct: 
            syb = '#leq'
            at, num = ct.split('#leq')
        elif '#geq' in ct:
            syb = '#geq'
            at, num = ct.split('#geq')
        elif '<' in ct:
            syb = '<'
            at, num = ct.split('<')
        elif '>' in ct:
            syb = '>'
            at, num = ct.split('>')
                
        if syb not in bound[at]['syb']:
            bound[at]['syb'].append(syb)
        
        if syb == '#geq' or syb == '>':
            bound[at]['LB'].append( float(num) )
            bound[at]['UB'].append( inf )
        elif syb == '#leq' or syb == '<':    
            bound[at]['LB'].append( -inf ) 
            bound[at]['UB'].append( float(num) )                 
    ct_text = {}
    w = 0        
    for a in bound:    
        bound[a]['B'].append( max(bound[a]['LB']) ) 
        bound[a]['B'].append( min(bound[a]['UB']) )
        #print bound
        if -inf == float( min(bound[a]['B']) ):
            bb = a + ' ' + bound[a]['syb'][0] + ' ' + str( max(bound[a]['B']) )
        elif inf is float( max(bound[a]['B']) ):    
            bb = a + ' ' + bound[a]['syb'][0] + ' ' + str( min(bound[a]['B']) )
        else:
            if '>' in bound[a]['syb']: 
                bb = str( min(bound[a]['B']) ) + ' < ' + a 
                if '<' in bound[a]['syb']:    
                    bb = bb + ' < ' + str( max(bound[a]['B']) )
                else:    
                    bb = bb + ' #leq ' + str( max(bound[a]['B']) )
            else:
                bb = str( min(bound[a]['B']) ) + ' #leq ' + a
                if '<' in bound[a]['syb']:    
                    bb = bb + ' < ' + str( max(bound[a]['B']) )           
                else:    
                    bb = bb + ' #leq ' + str( max(bound[a]['B']) )                            
        ct_text[a] = TLatex(.77, .51 - 0.04*w, bb)  
        ct_text[a].SetNDC()
        ct_text[a].SetTextSize(0.03)
        w += 1
    return ct_text    
###################################################################################################
if cut_on == 0:
    cut_text = {}
    cut_text['no cut'] = TLatex(.77, .51 - 0.04*0, 'no cuts')
    cut_text['no cut'].SetNDC()
    cut_text['no cut'].SetTextSize(0.03)
elif cut_on == 1:
    cut_text = cut_tex_gen(cutting_sgn) 

print('---------cut:')
print(cutting_sgn)



entry = {'entries': ''}
#++++++++++++++++++++++++++++cuts+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

####################
def file_dict_gen(path,chann):
    fd = {}
    for cc in channel:
         fd[cc] = TFile(path + chann[cc],"r")
    return fd
####################
file_dict = file_dict_gen(path0,channel)

plotrange = {}
tree = {}
hist = {}
hist_CHS = {}
if ct_dep == 0:    
    ##############################
    for cc in channel:
        hist[cc] = {}
        hist_CHS[cc] = {}
    ##############################
elif ct_dep == 1:     
    mean ={}        #dictionary to hold all mean values
    errors ={}      #dictionary to hold all errors
    entries_after_cut = {}
    yy ={}          #dictionary to hold all arrays of y values for TGraph
    ey ={}          #dictionary to hold all arrays of errors for TGraph
    ##############################
    for cc in channel:
        hist[cc] = {}
        hist_CHS[cc] = {} # to be opt
        mean[cc] ={}
        errors[cc] ={}
        entries_after_cut[cc] ={}
    ##############################
    entries_after_cut['QCD'] = {}
    entries_after_cut['sgn'] = {}
    yy['QCD'] ={} 
    yy['sgn'] ={}       
    ey['QCD'] ={} 
    ey['sgn'] ={} 
    #######################################
    x = array( 'd' )        # array to plot for TGraph
    ex = array( 'd' )
    for ll in life_time_float:
        x.append( ll )
        ex.append( 0.0  )
    #######################################


####################################################################################################################
def write_1(var,sample,cuts):
    for s in attr:
        if 'QCD' in sample:
            color1 = 3                 #880+1 #400+3	6 8 634 1
        elif 't#bar{t}' in sample:
            color1 = 2  
        elif 'H#rightarrowb#bar{b}' in sample:
            color1 = 7                 #800+10 860
        elif 'VBF' in sample:
            color1 = 4
        elif 'ZH' in sample:
            color1 = 6

        if 'energy' in s:
            h_par = [number_of_bin,0,20]  
        elif 'px' in s:
            h_par = [number_of_bin,-300,300]
        elif 'py' in s:
            h_par = [number_of_bin,-300,300]
        elif 'pz' in s:
            h_par = [number_of_bin,-300,300]
        elif 'ifTrack' in s:
            h_par = [number_of_bin,-2,2] 
  

        




        tree[sample] = file_dict[sample].Get('tree44')

        tb  = tree[sample].GetBranch(var)
        NoE = tb.GetEntries() 
        
        #histogram for chf plot
        #hist[sample][s] = TH1F(sample+s, '; %s; events' %s , 100, 0, 1)
        #hist[sample][s].Sumw2()
        #histogram for num of pf candidates plot
        hist[sample][s] = TH1F(sample+s, '; %s; events' %s , 50, 0, 50)
        hist[sample][s].Sumw2()

        NoE = 10000
        print "____________________________________________________________"
        for j in range(NoE):
            tb.GetEntry(j)
            chf    = 0
            E_tot  = 0
            emptyCounter = 0
            for kk in range(40):
                tlenergy = tb.GetLeaf('pfc' + str(kk+1) + '_' + 'energy')
                energyL  = tlenergy.GetValue()
                #print energyL
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~pt
                tlpx = tb.GetLeaf('pfc' + str(kk+1) + '_' + 'px')
                pxL  = tlpx.GetValue()
                tlpy = tb.GetLeaf('pfc' + str(kk+1) + '_' + 'py')
                pyL  = tlpy.GetValue()
                pt = pow( ( pow(pxL,2) + pow(pyL,2) ) , 0.5 )
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~pt
                
                print 'pt: ' + str(pt)
                print 'E: ' + str(energyL)
                E_tot    += energyL   

            for ii in range(40): 
                tlenergy = tb.GetLeaf('pfc' + str(ii+1) + '_' + 'energy')
                tlc = tb.GetLeaf('pfc' + str(ii+1) + '_' + 'ifTrack')
                energyL  = tlenergy.GetValue()
                ifTrackL = tlc.GetValue()
                chf += energyL * ifTrackL

                if ifTrackL == -1:
                    emptyCounter += 1  
                    print 'E(when empty):'
                    print energyL
            chf = chf/float(E_tot)

            tl3 = tb.GetLeaf('chf')   
            chfL = tl3.GetValue()

            print '#empty' + str(emptyCounter)
            print '++++++++++++++++'
            
            if abs(chf - chfL) > 0.1:
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!'
                print 'chf calculated:'
                print chf
                print 'chf given:'
                print chfL
           

            #plot chf
            #hist[sample][s].Fill(chf)
            #plot num of pf candidates
            hist[sample][s].Fill(40-emptyCounter)
            print ""
            print str(  float(100 * j) / NoE  ) + "% done!"
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!"
      


     
        """
        hist_o = TH1F('o', '; %s; events' %s , h_par[0], h_par[1], h_par[2])
        tree[sample].Project('o', var + '.' + 'pfc1_' + s, cuts )   
                         
        for i in range(38):
            #hist_o = TH1F('o'+str(i), '; %s; events' %s , h_par[0], h_par[1], h_par[2])
            #tree[sample].Project('o'+str(i), var + '.' + 'pfc' + str(i+1) + '_' + s, cuts )
            

            hist_temp = TH1F('t'+str(i), '; %s; events' %s , h_par[0], h_par[1], h_par[2])  
            tree[sample].Project('t'+str(i), var + '.' + 'pfc' + str(i+2) + '_' + s, cuts )
            hist_o.Add(hist_temp,hist_o)
            
            #hist_o.clear()
            #hist_temp.Delete()
            #delete hist_temp 
        """
        #hist[sample][s] = TH1F(sample+s, '; %s; events' %s , h_par[0], h_par[1], h_par[2])
        #hist[sample][s].Sumw2()
        #hist[sample][s] = hist_chf
        #hist[sample][s] = hist_o
        
        #tree[sample].Project(sample+s, var + '.' + s, cuts ) 
        #tree[sample].Project(sample+s, var + '.' + 'pfc2_energy', cuts )
        #tree[sample].Project(sample+s, var + '.' + 'pfc3_energy', cuts )
        #tree[sample].Project(sample+s, var + '.' + 'pfc4_energy', cuts )
        #tree[sample].Project(sample+s, var + '.' + 'pfc5_energy', cuts )







                  
        
        normalizationFactor = float(hist[sample][s].Integral())
        
        if normalizationFactor != 0: 
            normalizationFactor = 1 / normalizationFactor
            
            hist[sample][s].Scale( normalizationFactor )
               
        else:
            print("zero denominator!")

        entr = tree[sample].GetEntries(cuts)
        
        if ct_dep == 0:
            entry['entries'] = '[entries:' + str(entr) + ']'
            hist[sample][s].SetLineColor(color1)
            hist[sample][s].SetLineWidth(3)
            #hist[sample][s].SetTitle( attr_dict[s] )
            hist[sample][s].SetTitle( 'Number of PF-Candidates' )

            #hist_CHS[sample][s].SetLineColor(color1+44)
            #hist_CHS[sample][s].SetLineWidth(3)

            #hist[sample][s].SetTitleSize(0.4,'t')  
            plotrange[s] =  max( plotrange[s] , hist[sample][s].GetMaximum() + hist[sample][s].GetRMS() * normalizationFactor )
            print( 'Entries:' )  			
            print( hist[sample][s].GetEntries() )
        elif ct_dep == 1:
            entries_after_cut[sample][s] = tree[sample].GetEntries( cuts ) # to be optimized
            errors[sample][s] = hist[sample][s].GetStdDev() #saving errors of the histogram
            #errors[sample][s] = hist[sample][s].GetRMS()
            mean[sample][s] = hist[sample][s].GetMean()     #saving means of the histogram
            plotrange[s] =  max( plotrange[s] , mean[sample][s] + hist[sample][s].GetRMS() )
####################################################################################################################

###########################################################################
def write_2(sample):
    #for cc in channel:
    for s in attr:
        yy[sample][s] = array( 'd' )     #declaring the yy array
        ey[sample][s] = array( 'd' )     #declaring the ey array    
        for ll in enumerate(life_time):
            if sample == 'QCD':
                yy[sample][s].append( mean['QCD'][s] )
                ey[sample][s].append( errors['QCD'][s] )
            else:
                yy[sample][s].append( mean['ct'+ll[1]][s] )
                ey[sample][s].append( errors['ct'+ll[1]][s] )
###########################################################################

##########################################################
def plot_2(var,cuts):
    for s in attr:
        c1 = TCanvas("c1", "Signals", 1200, 800)
        c1.SetTopMargin(0.12)
        c1.SetLeftMargin(0.14)
        c1.SetRightMargin(0.24)
        c1.cd()
        c1.SetGrid()
        gStyle.SetTitleFontSize(0.08)
        if ct_dep == 0:
            if s in ('elf', 'muf', 'chm', 'cm', 'nm'):
                c1.SetLogx()
            for cc in channel:
                #hist[cc][s].SetMaximum(0.44)
                hist[cc][s].Draw('colz same')
                if CHS == 1:
                    hist_CHS[cc][s].Draw('colz same')
            legend = TLegend(0.76, 0.56, 0.99, 0.88)
            legend.SetHeader( entry['entries'] )
            for cc in channel:
                legend.AddEntry(hist[cc][s],cc)
                if CHS == 1:
                    legend.AddEntry(hist_CHS[cc][s],cc + 'CHS')
            legend.Draw()
            for ct in cut_text:
                cut_text[ct].Draw()
            
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Line for critical value
            """
            if s == 'chm':
                l = TLine(12.7,0.0,12.7,0.09)
            elif s == 'chf':
                l = TLine(0.48,0.0,0.48,0.027)
            elif s == 'nhf':
                l = TLine(0.16,0.0,0.16,0.29)
            l.SetLineColor(2)
            l.SetLineWidth(3)
            if critical_line == 1: 
                l.Draw('same')
            """


            c1.Print(path1 + s + var + cuts.replace('(','_').replace(')','_').replace('&&','_').replace('>','LG').replace('<','LS').replace('=','EQ').replace('.','P').replace('-','N').replace('Jet','J').replace('GenBquark','GBQ') + ".pdf")
            
        elif ct_dep == 1:
            eac0 = str( entries_after_cut['ct0'][s] )
            c1.SetLogx()
            #gr = TGraph( len_of_lt , x , yy['sgn'][s] )
            gr = TGraphErrors( len_of_lt , x , yy['sgn'][s] , ex , ey['sgn'][s] )
            gr.SetMarkerSize(1.5)
            gr.SetMarkerStyle(1)
            gr.GetYaxis().SetTitleOffset(1.6)
            gr.SetLineColor(4)
            gr.SetLineWidth(4)
            gr.SetTitle('mean ' + s )
            gr.GetXaxis().SetTitle('decaying length (mm)')
            gr.GetYaxis().SetTitle('mean normalized number of events')
            gr.GetXaxis().SetTitleOffset(1.4)
            gr.SetMaximum( plotrange[s] * 1.12 )
            gr.SetName('sgn')
            gr.Draw('ACP')  # '' sets up the scattering style
            gr1 = TGraphErrors( len_of_lt , x , yy['QCD'][s] , ex , ey['QCD'][s] )
            gr1.SetMarkerSize(1.0)
            gr1.SetMarkerStyle(1)
            gr.GetYaxis().SetTitleOffset(1.6)
            gr1.SetLineColor(2)
            gr1.SetLineWidth(2)
            gr1.SetName('QCD')
            #gr1.SetTitle('averaged ' + s)
            #gr1.GetXaxis().SetTitle('decaying length (mm)')
            #gr1.GetYaxis().SetTitle('mean frequency')
            gr1.Draw('CP')  # '' sets up the scattering style
            legend = TLegend(0.76, 0.56, 0.99, 0.88)
            legend.SetHeader( 'Entries: ' + eac0 )
            legend.AddEntry('QCD', legendb, 'l')
            legend.AddEntry('sgn', legends, 'l')
            legend.Draw()
            for ct in cut_text:
                cut_text[ct].Draw()
            c1.Print(path1 + 'mean_' + s + var + cuts.replace('(','_').replace(')','_').replace('&&','_').replace('>','LG').replace('<','LS').replace('=','EQ').replace('.','P').replace('-','N').replace('Jet','J').replace('GenBquark','GBQ') + ".pdf")
        c1.Update()
        c1.Close() 
        print('|||||||||||||||||||||||||||||||||||||||||||||||||||')        
##########################################################

########################################################################
def clear_hist(sample):
    for s in attr:
        if gROOT.FindObject( sample+s ) != None:
            hh = gROOT.FindObject( sample+s ) #to find the histogram
            hh.Delete()    #to delete the histogram
        if gROOT.FindObject( sample+s+'CHS' ) != None:
            hhchs = gROOT.FindObject( sample+s+'CHS' ) #to find the histogram
            hhchs.Delete()    #to delete the histogram
########################################################################

########################################################################
def set_hist_yrange():
    os = 1.15
    for cc in channel:
        for s in attr:
            hist[cc][s].SetMaximum( plotrange[s] * os )
            #hist_CHS[cc][s].SetMaximum( plotrange[s] *os )

            hist[cc][s].GetYaxis().SetTitleOffset(1.6)
            hist[cc][s].GetYaxis().SetTitle('normalized number of events')
            hist[cc][s].GetXaxis().SetTitle( s )
            
            #hist_CHS[cc][s].GetYaxis().SetTitleOffset(1.6)
            #hist_CHS[cc][s].GetYaxis().SetTitle('normalized number of events')
            #hist_CHS[cc][s].GetXaxis().SetTitle( s )
            if s == 'elf':
                hist[cc][s].SetAxisRange(0., 0.02,"Y")
                #hist_CHS[cc][s].SetAxisRange(0., 0.02,"Y")
            elif s == 'muf':
                hist[cc][s].SetAxisRange(0., 0.02,"Y")  
                #hist_CHS[cc][s].SetAxisRange(0., 0.02,"Y")        
########################################################################

########################################################################
def init_plotrange():
    for s in attr:
        plotrange[s] = 0           
########################################################################

#===========================================================================================
#===========================================================================================
init_plotrange()
if ct_dep == 0:
    for i in jet:    
        for cc in channel:
            
            if cc == "QCD":
                cutting = cutting_bkg
            else:
                cutting = cutting_sgn

            write_1(i,cc,cutting)

        set_hist_yrange()
        plot_2(i,cutting)

elif ct_dep == 1:
    for i in jet:
        for cc in channel:

            if cc == "QCD":
                cutting = cutting_bkg
            else:
                cutting = cutting_sgn

            write_1(i,cc,cutting)
        write_2('QCD')
        write_2('sgn')
        plot_2(i,cutting)

plotrange.clear()
for cc in channel:
    clear_hist(cc) 
    hist[cc].clear()
    tree.clear()

for cc in channel:
        file_dict[cc].Close()
#===========================================================================================
#===========================================================================================









end = timer() 
print("Time taken:", end-start) 
"""
###########################################################
def findDirName() 
    #file_dict['QCD']
    TIter next(file_dict['QCD'].GetListOfKeys())
    key = TKey
    while key = next():
        cl = TClass
        cl = gROOT.GetClass(key.GetClassName())
        if (cl.InheritsFrom("TDirectory")):
            dir = TDirectory
            dir = (TDirectory*)key.ReadObj()
            print( "Directory name: " + dir.GetName() )
###########################################################
"""




