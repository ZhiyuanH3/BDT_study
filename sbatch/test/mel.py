import numpy as np
from   numpy import (array, dot, arccos, arcsin)
from   numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import ROOT
from   ROOT import TFile, TLorentzVector, TTree
from   array import array
import sys
import time as tt
timeStart = tt.time()


path    = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/'
pathOut = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/test/'
Fname   = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC.root'
fin   = TFile(path + Fname)
fout  = TFile(pathOut + 'for2d.root','recreate')
tin   = fin.Get('ntuple/tree')
tout  = TTree('tree44','tree44')
b1    = tin.GetBranch('PFCandidates')

entries = tin.GetEntriesFast()
for jentry in xrange(entries):
    phi = []
    eta = []
    e   = []
    ientry = tin.GetEntry(jentry)
    
    print '~~~~~~~~~~~~~~~~entry'
    print jentry
    b1.GetEntry(jentry)
    
    energy1   = b1.FindBranch('PFCandidates.energy')
    phi1      = b1.FindBranch('PFCandidates.phi')
    eta1      = b1.FindBranch('PFCandidates.eta')
    jetindex1 = b1.FindBranch('PFCandidates.jetIndex')
    for num in range(0,100): #100
        #print 'jet index'
        #print jetindex1.GetValue(num,1)
        #print energy1.GetValue(num,1)
        if energy1.GetValue(num,1) == 0.: continue
        if eta1.GetValue(num,1) == 0.: continue
        if phi1.GetValue(num,1) == 0.: continue
        phi.append(phi1.GetValue(num,1))
        eta.append(eta1.GetValue(num,1))
        e.append(energy1.GetValue(num,1))
    #print phi
    #print e
    #print eta


tout.Write()
fout.Close()

fin.Close()

timeStop = tt.time()
print str(timeStop - timeStart)+'sec' 

