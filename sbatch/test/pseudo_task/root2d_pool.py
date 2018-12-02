#!/bin/env python

#SBATCH --partition=all
#SBATCH --job-name=mtp
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -c 70
#SBATCH --output /home/hezhiyua/logs/mtp-%j.out
#SBATCH --error  /home/hezhiyua/logs/mtp-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de





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
from time import time as tm

import multiprocessing as mp


def periodic(phi, eta, e):
  X=[]
  Y=[]
  Z=[]
  eta_max=2.5
  phi_scale=-np.pi #minus to invert axis
  eta_scale=abs(phi_scale) 
  for i in range(0,len(phi)):
    if (np.abs(eta[i]) < eta_scale):
        Y.append(phi[i]/phi_scale)
        X.append(eta[i]/eta_scale)
        Z.append(e[i])
        Y.append((phi[i]+2.*np.pi)/phi_scale) #??
        X.append(eta[i]/eta_scale)
        Z.append(e[i])
        Y.append((phi[i]-2.*np.pi)/phi_scale) #?
        X.append(eta[i]/eta_scale)
        Z.append(e[i])
  xil, yil = np.linspace(-1, 1, 1*Npix), np.linspace(-3, 3, 3*Npix)
  rbf = scipy.interpolate.Rbf(X, Y, Z, function='linear')
  xi, yi = np.meshgrid(xil, yil)
  zi =  rbf(xi,yi)
  zi[zi < 0]= 0.
  zi[np.abs(xi) > eta_max/eta_scale]= 0.
  return xi, yi, zi

nnn    = 40#2
Npix   = int(nnn) 
imgpix = int(nnn)

path    = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/'
pathOut = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/test/'
Fname   = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC.root'


entries = 200


def worker(jentry):
    fin   = TFile(path + Fname)
    tin   = fin.Get('ntuple/tree')
    b1    = tin.GetBranch('PFCandidates')

    phi = []
    eta = []
    e   = []
    ientry = tin.GetEntry(jentry)
    b1.GetEntry(jentry)

    energy1   = b1.FindBranch('PFCandidates.energy')
    phi1      = b1.FindBranch('PFCandidates.phi')
    eta1      = b1.FindBranch('PFCandidates.eta')
    jetindex1 = b1.FindBranch('PFCandidates.jetIndex')

    n_depth = 100
    for num in range(0,n_depth): 
        if energy1.GetValue(num,1) == 0.: continue
        if eta1.GetValue(num,1) == 0.: continue
        if phi1.GetValue(num,1) == 0.: continue
        phi.append(phi1.GetValue(num,1))
        eta.append(eta1.GetValue(num,1))
        e.append(energy1.GetValue(num,1))
    if len(e) == 0: return#continue
    
    xi, yi, zi = periodic(phi, eta, e)
    #q.put(np.max(e))
    return np.max(e)






L={}
def split_job(n_task, L, n_cpu=12):
    sub_range = int(n_task/float(n_cpu))
    res       = n_task - sub_range*n_cpu
    for i in range(n_cpu):
        L[i]            = {}
        L[i]['result']  = []
        if i == n_cpu - res:
            cut_i = i   
            cut = cut_i * sub_range 
        if i >= n_cpu - res:    L[i]['range']  = range(cut + (i-cut_i)*(1+sub_range), cut + (i-cut_i+1)*(1+sub_range)) 
        else               :    L[i]['range']  = range(i*sub_range, (i+1)*sub_range) 


def sub_loop(func, return_dict, ind, sub_range):
    tmp_list = []
    for jj in sub_range:
        tmp_list.append( func(jj) )  
    return_dict[ind] = tmp_list


"""
q=mp.Queue()

n_cpus = 12
record = []
split_job(100, L, n_cpus)

for j in xrange(n_cpus):
    process = mp.Process(target=sub_loop, args=(worker, L[j]['result'], L[j]['range'])  )
    process.start()
    record.append(process)
for p in record:
    p.join()
LL=[]
while not q.empty():
    LL.append(q.get())
"""
#pool = mp.Pool(12)
#pool.map(worker,range(10))

#"""
tm1 = tm()
n_cpus = 12#20#70#12
split_job(entries, L, n_cpus)
out_list = []
with mp.Manager() as manager:
    Dict = manager.dict()
    processes = []
    for i in range(n_cpus): 
        process = mp.Process(target=sub_loop, args=(worker, Dict, i, L[i]['range'])  )
        process.start()
        processes.append(process)
    """
    for p in processes:
        slp(0.4)
        p.is_alive()
    """
    for p in processes:
        p.join()
    for i in Dict.values():
        out_list += i
print out_list
tm2 = tm()
print str(tm2 - tm1)+'sec'
#"""

"""
tm1 = tm()
n_cpus = 12
split_job(100, L, n_cpus)
dd = {}
print L
for i in range(n_cpus): 
    sub_loop(worker, dd, i, L[i]['range'])
print dd
tm2 = tm()
print str(tm2 - tm1)+'sec' 
"""

#"""
tm1 = tm()
LLL=[]
for i in xrange(entries):
    LLL.append( worker(i) )
print LLL
tm2 = tm()
print str(tm2 - tm1)+'sec'
#"""



print LLL == out_list






