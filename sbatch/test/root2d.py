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
"""
if len(sys.argv) != 3:
    print "Provide infile and outfile"
    sys.exit()
"""

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
  #print xi
  #print np.array(zi).shape
  return xi, yi, zi

Npix   = int(40)#int(128)#int(512)#int(32) 
imgpix = int(40)#int(128)#int(512)#int(32)


path    = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/'
pathOut = '/beegfs/desy/user/hezhiyua/backed/dustData/crab_folder_v2/test/'
Fname   = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC.root'
fin   = TFile(path + Fname)
fout  = TFile(pathOut + 'for2d.root','recreate')
tin   = fin.Get('ntuple/tree')
tout  = TTree('tree44','tree44')
b1    = tin.GetBranch('PFCandidates')

img = np.zeros(imgpix**2, dtype=float)

imgVec = ROOT.std.vector('float')()
tout.Branch('imgv', imgVec)

tout.Branch('img', img, 'img[' + str(imgpix**2) + ']/D')

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

    if len(e) == 0: continue
    

    xi, yi, zi = periodic(phi, eta, e)

    #--------define roi
    zroi = zi[ Npix : int(2 * Npix) ]
    xroi = xi[ Npix : int(2 * Npix) ]
    yroi = yi[ Npix : int(2 * Npix) ]
    
    #print np.array(zroi).shape

    #--------align
    maximap  = (  zroi == filters.maximum_filter( zroi , size=(3,3) )  ) #3,3
    #print maximap
    maximapi = maximap.nonzero()[0] 
    #print maximapi 
    #print maximap.nonzero()
    maximapj = maximap.nonzero()[1]
    maximapz = zroi[maximap]
    maskp    = np.argsort(maximapz)[::-1]
    #print np.argsort(maximapz)
    #print maskp
    p1     = [ maximapi[maskp][0] , maximapj[maskp][0] ]
    #print maximapi[maskp]
    p2     = [ maximapi[maskp][1] , maximapj[maskp][1] ]
    p3     = [ maximapi[maskp][2] , maximapj[maskp][2] ]
    shift  = [ int(3 * Npix / 2)-p1[0]-1 , int(Npix/2)-p1[1]-1 ]
    zp     = scipy.ndimage.interpolation.shift(zi, shift)
    zp     = zp[ Npix : int(2 * Npix) ]
    
    if p2[0]-p1[0] < -int(Npix/2) :
        p2[0]+=Npix
    if p2[0]-p1[0] > int(Npix/2) :
        p2[0]-=Npix
    if p3[0]-p1[0] < -int(Npix/2) :
        p3[0]+=Npix
    if p3[0]-p1[0] > int(Npix/2) :
        p3[0]-=Npix

    center = np.matrix([ xi[Npix+p1[0],p1[1]],
                         yi[Npix+p1[0],p1[1]] ])
    second = np.matrix([ xi[Npix+p2[0],p2[1]],
                         yi[Npix+p2[0],p2[1]] ])
    third  = np.matrix([ xi[Npix+p3[0],p3[1]],
                         yi[Npix+p3[0],p3[1]] ])

    #--------rotation
    ex    = np.matrix([[1,0]])
    #print ex
    theta = arccos(( (-center+second)*ex.T/norm(center-second)))[0,0]
    if p2[0]<p1[0]:
        theta*=-1.
    theta+=np.pi/2

    zp = scipy.ndimage.interpolation.rotate(zp,theta*180./np.pi,reshape=False)
    #--------flip
    thirdp=(np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta),  np.cos(theta)]]).dot((third-center).T)).T
    if thirdp[0,0] < -0.001:
        zp=np.fliplr(zp) 
    zp = zp[Npix/2-imgpix/2:Npix/2+imgpix/2,Npix/2-imgpix/2:Npix/2+imgpix/2]
    
    #print '----------------------------------'
    #print zp.shape


    image = zp.flatten()
    #print len(image)

    imgVec.clear()
    for j in range(len(image)):
      img[j]=image[j]
      imgVec.push_back(image[j])
    tout.Fill()
    
   
        #--------plot
    if 0:    
        f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2)
        im = ax1.imshow(zroi,origin='lower',extent=[-1.,1.,-1.,1.])
        ax2.imshow(zi,origin='lower',extent=[-1.,1.,-3.,3.],aspect='auto')
        ax2.scatter([center[0,0]],[center[0,1]], marker='o', color='r', s=10)
        ax2.scatter([second[0,0]],[second[0,1]], marker='o', color='magenta', s=10)
        ax2.scatter([third[0,0]],[third[0,1]], marker='o', color='black', s=10)
        
        ax3.imshow(zp,origin='lower',extent=[-1.,1.,-1.,1.])
        
        ax4.imshow(zp,origin='lower',extent=[-1.,1.,-1.,1.],interpolation='nearest')
        ax1.set_xlabel("$ \eta' $")
        ax1.set_ylabel("$ \phi' $")
        ax2.set_xlabel("$ \eta' $")
        ax2.set_ylabel("$ \phi' $")
        ax3.set_xlabel("$ \eta'' $")
        ax3.set_ylabel("$ \phi'' $")
        ax4.set_xlabel("$ \eta'' $")
        ax4.set_ylabel("$ \phi'' $")
        ax1.xaxis.labelpad = 20
        ax2.xaxis.labelpad = 20
        ax3.xaxis.labelpad = 20
        ax4.xaxis.labelpad = 20
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.02, 0.7])
        cb = f.colorbar(im, cax=cbar_ax)
        cb.set_label("E")
        f.subplots_adjust(hspace=.5)
        #f.show()
        #f.savefig("%d.pdf" %jentry)

tout.Write()
fout.Close()

fin.Close()

timeStop = tt.time()
print str(timeStop - timeStart)+'sec' 

