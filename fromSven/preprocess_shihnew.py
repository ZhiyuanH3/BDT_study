#!/usr/bin/env python

import time
import numpy as np
import pandas
import sys
import scipy
import scipy.ndimage.filters
import matplotlib.pyplot as plt
import os.path


# Start: takes 3 arguments after python constit2img.py .../input.h5 .../output.h5 n_events (for output-data)
#
# Info:
# input should include E, PX, PY and PZ column to work
# n_constit should be sorted by pT, highest pT first, but code should work in principal with
# unsorted data (except the debugging 2 will give wrong warnings))
#
# problem with panda if n_pixel to large! 64kb limit for column,
# roughly 2000 columns possible, 59*59 pixels was still possible for me
#
# be carefull about meaning of x and y, x is eta and phi is y
#
# Changes:
# included (complib = "blosc", complevel=5) to to_hdf
# included mass, corrected for negative values give 0
# included assert outputfile

# debugging 1
assert len(sys.argv) == 4, "need 4 input parameters,\
 python constit2img.py .../input.h5 .../output.h5 n_events"

########################################
# Settings
########################################

# input/output settings
signal_col  = "is_signal_new"
mass_col    = "mass"
#n_constit   = 200
n_constit   = 40
#batch_size  = 1000
#batch_size  = 3000
batch_size  = 500
intensity = "pT"  # or "E" what to use for filling images
n_pixel = 40

# image preprocessing options
Rotate, Flip, Norm  = True, True, True  # full preprocessing
# Rotate, Flip, Norm  = False, False, True # minimal preprocessing with norm

max_batches = float(sys.argv[3])/batch_size

# grid settings
xpixels = np.linspace(-0.58, 0.58, n_pixel)
ypixels = np.linspace(-0.7, 0.7, n_pixel)

#ypixels = np.linspace(-1.745, 1.745, n_pixel)
#xpixels = np.linspace(-2, 2, n_pixel)


#########################################

# deugging 2
assert not os.path.isfile(sys.argv[2]), "output-file already exists,\
 choose different filename or delete existing file"

# probably doesn't need a square grid any longer?
assert len(xpixels) == len(ypixels), "need a square grid"

assert float(sys.argv[3]) >= batch_size, "event number\
 should be larger than batch size"

# check if hottest constituent is close to centre afer shifting
# value depends on jet-radius of the jets
# just for debugging, consistency check, can be ignored
n_warning = 0.7

##################################

# calculate pseudorapidity of pixel entries
def eta(pT, pz):
    small = 1e-10

    small_pT = (np.abs(pT) < small)
    small_pz = (np.abs(pz) < small)
    not_small = ~(small_pT | small_pz)

    theta = np.arctan(pT[not_small]/pz[not_small])
    theta[theta < 0] += np.pi

    etas = np.zeros_like(pT)
    etas[small_pz] = 0
    etas[small_pT] = 1e-10
    etas[not_small] = np.log(np.tan(theta/2))
    return etas

#calculate phi (in range [-pi,pi]) of pixel entries
def phi(px, py):
    ''' phis are returned in rad., np.arctan(0,0)=0 -> zero constituents set to -np.pi
    '''
    phis = np.arctan2(py,px)
    phis[phis < 0] += 2*np.pi
    phis[phis > 2*np.pi] -= 2*np.pi
    phis = phis - np.pi
    return phis

#put eta-phi entries on grid
def orig_image(etas,phis,es):
    ''' gives always the larger value on the grid, eg. for xpixel = (0,1,2,3,..)  eta=1.3 -> xpixel=2,
    np.argmax gives position of first True value in array
    '''
    z = np.zeros((etas.shape[0],len(xpixels),len(ypixels)))
    in_grid = ~((etas < xpixels[0]) | (etas > xpixels[-1]) | (phis < ypixels[0]) | (phis > ypixels[-1]))
    xcoords = np.argmax(etas[:,None,:] < xpixels[None,:,None],axis=1)
    ycoords = np.argmax(phis[:,None,:] < ypixels[None,:,None],axis=1)
    ncoords = np.repeat(np.arange(etas.shape[0])[:,None],etas.shape[1],axis=1)
    z[ncoords[in_grid],ycoords[in_grid],xcoords[in_grid]] = es[in_grid]

    return z

#put eta-phi entries on grid
def orig_image2(etas,phis,es):
    ''' Alternative version of orig_image: Gives the value on grid with minimal distance,
    eg. for xpixel = (0,1,2,3,..) eta=1.3 -> xpixel=1, eta=1.6 ->xpixel=2
    '''
    z = np.zeros((etas.shape[0],len(xpixels),len(ypixels)))
    in_grid = ~((etas < xpixels[0]) | (etas > xpixels[-1]) | (phis < ypixels[0]) | (phis > ypixels[-1]))
    xcoords = np.argmin(np.abs(etas[:,None,:] - xpixels[None,:,None]),axis=1)
    ycoords = np.argmin(np.abs(phis[:,None,:] - ypixels[None,:,None]),axis=1)
    ncoords = np.repeat(np.arange(etas.shape[0])[:,None],etas.shape[1],axis=1)
    z[ncoords[in_grid],ycoords[in_grid],xcoords[in_grid]] = es[in_grid]

    return z

def print_time(msg):
    print("[%8.2f] %s"%(time.time()-time_start,msg))

def img_mom(x, y, weights, x_power, y_power):
    '''returns image momenta for centroid and principal axis'''
    return ((x**x_power)*(y**y_power)*weights).sum()

def preprocessing(x ,y, weights):
    ''' (x,y) are the coordinates and weights the corresponding values, shifts
    centroid to origin, rotates image, so that principal axis is vertical,
    flips image, so that most weights lay in (x<0, y>0)-plane.
    Method for calculating principal axis (similar to tensor of inertia):
    https://en.wikipedia.org/wiki/Image_moment
    here: y=phi, phi has modulo 2*np.pi but it's not been taken care of hear,
    so possible issues with calculating the centroid
    -> pre-shifting of events outside of this function solves the problem
    for iamge-data with Delta_phi(jet) << 2*np.pi
    '''

    # shift
    x_centroid = img_mom(x, y, weights, 1, 0) / weights.sum()
    y_centroid = img_mom(x, y, weights, 0, 1)/ weights.sum()
    x = x - x_centroid
    y = y - y_centroid

    # check if shifting worked, there can be problems with modulo variables like phi (y)
    # x and y are sorted after highest weight, 0-comp. gives hottest event
    # for Jet-like Images Centroid should be close to hottest constituen (pT-sorted arrays)
    global n_shift_phi
    global n_shift_eta
    if np.abs(x[0]) > n_warning:
        n_shift_eta += 1
    if np.abs(y[0]) > n_warning:
        n_shift_phi += 1

    if Rotate:
        # covariant matrix, eigenvectors corr. to principal axis
        u11 = img_mom(x, y, weights, 1, 1) / weights.sum()
        u20 = img_mom(x, y, weights, 2, 0) / weights.sum()
        u02 = img_mom(x, y, weights, 0, 2) / weights.sum()
        cov = np.array([[u20, u11], [u11, u02]])

        # Eigenvalues and eigenvectors of covariant matrix
        evals, evecs = np.linalg.eig(cov)

        # sorts the eigenvalues, v1, [::-1] turns array around,
        sort_indices = np.argsort(evals)[::-1]
        e_1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        e_2 = evecs[:, sort_indices[1]]

        # theta to x_asix, arctan2 gives correct angle
        theta = np.arctan2(e_1[0], e_1[1])

        # rotation, so that princple axis is vertical
        # anti-clockwise rotation matrix
        rotation = np.matrix([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation * np.stack([x,y])
        x_rot, y_rot = transformed_mat.A
    else:
        x_rot, y_rot = x, y

    # flipping
    n_flips = 0
    if Flip:
        if weights[x_rot<0.].sum() < weights[x_rot>0.].sum():
            x_rot = -x_rot
            n_flips  += 1
        if weights[y_rot<0.].sum() > weights[y_rot>0.].sum():
            y_rot = -y_rot
            n_flips += 1

    #print('number of flips: ' + str(n_flips))

    return x_rot, y_rot

def mass(E,px,py,pz):
    '''Calculating the invariant mass of the total momentum of all
    constituents, negative massses can occure! -> is set to 0'''
    mass2 = E**2-px**2-py**2-pz**2
    mass2 = np.maximum(0.,mass2)
    mass = np.sqrt(mass2)
    return mass


def process_batch(start_id):
    print_time("Loading input file (events %i to %i)"%(start_id,start_id+batch_size))
    df = pandas.read_hdf(sys.argv[1],'table',start=start_id,stop=start_id+batch_size)
    if df.shape[0] == 0:
        return False

    print_time("Extracting 4-vectors")
    feat_list =  ["E","PX","PY","PZ"]
    cols = ["{0}_{1}".format(feature,constit)
            for feature in feat_list for constit in range(n_constit)]
    
    # search for events that only contain 0 <=> E_0 = 0 (because the events are pT sorted)
    # the image preprocessing doesn't work for events with only 0 so we set E_1, PX_1, PY_1 and PZ_1 to 1
    # then the code runs without any problems and we can change the event later in the image
    df.loc[df.E_0 == 0, ['E_1', 'PX_1', 'PY_1', 'PZ_1']] = 1
    
    vec4 = np.expand_dims(df[cols],axis=-1).reshape(-1, len(feat_list), n_constit)
    isig = df[signal_col]
    # keep the event ID if it exists
    event_ID_exists = 'event_ID' in df.columns
    if event_ID_exists:
        event_ID = df['event_ID']

    print_time("Calculating pT")
    E     = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    pT    = np.sqrt(pxs**2+pys**2)

    print_time("Calculating eta")
    etas  = eta(pT,pzs)
    print_time("Calculating phi")
    phis  = phi(pxs,pys)

    print_time("Calculating the mass")
    E_tot = E.sum(axis=1)
    px_tot = pxs.sum(axis=1)
    py_tot = pys.sum(axis=1)
    pz_tot = pzs.sum(axis=1)
    j_mass = mass(E_tot, px_tot, py_tot, pz_tot)

    #pre-shifting of phi
    phis = (phis.T - phis[:,0]).T
    phis[phis < -np.pi] += 2*np.pi
    phis[phis > np.pi] -= 2*np.pi

    print_time("Preprocessing")
    if intensity == "pT":
        weights = pT
    elif intensity == "E":
        weights = E

    for i in np.arange(0,batch_size):
        etas[i,:], phis[i,:] = preprocessing(etas[i,:], phis[i,:], weights[i,:])

    # using pT instead of energy E
    print_time("Creating images")
    z_ori = orig_image2(etas,phis,pT)

    # normalize
    print_time("Normalize")
    z_new = np.zeros((z_ori.shape[0],n_pixel, n_pixel))
    if Norm:
        for i in range(z_ori.shape[0]):
            z_sum = z_ori[i,:,:].sum()
            if z_sum != 0.:
                z_new[i,:,:] = z_ori[i,:,:]/z_sum
    else:
        z_new = z_ori

    print_time("Reshaping output")
    z_out = z_new.reshape((z_new.shape[0],-1))

    print_time("Creating output dataframe")
    out_cols = (["img_{0}".format(i) for i in range(z_new.shape[1]*z_new.shape[2])]
                + [signal_col] +[mass_col])
    df_out = pandas.DataFrame(data=np.concatenate((z_out,isig[:,None],j_mass[:,None]),axis=1),
                              index=np.arange(start_id,start_id+batch_size),
                              columns=out_cols)
    
    # replace the images of the events that only contain 0 with rows that only contain 0 except the signal column
    # the order in the image didn't change so we can find these events by searching them in the original data frame
    # we didn't changed E_0 so we can search like before for events that have E_0 = 0
    out_cols_0 = (["img_{0}".format(i) for i in range(z_new.shape[1]*z_new.shape[2])] + [mass_col])
    df_out.loc[(df.E_0 == 0).values, out_cols_0] = 0
    # if the event_ID exists, put it back to the data frame
    if event_ID_exists:
        # event_ID and df_out must have the same indices
        event_ID.index = df_out.index
        df_out['event_ID'] = event_ID
    print_time("Writing output file")
    df_out.to_hdf(sys.argv[2],'table',append=(start_id!=0),format='table',complib = "blosc", complevel=5)

    return True


#--------------------------------------------------------------

#print Settings
print "----------------------------------------------------"
print "number of x_pixel (eta): " + str(len(xpixels))
print "number of y_pixel (phi): " + str(len(ypixels))
print "x_min, x_max (eta): {0:4.2f} {1:4.2f}".format(xpixels.min(), xpixels.max())
print "y_min, y_max (phi): {0:4.2f} {1:4.2f}".format(ypixels.min(), ypixels.max())
print "x: step_size (eta): {0:5.3f}".format(xpixels[1]-xpixels[0])
print "y: step_size (phi): {0:5.3f}".format(ypixels[1]-ypixels[0])
print "signal_col  = " + signal_col
print "mass_col    = " + mass_col
print "n_constit   = " + str(n_constit)
print "batch_size  = " + str(batch_size)
print "number of events = " + str(sys.argv[3])
print "Rotate, Flip, Norm  = " + str(Rotate)+ " " + str(Flip) +  " " +str(Norm)
print "For intenisty of pixel used: " + intensity
print "Output-images: {0}x{0} pixel with range: eta = {1:4.2f}, phi = {2:4.2f}".format(n_pixel, xpixels.max()-xpixels.min(), ypixels.max()-ypixels.min())
print "---------------------------------------------------"

time_start = time.time()
start_id   = 0
n_shift_phi = 0         # number of times shifting failed
n_shift_eta = 0


while process_batch(start_id):
    start_id += batch_size
    if start_id // batch_size == max_batches:
        break

if n_shift_eta != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time("Number of times eta of hottest const. was not close to origin: " + str(n_shift_eta))
if n_shift_phi != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time("Number of times phi of hottest const. was not close to origin: " + str(n_shift_phi))

print_time("Finished")

