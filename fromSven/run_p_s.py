from os import system as act
import numpy          as np

main_str  = 'python preprocess_shihnew.py '

#pth_root  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forLola/h5/'
pth_root  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/'
#pth_in    = pth_root + 'lola/c/' #ch
#pth_in    = pth_root + 'pfc_400/large_sgn/output/'
pth_in    = pth_root + 'pfc_400/large_sgn/output/train/'


color     = 'C'
color     = 'E'

#pth_out   = pth_root+'2d/' + 'augmented/e/'#+ 'rot_270/'#'rot_90/'#'rot_0/'#'rot_180/'#'rot0flp1/'#'rot1flp1/'
#pth_out   = pth_root + '2d/' + 'no_pType/dR_test/'
#pth_out   = pth_root + '2d/' + 'no_pType/dR_test/' + color + '/'
pth_out   = pth_root + 'pfc_400/large_sgn/2d/' + color + '/'
act('mkdir ' + pth_out)


#in_typ    = {'test': 13266, 'val': 3981, 'train': 9588} 
#in_typ    = {'test': 5369, 'val': 5345, 'train': 16138} 

in_typ    = {'test': 7966, 'val': 7948, 'train': 23854}

rotation  = str(1)  #0#True
flip      = str(1)  #0#False

aug_mode  = 0
dR        = 0.4
n_pixels  = 42#41#43#44#40#28#40
n_constit = 400#40

def run_i(pth_out, rot_angle):
    for key, item in in_typ.iteritems():
        arg_list = [] 
        arg_list.append(pth_in+'vbf_qcd-'+key+'-v0_40cs.h5')
        arg_list.append(pth_out+'vbf_qcd-'+key+'-v0_40cs.h5')
        arg_list.append(str(item))
        #arg_list.append(key)
        arg_list.append(str(n_constit))
        arg_list.append(color)
        arg_list.append(rotation)
        arg_list.append(flip)
        arg_list.append(rot_angle)
        arg_list.append(str(aug_mode))
        arg_list.append(str(dR))
        arg_list.append(str(n_pixels))
        #arg_list.append(str())
        arg_str  = ' '
        for i in arg_list:
            arg_str += i + ' '
        act(main_str + ' ' + arg_str)
        
 


if aug_mode:
    n_rot          = 1#4
    ele_angle      = int(360/n_rot)
    ele_angle_r    = 2*np.pi/n_rot
    for i in range(n_rot):
        str_i      = 'rot_' + str(ele_angle*i)
        angle      = str(i*ele_angle_r)    
        
        pth_out_i  = pth_out + str_i + '/'
        act('mkdir ' + pth_out_i)
        
        run_i(pth_out_i, angle) 

else:
    run_i(pth_out, '0')

   
