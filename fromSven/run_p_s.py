from os import system as act
import numpy          as np

main_str  = 'python preprocess_shihnew.py '

#pth_root  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/'
pth_root  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/nn_format/'
#pth_in    = pth_root + 'pfc_400/raw/output/test/' + '50_5000/'
#pth_in    = pth_root + 'pfc_400/raw/output/train/test_from_50_5000/' 
#pth_in    = pth_root + 'pfc_400/raw/2jets/output/test/50_5000/test_from_1j/'
#pth_in    = pth_root + 'pfc_400/raw/2jets/output/train/test_from_50_5000/'

pth_in    = pth_root + '2jets/playground/lola/train40_5000val40_2000test50_5000/'

color     = 'HE'
#color     = 'CE'
#color     = 'E'
#color     = 'CHE'

#pth_out   = pth_root + 'pfc_400/raw/output/test/50_5000/2d/' + color + '/'
#pth_out   = pth_root + 'pfc_400/raw/output/train/test_from_50_5000/2d/' + color + '/'

#pth_out   = pth_root + 'pfc_400/raw/2jets/output/test/50_5000/test_from_1j/2d/' + color + '/'
#pth_out   = pth_root + 'pfc_400/raw/2jets/output/train/test_from_50_5000/2d/' + color + '/'

pth_out   = pth_root + '2jets/playground/lola/train40_5000val40_2000test50_5000/'+'2d/'+color+'/'
act('mkdir ' + pth_out)

#in_typ    = {'test': 13266, 'val': 3981, 'train': 9588} 
#in_typ    = {'test': 5369, 'val': 5345, 'train': 16138} 
#in_typ    = {'test': 7966, 'val': 7948, 'train': 23854}

#in_typ    = {'test': 5171, 'val': 5160, 'train': 15487}
#in_typ    = {'test': 5171, 'val': 7312, 'train': 21947}


#in_typ    = {'test': 5171, 'val': 10208, 'train': 30633}
#in_typ    = {'test': 5171, 'val': 13776, 'train': 41342}

in_typ    = {'test': 8924, 'val': 18311, 'train': 25834}


rotation  = str(1)  #0#True
flip      = str(1)  #0#False

aug_mode  = 0
dR        = 0.4
n_pixels  = 36#42#41#43#44#40#28#40
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

   
