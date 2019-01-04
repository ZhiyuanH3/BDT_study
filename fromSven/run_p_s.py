from os import system as act

main_str = 'python preprocess_shihnew.py '

pth_in   = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forLola/h5/'
pth_out  = pth_in+'2d/'

in_typ   = {'test': 13266, 'val': 3981, 'train': 9588} 

for key, item in in_typ.iteritems():
    arg_1    = pth_in+'vbf_qcd-'+key+'-v0_40cs.h5'
    arg_2    = pth_out+'vbf_qcd-'+key+'-v0_40cs.h5'
    arg_3    = str(item)
    arg_4    = key
    
    act(main_str + ' ' + arg_1 + ' ' + arg_2 + ' ' + arg_3 + ' ' + arg_4)



