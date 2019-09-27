#!/bin/python2.7
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --job-name bdts
#SBATCH --output /home/hezhiyua/logs/BDT-%j.out
#SBATCH --error  /home/hezhiyua/logs/BDT-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de

from os    import system      as act
from time  import sleep       as slp
from combi import *

flags                 = {}
"""
flags['trnm']         = {}
flags['trnm']['flag'] = ' --trnm '
flags['trnm']['list'] = [20,30,40,50,60]
flags['trnl']         = {}
flags['trnl']['flag'] = ' --trnl '
flags['trnl']['list'] = [500,1000,2000,5000] 
"""
flags['tstm']         = {}
flags['tstm']['flag'] = ' --tstm '
flags['tstm']['list'] = [20,30,40,50]#[20,30,40,50,60] # range(15,65,5)
flags['tstl']         = {}
flags['tstl']['flag'] = ' --tstl '
flags['tstl']['list'] = [500,1000,2000,5000]


flags['inputs']         = {}
flags['inputs']['flag'] = ' --inputs '
flags['inputs']['list'] = ['full']#['2best']#['full']#['2best','full']
flags['kin'   ]         = {}
flags['kin'   ]['flag'] = ' --kin '
flags['kin'   ]['list'] = [1]#[0]#[0,1]


wait_time = 8 # Seconds
main_str  = 'sbatch bdt_batch.py '

flag_str  = []
leng_list = []
for key, item in flags.iteritems():
    flag_str.append(item['flag'])
    leng_list.append( len(item['list']) )
#print flag_str

cmb_lst   = combi_index( leng_list )
combi     = [  [ flags[k[1]]['list'][i[k[0]]] for k in enumerate(flags) ] for i in cmb_lst  ]
#print combi




#"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>> Find 2 Best Attribute-Combination:
#attr_list = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE']
attr_list = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1ecalE']
attr_2combi_list = combi_2ofN(attr_list)
attr_flag_str    = [' --attr1 ',' --attr2 ']
fix_str          = ' --train 1 --trnm 30 --trnl 500 --tstm 30 --tstl 500 --kin 1 --inputs find2b '
#fix_str          = ' --train 0 --trnm 60 --trnl 5000 --tstm 40 --tstl 500 --kin 0 --inputs find2b '
loop_list        = attr_2combi_list[:]
for i in loop_list:
    out_string = main_str + fix_str
    for j in enumerate(attr_flag_str):
        out_string += j[1]+str(i[j[0]])  
    print out_string
    act(out_string)
    slp(wait_time)
#"""


"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>> 2D Parameter space:
skp_m = [20,30,40,50]#[40]
skp_l = [500,1000,2000,5000]#[5000]
fix_str = ' --train 0 '

for m_skp in skp_m:#[40,50]:
    for l_skp in skp_l:#[1000,5000]:
        train_str      = ' --trnm '+str(m_skp)+' --trnl '+str(l_skp)+' '
        skip_point_str = ' --tstm '+str(m_skp)+' --tstl '+str(l_skp)+' '

        for i in combi:
            out_string = main_str + fix_str + train_str
            for j in enumerate(flag_str):
                out_string += j[1]+str(i[j[0]])
            if skip_point_str in out_string: continue
            #if skip_point_str1 in out_string: continue
            #if skip_point_str2 in out_string: continue
            print out_string
            act(out_string)
            slp(wait_time)
"""










