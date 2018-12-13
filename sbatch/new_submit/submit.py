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
from combi import combi_index as cmbi

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
flags['tstm']['list'] = [20,30,40,50,60] # range(15,65,5)
flags['tstl']         = {}
flags['tstl']['flag'] = ' --tstl '
flags['tstl']['list'] = [500,1000,2000,5000]



flags['inputs']         = {}
flags['inputs']['flag'] = ' --inputs '
flags['inputs']['list'] = ['full']#['2best','full']

flags['kin']            = {}
flags['kin']['flag']    = ' --kin '
flags['kin']['list']    = [0]#[0,1]


wait_time  = 20 # Seconds
main_str   = 'sbatch bdt_batch.py '

flag_str  = []
leng_list = []
for key, item in flags.iteritems():
    flag_str.append(item['flag'])
    leng_list.append( len(item['list']) )
print flag_str

cmb_lst = cmbi( leng_list )
combi   = [  [ flags[k[1]]['list'][i[k[0]]] for k in enumerate(flags) ] for i in cmb_lst  ]
#print combi




"""
skip_point_str = #' --tstm 30 --tstl 500 '#' --tstm 60 --tstl 500 '#' --tstm 50 --tstl 500 '#' --tstm 40 --tstl 500 '



    for i in combi:
        out_string = main_str
        for j in enumerate(flag_str):
            out_string += j[1]+str(i[j[0]])
        if skip_point_str in out_string: continue
        print out_string
        #act(out_string)
        #slp(wait_time)
"""



#"""
skip_point_str = ' --tstm 20 --tstl 1000 '#' --tstm 30 --tstl 1000 '#' --tstm 60 --tstl 1000 '#' --tstm 30 --tstl 1000 '#' --tstm 30 --tstl 5000 '#' --tstm 60 --tstl 2000 '#' --tstm 60 --tstl 5000 '#' --tstm 20 --tstl 5000 ' 
#' --tstm 20 --tstl 500 '#' --tstm 30 --tstl 500 '#' --tstm 60 --tstl 500 '#' --tstm 50 --tstl 500 '#' --tstm 40 --tstl 500 '
#skip_point_str1 = ' --trnm 40 '
#skip_point_str2 = ' --trnl 500 '
for i in combi:
    out_string = main_str
    for j in enumerate(flag_str):
        out_string += j[1]+str(i[j[0]])
    if skip_point_str in out_string: continue
    #if skip_point_str1 in out_string: continue
    #if skip_point_str2 in out_string: continue
    print out_string
    act(out_string)
    slp(wait_time)
#"""













# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing Mode:
"""
#scan_typ   = 'tstm'
#skip_point = 60#50#40#30#20
scan_typ   = 'tstl'
skip_point = 5000#2000#1000#500

for i in flags[scan_typ]['list']:

    if i == skip_point: continue

    fix_str = flags[scan_typ]['flag'] + str(i) 

    act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' 2best')
    slp(wait_time)
    act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' full' )
    slp(wait_time)
    #act(main_str+fix_str+' --kin'+' 1'+' --inputs'+' 2best')
    #slp(wait_time)
    #act(main_str+fix_str+' --kin'+' 1'+' --inputs'+' full' ) 
    #slp(wait_time) 
"""



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training Mode:
"""
act(main_str+' --kin'+' 0'+' --inputs'+' 2best')
slp(wait_time)
act(main_str+' --kin'+' 0'+' --inputs'+' full' )
slp(wait_time)
act(main_str+' --kin'+' 1'+' --inputs'+' 2best')
slp(wait_time)
act(main_str+' --kin'+' 1'+' --inputs'+' full' )
"""


"""
#scan_typ   = 'trnm'
scan_typ   = 'trnl'

for i in flags[scan_typ]['list']:

    fix_str = flags[scan_typ]['flag'] + str(i) 

    #act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' 2best')
    #slp(wait_time)
    #act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' full' )
    #slp(wait_time)
    act(main_str+fix_str+' --kin'+' 1'+' --inputs'+' 2best')
    slp(wait_time)
    act(main_str+fix_str+' --kin'+' 1'+' --inputs'+' full' ) 
    slp(wait_time) 
"""


























