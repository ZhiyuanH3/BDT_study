#!/bin/python2.7
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --job-name bdts
#SBATCH --output /home/hezhiyua/logs/BDT-%j.out
#SBATCH --error  /home/hezhiyua/logs/BDT-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de

from os   import system as act
from time import sleep  as slp

flags                 = {}

flags['trnm']         = {}
flags['trnm']['flag'] = ' --trnm '
flags['trnm']['list'] = [20,30,40,50,60]

flags['tstm']         = {}
flags['tstm']['flag'] = ' --tstm '
flags['tstm']['list'] = [20,30,40,50,60] # range(15,65,5)

flags['trnl']         = {}
flags['trnl']['flag'] = ' --trnl '
flags['trnl']['list'] = [100,500,1000,2000,5000] 

flags['tstl']         = {}
flags['tstl']['flag'] = ' --tstl '
flags['tstl']['list'] = [100,500,1000,2000,5000]


wait_time  = 20 # Seconds


main_str   = 'sbatch bdt_batch.py '

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing Mode:
#"""
#scan_typ   = 'tstm'
#skip_point = 60#50#40#30#20
scan_typ   = 'tstl'
skip_point = 100#5000#2000#1000#500#100

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
#"""






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


























# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing:

"""
for key in flags:
    for i in flags['mass']['list']:
        act('sbatch bdt_batch.py ' + flags['mass']['flag'] + i)
"""

"""
comm_str_list = []
comm_str_dict = {}

for key, item in flags.iteritems():
    comm_str_dict[key] = []
    for i in item['list']:
        tmp_str = item['flag'] + str(i)
        comm_str_dict[key].append(tmp_str)


for key, item in comm_str_dict.iteritems():
    tmp_str = tmp_stritem  
    comm_str_list.append(item)


print comm_str_list
"""
