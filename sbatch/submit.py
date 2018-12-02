from os   import system as act
from time import sleep  as slp

flags                 = {}

flags['trnm']         = {}
flags['trnm']['flag'] = ' --trnm '
flags['trnm']['list'] = []

flags['tstm']         = {}
flags['tstm']['flag'] = ' --tstm '
flags['tstm']['list'] = [20,30,40,50,60] # range(15,65,5)

wait_time  = 40


main_str   = 'sbatch bdt_batch.py '
skip_point = 40#60#30#20#60#35#25#15#40



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing Mode:
#"""
for i in flags['tstm']['list']:

    if i == skip_point: continue

    fix_str = flags['tstm']['flag'] + str(i)

    #act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' 2best')
    #slp(wait_time)
    #act(main_str+fix_str+' --kin'+' 0'+' --inputs'+' full' )
    #slp(wait_time)
    act(main_str+fix_str+' --kin'+' 1'+' --inputs'+' 2best')
    slp(wait_time)
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
