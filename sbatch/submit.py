from os import system as act



flags = {}
flags['mass'] = {}
flags['mass']['flag'] = ' --mass '
flags['mass']['list'] = range(15,65,5)

flags['t'] = {}
flags['t']['flag'] = ' -t '
flags['t']['list'] = ['y','n']



for i in flags['mass']['list']:
    if i==40: continue
    act('sbatch bdt_batch.py ' + flags['mass']['flag'] + str(i))


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
