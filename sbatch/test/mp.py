#!/bin/env python

#SBATCH --partition=all
#SBATCH --job-name=mtp
#SBATCH --time=01:00:00
#SBATCH --nodes=1

#SBATCH --output /home/hezhiyua/logs/mtp-%j.out
#SBATCH --error  /home/hezhiyua/logs/mtp-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de
"""
#!/bin/python2.7
#SBATCH --exclusive
"""
import multiprocessing as mp
import sys
import os
from   time import time as tm
# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

LL = range(100000000)



ll = []
def work(x):
    #print x*2
    return x*2




# get number of cpus available to job
try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = mp.cpu_count()
# create pool of ncpus workers
p = mp.Pool(ncpus)
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#cpu: ', str(ncpus)




A=tm()
p.map(work, LL)
B=tm()
print '>>>>>>>>>>>>>>>>>>>>>>>> Time: ', str(B-A)




A=tm()
for i in LL:
    work(i)
B=tm()
print '>>>>>>>>>>>>>>>>>>>>>>>> Time: ', str(B-A)

#print ll
























"""
#!/usr/bin/python2.7

#SBATCH -n 5 # 5 cores

import sys
import os
import multiprocessing

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def hello():
    print("Hello World")

pool = multiprocessing.Pool() 
jobs = [] 
for j in range(len(10)):
    p = multiprocessing.Process(target = run_rel)
    jobs.append(p)
    p.start() 
"""


