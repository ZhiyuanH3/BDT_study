#!/bin/python2.7
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name bd
#SBATCH --output /home/hezhiyua/logs/BDT-%j.out
#SBATCH --error /home/hezhiyua/logs/BDT-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de


from os import system as act
import argparse

pars = argparse.ArgumentParser()

#pars.add_argument('-a',action='store',type=str,help='argument')
pars.add_argument('--mass',action='store',type=str,help='argument')


args = pars.parse_args()

a = args.a
m = args.mass

#act('cd /home/hezhiyua/desktop/BDT_study/sbatch/')

act('cd /home/hezhiyua/desktop/BDT_study/')
#act('python t.py' + ' -b ' + a)

act('python bdtTrain_new_test.py ' + ' --mass ' + str(m))



#python bdtTrain_new_test.py --attr 





