#!/bin/python2.7
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name bd
#SBATCH --output /home/hezhiyua/logs/BDT-%j.out
#SBATCH --error /home/hezhiyua/logs/BDT-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de


from   os        import system as act
import argparse

pars = argparse.ArgumentParser()

pars.add_argument('--trnm' ,action='store',type=str,help='train mass')
pars.add_argument('--tstm' ,action='store',type=str,help='test mass' )
pars.add_argument('--train',action='store',type=str,help='train model')

pars.add_argument('--kin'   ,action='store',type=str,help='kinematics'  )
pars.add_argument('--inputs',action='store',type=str,help='model inputs')



args = pars.parse_args()

trnm  = args.trnm
tstm  = args.tstm
train = args.train

kin    = args.kin
inputs = args.inputs


path = '/home/hezhiyua/desktop/BDT_study/'


#act( 'python ' + path + 'bdtTrain_new_test.py ' + ' --tstm ' + str(tstm) )

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training Mode:
act('python '+path+'bdtTrain_new_test.py'+' --train '+'1'+' --kin '+kin+' --inputs '+inputs)










# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Tesging Mode:





