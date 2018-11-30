#!/bin/python2.7
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name bdt
#SBATCH --output /home/hezhiyua/logs/BDT-%j.out
#SBATCH --error  /home/hezhiyua/logs/BDT-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de

from   os        import system as act
import argparse

pars   = argparse.ArgumentParser()
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> flags from submit.py
pars.add_argument('--trnm'   ,action='store',type=str,help='train mass')
pars.add_argument('--tstm'   ,action='store',type=str,help='test mass' )
pars.add_argument('--train'  ,action='store',type=str,help='train model')
pars.add_argument('--kin'    ,action='store',type=str,help='kinematics'  )
pars.add_argument('--inputs' ,action='store',type=str,help='model inputs')

#[100,500,1000,2000,5000]
args   = pars.parse_args()

trnm   = args.trnm
tstm   = args.tstm
train  = args.train
kin    = args.kin
inputs = args.inputs

path        = '/home/hezhiyua/desktop/BDT_study/'
script_name = 'bdtTrain_new_test.py' 
main_str    = 'python '+path+script_name

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> whether to train bdt:
fix_str     = ' --train '+'0'
#fix_str     = ' --train '+'1'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> choose which trained model:
train_str   = ' --trnm '+str(40)




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training Mode:
#act(main_str+fix_str    +    ' --kin '+kin+' --inputs '+inputs)



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Tesging Mode:
act(main_str+fix_str    +    train_str+' --tstm '+str(tstm)+' --kin '+kin+' --inputs '+inputs)







