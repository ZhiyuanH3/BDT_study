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
pars.add_argument('--trnl'   ,action='store',type=str,help='train life time')
pars.add_argument('--tstl'   ,action='store',type=str,help='test life time' )
pars.add_argument('--train'  ,action='store',type=str,help='train model')
pars.add_argument('--kin'    ,action='store',type=str,help='kinematics'  )
pars.add_argument('--inputs' ,action='store',type=str,help='model inputs')

args        = pars.parse_args()

trnm        = args.trnm
tstm        = args.tstm
trnl        = args.trnl
tstl        = args.tstl
train       = args.train
kin         = args.kin
inputs      = args.inputs

path        = '/home/hezhiyua/desktop/BDT_study/'
script_name = 'bdtTrain_new_test.py' 
main_str    = 'python '+path+script_name

test        = 0#1

ll          = 500#5000#2000#1000#500#100#[100,500,1000,2000,5000]
mm          = 60#50#40#30#20#40#60#40#60#50#40#30#20    

if test == 0:
    combi_str   = ' --kin '+kin+' --inputs '+inputs
else        :
    combi_str   = ' --kin '+kin+' --inputs '+inputs


if test == 0:
    
    """
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training Mode:
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> whether to train bdt:
    fix_str     = ' --train '+'1'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> choose which trained model:
    train_str   = ' --trnm '+str(mm) + ' --trnl '+str(trnl)
    train_str   = train_str+' --tstm '+str(mm) + ' --tstl '+str(trnl)    
    #train_str   = ' --trnm '+str(trnm) + ' --trnl '+str(ll)
    #train_str   = train_str+' --tstm '+str(trnm) + ' --tstl '+str(ll)    


    """
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Tesging Mode:
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> whether to train bdt:
    fix_str     = ' --train '+'0'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> choose which trained model:
    train_str   = ' --trnm '+str(mm) + ' --trnl '+str(ll)
    # for testing mass
    train_str   = train_str+' --tstm '+str(tstm) + ' --tstl '+str(ll)
    # for testing life time
    #train_str   = train_str+' --tstm '+str(mm) + ' --tstl '+str(tstl)
    #"""

if test == 0:
    act(main_str+fix_str    +    train_str+combi_str)















# single test
if test == 1:
    fix_str     = ' --train '+'0'#'1'
    #act(main_str+fix_str    +    ' --trnm 60 --tstm 50 --trnl 500 --tstl 500 '+combi_str)
    #act(main_str+fix_str    +    ' --trnm 40 --tstm 50 --trnl 500 --tstl 500 '+combi_str)
    #act(main_str+fix_str    +    ' --trnm 40 --tstm 60 --trnl 500 --tstl 500 '+combi_str)

    #act(main_str+fix_str    +    ' --trnm 50 --tstm 20 --trnl 500 --tstl 500 '+combi_str)
    #act(main_str+fix_str    +    ' --trnm 50 --tstm 30 --trnl 500 --tstl 500 '+combi_str)
    act(main_str+fix_str    +    ' --trnm 50 --tstm 40 --trnl 500 --tstl 500 '+combi_str)





