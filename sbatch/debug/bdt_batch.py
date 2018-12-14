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
pars.add_argument('--attr1' ,action='store',type=str,help='attribute1')
pars.add_argument('--attr2' ,action='store',type=str,help='attribute2')
args        = pars.parse_args()

trnm        = args.trnm
tstm        = args.tstm
trnl        = args.trnl
tstl        = args.tstl
train       = args.train
kin         = args.kin
inputs      = args.inputs
attr1       = args.attr1  
attr2       = args.attr2

path        = '/home/hezhiyua/desktop/BDT_study/'
#script_name = 'bdt_train.py' 
script_name = 'debug_bdt.py'
main_str    = 'python '+path+script_name
test        = 0


if   test == 0:
    fix_str     = ' --train ' + str(train)
    train_str   = ' --trnm '  + str(trnm)  + ' --trnl '   + str(trnl)
    test_str    = ' --tstm '  + str(tstm)  + ' --tstl '   + str(tstl)
    combi_str   = ' --kin '   + str(kin)   + ' --inputs ' + str(inputs)
    attr_str    = ' --attr1 ' + str(attr1) + ' --attr2 '  + str(attr2)
    act(main_str+fix_str+train_str+test_str+combi_str+attr_str)











# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> single test:
#elif test == 1:
#    fix_str     = ' --train '+'0'#'1'
#    act(main_str+fix_str    +    ' --trnm 20 --tstm 60 --trnl 500 --tstl 5000 '+' --kin 0 '+' --inputs full ')





