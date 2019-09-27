from os import system as act
from time import sleep as slp


A = 'python '
B = '/home/hezhiyua/desktop/BDT_study/'
C = 'bdt_train.py'

D = ' --train ' + '1'
E = ' --kin '   + '1'#'0'
F = ' --inputs '+ 'find2b'#'full'#'2best'
G = ' --trnm '  + '30'
H = ' --trnl '  + '500'
I = ' --tstm '  + '30'
J = ' --tstl '  + '500'


att_lst = []




K = ' --attr1 ' + 
L = ' --attr2 ' + 


#mass_list = [20,30,40,50]
#ctau_list = [500,1000,2000,5000]
#mass_list = [50]#[40]#[50]#[50]
#ctau_list = [5000]#[2000]#[500]

"""
for mm in mass_list:
    mm = str(mm)
    G  = ' --trnm '  + mm
    I  = ' --tstm '  + mm
    for ct in ctau_list:
        ct = str(ct)
        H  = ' --trnl '  + ct
        J  = ' --tstl '  + ct 

        act(A+B+C+D+E+F+G+H+I+J)
        slp(1*1)
"""
