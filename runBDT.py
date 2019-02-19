from os import system as act
from time import sleep as slp


A = 'python '
B = '/home/hezhiyua/desktop/BDT_study/'
C = 'bdt_train.py'

D = ' --train ' + '1'
E = ' --kin '   + '0'#'1'#'0'
F = ' --inputs '+ '2best'#'full'#'2best'
#G = ' --trnm '  + '50'
#H = ' --trnl '  + '2000'
#I = ' --tstm '  + '50'
#J = ' --tstl '  + '2000'

#K = ' --attr1 ' + 'J1nSelectedTracks'#'J1npr'#'J1cMulti'#'J1nSelectedTracks'
#L = ' --attr2 ' + 'J1nSelectedTracks'#'J1cEmEFrac'#'J1cMulti'#'J1cHadEFrac'#'J1cMulti'#'J1cHadMulti'#'J1cHadEFrac'#'J1cHadMulti'#'J1photonMulti'#'J1cHadEFrac'#'J1muEFrac'#'J1nSelectedTracks'


#mass_list = [20,30,40,50]
#ctau_list = [500,1000,2000,5000]
mass_list = [50]#[40]#[50]#[50]
ctau_list = [5000]#[2000]#[500]


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
