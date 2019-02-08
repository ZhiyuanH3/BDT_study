from os import system as act
from time import sleep as slp


A = 'python '
B = '/home/hezhiyua/desktop/BDT_study/'
C = 'bdt_train.py'

D = ' --train ' + '1'
E = ' --kin '   + '0'
F = ' --inputs '+ 'find2b'#'full'#'2best'
#G = ' --trnm '  + '50'
#H = ' --trnl '  + '2000'
#I = ' --tstm '  + '50'
#J = ' --tstl '  + '2000'

attr_t = 'J1nSelectedTracks'#'J1cMulti'#'J1nSelectedTracks'
K = ' --attr1 ' + attr_t#'J1cHadEFrac'
L = ' --attr2 ' + attr_t#'J1cHadEFrac'

#'J1cEmEFrac'#'J1cMulti'#'J1cHadEFrac'#'J1cMulti'#'J1cHadMulti'#'J1cHadEFrac'#'J1cHadMulti'#'J1photonMulti'#'J1cHadEFrac'#'J1muEFrac'#'J1nSelectedTracks'


mass_list = [40]#[20,30,40,50]
ctau_list = [500]#[500,1000,2000,5000]

for mm in mass_list:
    mm = str(mm)
    G  = ' --trnm '  + mm
    I  = ' --tstm '  + mm
    for ct in ctau_list:
        ct = str(ct)
        H  = ' --trnl '  + ct
        J  = ' --tstl '  + ct 

        act(A+B+C+D+E+F+G+H+I+J  +K+L)
        slp(1*1)
