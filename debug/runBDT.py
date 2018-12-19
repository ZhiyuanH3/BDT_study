from os import system as act



A = 'python '
B = '/home/hezhiyua/desktop/BDT_study/'
C = 'debug_bdt.py'

D = ' --train ' + '1'
E = ' --kin '   + '0'
F = ' --inputs '+ 'find2b'
G = ' --trnm '  + '60'
H = ' --trnl '  + '5000'
I = ' --tstm '  + '60'
J = ' --tstl '  + '5000'
K = ' --attr1 ' + 'J1nSelectedTracks'
L = ' --attr2 ' + 'J1nHadEFrac'#'J1muEFrac'#'J1nSelectedTracks'



act(A+B+C+D+E+F+G+H+I+J+K+L)
