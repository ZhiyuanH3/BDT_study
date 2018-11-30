import argparse

pars = argparse.ArgumentParser()

pars.add_argument('-b',action='store',type=str,help='test')

args = pars.parse_args()

b = args.b

print 'testing~~'
print b
