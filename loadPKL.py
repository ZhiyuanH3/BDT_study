from sklearn.externals import joblib
from matplotlib        import pyplot  as plt

path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/generalization_bdt/dumps/tst_40GeV_500mm_slct1_attr_full_kin1_v0/'

s = joblib.load(path+'s.pkl')
b = joblib.load(path+'b.pkl')

print len(s)
print len(b)
print s

#plt.hist(s['signal'].values,bins=100)
#plt.show()

line_pos = 0.7144224231#0.60725

weight_on = 0

if weight_on == 1:
    weight_b = b['weight'].values
    weight_s = s['weight'].values
    vline_h  = 1
else:
    weight_b = None
    weight_s = None
    vline_h  = 100000 

plt.hist(b['signal'].values, bins=100, weights=weight_b, alpha=0.5, label='background')
plt.hist(s['signal'].values, bins=100, weights=weight_s, alpha=0.5, label='signal')
plt.vlines(line_pos, 0, vline_h)
plt.yscale('log', nonposy='clip')
plt.legend()
plt.show()
