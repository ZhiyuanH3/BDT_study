from sklearn.externals import joblib
from matplotlib        import pyplot  as plt

s = joblib.load('s.pkl')
b = joblib.load('b.pkl')

print len(s)
print len(b)
print s

#plt.hist(s['signal'].values,bins=100)
#plt.show()

plt.hist(b['signal'].values, bins=100, weights=b['weight'].values, alpha=0.5, label='background')
plt.hist(s['signal'].values, bins=100, weights=s['weight'].values, alpha=0.5, label='signal')
plt.vlines(0.6072553645074628,0,1)
plt.yscale('log', nonposy='clip')
plt.legend()
plt.show()
