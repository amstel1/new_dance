from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pickle
from numpy.random import seed
seed(10)
d = defaultdict(LabelEncoder)



with open('d.pckl', 'wb') as f:
    pickle.dump(d, f)

with open('d.pckl', 'rb') as f:
    d = pickle.load(f)

# Encoding the variable
tr = X.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
tr.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
X.apply(lambda x: d[x.name].transform(x))