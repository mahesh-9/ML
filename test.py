from ml.preprocess.util import *
from ml.nn.network import *
path = "/home/pirate/sonu/datasets/10_cat"
i = Preprocess(path)
X,Y = i.direc_to_array()
i =brain()
i.fit(X,Y,layers=3,neurons=[784,30,10])
w,b=i.train()
print(w)
print(b)
