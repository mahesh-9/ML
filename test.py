<<<<<<< HEAD
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
=======
from ml.preprocess.util import Preprocess,normalize,Flatten
from ml.layer.layer import *
from ml.graph import *
from ml.preprocess.util import normalize,sub_mean_ch
import numpy as np
path="./ml/dataset/train/"
t_path="./ml/dataset/test/"
i=Preprocess(path)
X,Y=i.direc_to_array()
X=normalize(X)
print("training on %d examples"%(len(X)))
p=Graph()
p.add(Input(784))
p.add(DNN(1024,activation="sigmoid"))
p.add(DNN(1024,activation="sigmoid"))
p.add(DNN(10,activation="relu"))
p.get_graph
p.train(X=X,Y=Y,lr=0.1,epochs=30)
j=Preprocess(t_path)
X,Y=i.direc_to_array()
X=Flatten(normalize(X))
outs=p.predict(X)
print(outs)
>>>>>>> 81370b43bc1510cb29b08fb5441bcfaaae2bde81
