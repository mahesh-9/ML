from ml.preprocess.util import Preprocess,normalize
from ml.layer.layer import *
from ml.graph import *
from ml.preprocess.util import normalize,sub_mean_ch
import numpy as np
path="./ml/dataset/train/"
i=Preprocess(path)
X,Y=i.direc_to_array()
X=normalize(X)
p=Graph()
p.add(Input(784))
p.add(DNN(1024,activation="relu"))
p.add(DNN(10,activation="relu"))
p.get_graph
p.train(X=X,Y=Y)
