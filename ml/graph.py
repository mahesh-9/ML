from .layer import layer
from .preprocess.util import Flatten
from .optimizers import optimizer
from ml import losses
import numpy as np	
class Graph():
	def __init__(self,flatten=True):
		self.in_flag=flatten
		self.history=[]
	def add(self,ly):
		if not isinstance(ly,layer.Layer):
			raise ValueError("Not a layer instance")
		else:
			if not self.history:self.history.append(ly)
			else:
				ly.set_weights(self.history[-1].get_units)
				self.history.append(ly)
	@property
	def get_graph(self):
		print("\t\tNetwork Graph:\n")
		for i in self.history:
			print(i.__class__.__name__,"\tunits:%d\tactivation:%s\n"%(i.get_units,i.get_activation.__name__))
	def train(self,X=None,Y=None,lr=1e-3,epochs=10):
		if self.in_flag:
			X=Flatten(X)
		print("training_shape:",X.shape)
		Graph.opt=optimizer.Optimizer(X=X,Y=Y,lr_rate=lr,epochs=epochs)
		self.opt
		Graph.__dict__["opt"].optimize()
	def next_layer(self,l):return self.history[self.history.index(l)+1]
	def prev_layer(self,l):return self.history[self.history.index(l)-1]
	def predict(self,X):
		class_vect=[]
		for j in X:	
			pred=j
			for i in self.history:
					pred=i(pred)
			class_vect.append(np.argmax(pred))
		return class_vect
		
		

