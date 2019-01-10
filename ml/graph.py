from .layer import layer
from .optimizers import optimizer
from ml import losses	
class Graph():
	def __init__(self):
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
		X.resize([X.shape[0],X.shape[1],1])
		print("training_shape:",X.shape)
		Graph.opt=optimizer.Optimizer(X=X,Y=Y)
		self.opt
		Graph.__dict__["opt"].optimize()
	def next_layer(self,l):return self.history[self.history.index(l)+1]
	def prev_layer(self,l):return self.history[self.history.index(l)-1]
	def predict(self,X):
		pred=X
		for i in self.history:
			pred=i(pred)
		return pred

