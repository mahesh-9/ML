from layer import layer
#from optimizers.SGD import *
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
	def train(self,X,Y,lr=1e-3,epochs=None):pass
	
