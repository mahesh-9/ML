import numpy as np
from ml import losses
from ml import epoch_an
from ml.activation import activations
from ml.layer import layer
class Optimizer:
	def __init__(self,X,Y,lr_rate=1e-3,epochs=10,batch_size=16,loss="cost"):
		self.X=X
		self.Y=Y
		self.lr=lr_rate
		self.epochs=epochs
		self.batch_size=batch_size
		self.data=np.asarray(list(zip(X,Y)))
		self.loss=loss
	def make_batches(self,batch_size):
		"""makes batches

			INPUT:
				batch_size	=	batch_size
		"""
		ite=0
		w=lambda x:(x,x+batch_size) if (x+batch_size)<len(self.data) else (x,x+(len(self.data)-x))
		q=lambda x:w(x) if x==0 else w(x[1])
		size_index_list=[]
		while(1):
			ite=q(ite)
			if ite==(len(self.data),len(self.data)):break
			else:size_index_list.append(ite)
		self.batches=[self.data[_[0]:_[1]] for _ in size_index_list]
	def optimize(self,cost="cost"):
		self.count=0
		self.make_batches(self.batch_size)
		self.loss=losses.cost
		for i in range(self.epochs):
			print("\t\t\t\trunning epoch:%d"%(i+1))
			#epoch_an.a(9209)
			for i in self.batches:self.update_on_batch(i,batch_size=self.batch_size)
	def update_on_batch(self,batch,batch_size=16):
		trainable_layers=self.graph_layers[1:]
		batch_grads_w=[np.zeros(i.get_weights.shape) for i in trainable_layers]
		batch_grads_b=[np.zeros(i.get_biases.shape) for i in trainable_layers]
		for ex in batch:
			self.count+=1
			single_grads,cpl,loss=self.update_on_single_example(ex[0],ex[1])
			batch_grads_w=[a+b for a,b in zip(batch_grads_w,single_grads)]
			batch_grads_b=[a+b for a,b in zip(batch_grads_b,cpl)]
		for i in range(len(trainable_layers)):
			trainable_layers[i].weights-=(self.lr/batch_size)*batch_grads_w[i]
			trainable_layers[i].biases-=(self.lr/batch_size)*batch_grads_b[i]
	def update_on_single_example(self,X,Y):
		gradient_updates_w=[]
		gradient_updates_b=[]
		costs_per_layer=[]
		pred=X
		trainable_layers=self.graph_layers[1:]
		for i in trainable_layers:
			pred=i(pred)
		network_loss=losses.cost(pred,Y)
		trainable_layers.reverse()
		rev=trainable_layers
		for i in range(len(rev)):
				if not costs_per_layer:
					loss=(network_loss)*(rev[i].get_activation(rev[i].get_weighted_sum,prime=True))
					grad=np.dot(loss,self.graph_i.prev_layer(rev[i]).get_activations.T)
					gradient_updates_w.append(grad)
					gradient_updates_b.append(loss)
					rev[i].loss=loss
					costs_per_layer.append(loss)	
				else:
					res=np.dot(self.graph_i.next_layer(rev[i]).get_weights.T,self.graph_i.next_layer(rev[i]).loss)
					loss=np.dot(self.graph_i.next_layer(rev[i]).get_weights.T,self.graph_i.next_layer(rev[i]).loss)*(rev[i].get_activation(rev[i].get_weighted_sum,prime=True))
					costs_per_layer.append(loss)
					rev[i].loss=loss
					gradient_updates_b.append(loss)
					if isinstance(self.graph_i.prev_layer(rev[i]),layer.Input):
						grad=grad=np.dot(loss,X.T)
					else:
						grad=np.dot(loss,self.graph_i.prev_layer(rev[i]).get_activations.T)
					gradient_updates_w.append(grad)
		return  reversed(gradient_updates_w),reversed(costs_per_layer),network_loss





	def __get__(self,g_inst,o):
		self.graph_i=g_inst
		if not g_inst==None:
			self.graph_layers=self.graph_i.__dict__["history"]
