#TODO add momentum
import numpy as np
from ..activation.util import sigmoid,sigmoidDerivative
from ..losses import LOSS
class SGD:
	"""
	This class implements Sochastic Gradient Descent algorithm for optimizing neural nets
		
	"""
	def __init__(self,X,Y,n_epoch,learning_rate=10e-4,shuffle=True,batch_size=15,back_bool=True):
		"""
			INPUT:
				X	=	feature vector 
				Y	=	target vector
				n_epoch	=	number of training steps to be performed
				learning_rate	=	learning rate(default = 10e-4)
				shuffle	=	To perform shuffle over training data or not(default=True)
				batch_size	=	size of the mini batch for each epoch
				back_bool	=	to perform backprop or not(default = True)
		"""
		self.feat=X
		self.target=Y
		self.epoch=n_epoch
		self.lr=learning_rate
		self.back_bool=back_bool
		self.batch_size=batch_size
		self.train_data=np.asarray(list(zip(self.feat,self.target)))
		self.LD=len(self.train_data)
	def make_batches(self,batch_size):
		"""makes batches

			INPUT:
				batch_size	=	batch_size
		"""
		ite=0
		#self.train_data=np.asarray(list(zip(self.ob.feat,self.ob.target)))
		#LD=len(self.train_data)
		w=lambda x:(x,x+batch_size) if (x+batch_size)<self.LD else (x,x+(self.LD-x))
		q=lambda x:w(x) if x==0 else w(x[1])
		size_index_list=[]
		while(1):
			ite=q(ite)
			if ite==(self.LD,self.LD):break
			else:size_index_list.append(ite)
		batches=[self.train_data[_[0]:_[1]] for _ in size_index_list]
		return np.asarray(batches)		
	def optimize(self,weights_init,biases_init,momentum=True):
		"""optimizes weights and biases"""
		self.weights=weights_init
		self.biases=biases_init
		for i in range(self.epoch):
			np.shuffle(self.train_data)
			batches=self.make_batches(self.batch_size)
			for i in batches:self._get_grads(i)
		return self.weights,self.biases
	def _get_grads(self,mini_batch):
		"""
			updates network weights and biases
			
			INPUT:
				mini_batch	=	batch of training data
		"""
		random_weights=[np.zeros(i.shape) for i in self.weights]
		random_biases=[np.zeros(j.shape) for j in self.biases]
		for f,t in mini_batch:
			dr_b,dr_w=self.backprop(f,t)
			random_weights=[i+j for i,j in zip(random_weights,dr_w)]
			random_biases=[i+j for i,j in zip(random_biases,dr_b)]
		self.weights=[i-(self.lr/len(mini_batch))*j for i,j in zip(self.weights,random_weights)]
		self.biases=[i-(self.lr/len(mini_batch))*j for i,j in zip(self.biases,random_biases)]
	def backprop(self,x,y):
		"""
			performs backpropagation
			
			INPUT:
				x	=	input feature vector
				y	=	output target vector
		"""
		u_w=[np.zeros(w.shape) for w in self.weights]
		u_b=[np.zeros(b.shape) for b in self.biases]
		weight_sum_list,act_list=self._forward_pass(x,self.weights,self.biases)
		return self._backward_pass(weight_sum_list,act_list,u_w,u_b)
	def _forward_pass(self,in_,weights,biases):
		"""
			performs forward pass
		
			INPUT:
				in_	=	input feature vector
				weights	=	weight vector
				biases	=	bias vector
			returns weighted sum list and activation list
		"""
		weight_sum_list=[]
		act_list=[]
		for w,b in zip(weights,biases):
			weight_sum=np.dot(in_,weights)+biases
			weight_sum_list.append(weight_sum)
			act_list.append(sigmoid(weight_sum))
		return weight_sum_list,act_list
	def _backward_pass(self,z_l,a_l,y,w_v,b_v):
		"""
			performs backward pass
			
			INPUT:
				z_l	=	weighted sum  list
				a_l	=	activation list
				y	=	target vector
				w_v	=	weight vector 
				b_v	=	bias vector
		
			returns derivative vector (bias and weights)
		"""
		d_L=LOSS.categorical_cross_entropy(a_l[-1],y,model="nn")*sigmoidDerivative(z_l[-1])
		b_v[-1]=d_L
		w_v[-1]=np.dot(d_L,a_l[-2].transpose())
		for i in range(2,self.layers):
			w=z_l[-i]
			a=sigmoidDerivative(w)
			d_L=np.dot(self.weights[-i+1].transpose(),d_L)*a
			b_v[-i]=d_L
			w_v[-i]=np.dot(d_L,a_l[-i-1].transpose())
		return b_v,w_v

					

				
									
				
			
				
		
			
		
