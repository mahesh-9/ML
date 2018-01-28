
import numpy as np
from ..activation.util import sigmoid,sigmoidDerivative,relu,reluDerivative,stable_sigmoid
from ..losses import *
class SGD:
	"""
	This class implements Sochastic Gradient Descent algorithm for optimizing neural nets
		
	"""

	def __init__(self,X,Y,n_epoch,layers,learning_rate=3,shuffle=True,batch_size=15,back_bool=True):
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
		self.layers=layers
	def make_batches(self,batch_size):
		"""makes batches

			INPUT:
				batch_size	=	batch_size
		"""
		ite=0
		w=lambda x:(x,x+batch_size) if (x+batch_size)<self.LD else (x,x+(self.LD-x))
		q=lambda x:w(x) if x==0 else w(x[1])
		size_index_list=[]
		while(1):
			ite=q(ite)
			if ite==(self.LD,self.LD):break
			else:size_index_list.append(ite)
		batches=[self.train_data[_[0]:_[1]] for _ in size_index_list]
		return batches	
	def optimize(self,weights_init,biases_init,momentum=True):
		"""optimizes weights and biases"""
		self.weights=weights_init
		self.biases=biases_init
		for i in range(self.epoch):
			print("\t\t\t|Running EPOCH:{0} |".format(i+1))
			np.random.shuffle(self.train_data)
			batches=self.make_batches(self.batch_size)
			for j in range(len(batches)):
				print("|running mini batch",j+1," |")
				self._get_grads(batches[j])
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
			w_l,a_l=self._forward_pass(f,random_weights,random_biases)
			dr_b,dr_w=self._backward_pass(w_l,a_l,t,random_weights,random_biases)
			random_weights=[i+j for i,j in zip(random_weights,dr_w)]
			random_biases=[i+j for i,j in zip(random_biases,dr_b)]
		self.weights=[w-(self.lr/len(mini_batch))*q for w,q in zip(self.weights,random_weights)]
		self.biases=[b-(self.lr/len(mini_batch))*e for b,e in zip(self.biases,random_biases)]
	def _forward_pass(self,in_,weights,biases):
		"""
			performs forward pass
			INPUT:
				in_	=	input feature vector
				weights	=	weight vector
				biases	=	bias vector
			returns weighted sum list and activation list
		"""
		act=in_
		weight_sum_list=[]
		act_list=[in_]
		for w,b in zip(self.weights,self.biases):
			weight_sum=np.matmul(w,act)+b
			weight_sum_list.append(weight_sum)
			act=stable_sigmoid(weight_sum)
			act_list.append(act)
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
		er=cost(a_l[-1],y)
		d_L=cost(a_l[-1],y)*sigmoidDerivative(z_l[-1])
		print("Error:",er)
		d_L=categorical_cross_entropy(a_l[-1],y,model="nn")*sigmoidDerivative(z_l[-1])
		#d_L = cost(a_l[-1],y)*sigmoidDerivative(z_l[-1])
		b_v[-1]=d_L
		temp1=np.reshape(d_L,(d_L.shape[0],1))
		temp2=np.reshape(a_l[-2],(1,a_l[-2].shape[0]))
		w_v[-1]=np.dot(temp1,temp2)
		for i in range(2,self.layers):
			w=z_l[-i]
			a=sigmoidDerivative(w)
			d_L=np.dot(self.weights[-i+1].T,d_L)*a
			b_v[-i]=d_L
			temp1=np.reshape(d_L,(d_L.shape[0],1))
			temp2=np.reshape(a_l[-i-1],(1,a_l[-i-1].shape[0]))
			w_v[-i]=np.dot(temp1,temp2)
		return b_v,w_v

def cost(a,b):
	return -(np.sum(b-a))

					
									
				
			
				
		
			
		
