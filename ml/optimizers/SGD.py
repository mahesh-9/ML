#TODO add momentum
from ..loss import LOSS
class SGD:
	def __init__(self,X,Y,n_epoch,learning_rate=10e-4,shuffle=True,batch_size=15,back_bool=True):
		self.feat=X
		self.target=Y
		self.epoch=n_epoch
		self.lr=learning_rate
		self.back_bool=back_bool
		self.batch_size=batch_size
		self.train_data=np.asarray(list(zip(self.feat,self.target)))
		self.LD=len(self.train_data)

	def make_batches(self,batch_size):
		"""makes batches"""
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
	
	def optimize(self,momentum=True):
		"""optimizes weights and biases"""
		for i in range(self.epoch):
			np.random.shuffle(self.train_data)
			batches=self.make_batches(self.batch_size)
			for x in batches:
				_get_grads(x)

	def _get_grads(self,mini_batch):
		for f,t in mini_batch:pass	
		return NotImplemented 

			
						
				
			
				
		
			
		
