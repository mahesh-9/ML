from numpy import np
from math import log

class LOSS:
	
	def cross_entropy(self,target,feat,hyp):
		self.target  = target
		self.feat = feat
		self.hyp = hyp
		self.m = len(self.feat)
		L=np.zeros((self.m,))
		for i in range(self.m):
			L[i] = -log(softmax(self.hyp(i)))
		return np.mean(L)

	def cat_cross_entropy(self,classes):
		self.target= one_hotvector(self.target)
		for i in range(self.m):
			for j in range(classes):
				res= self.target[i][j]*log(self.hyp(i))
				L += res
		
		L = -L/self.m
		return L


	def one_hotvector(self,values):
		self.values= values
		n_values = np.max(self.values)+1
		vector=np.eye(n_values)[values]
		return vector
