"""Stochastic gradient descent for optimising inputs in NN ,Deepy learning"""

class sgd():
	def __init__(self,par,label,learn_rate=0.1):
		self.par = par
		self.learn_rate=learn_rate
		self.label =label
		self.theta=np.zeros(self.par.shape[1])
			

	
	def opt(self):
	"""implemented for LR for now"""
		while repeat>0:
			for j in range(len(self.theta)):
				for i in range(self.m):
					res= self.par*(self.hypo(i))-self.label[i])*self.par[i][j]
					self.theta[j] += -res


	
	 
	
