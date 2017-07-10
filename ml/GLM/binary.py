"""Binary classification"""
from math import log,e
from ..numc import *
from .lr import LR
class LogisticRegression():
	def fit(self,X,Y):
		LR.fit(self,X,Y)
		self.classes=set(Y)
		if len(self.classes)>2:
			raise ValueError("This problem is for binary classification expected only 2 classes but given %d"%(len(self.classes)))
		self.gd()
		return self
	def hypo(self,it):
		res=1/(1+e**(LR.hyp(self,it)))
		return res		
	def cost(self):
		s=0
		for _ in range(self.m):
			s+=self.target[_]*log(self.hypo(_))+(1-self.target[_])*log(1-self.hypo(_))
		r=(1/self.m)*s
		return r
	def gd(self,rate=0.01,loops=200):
		for k in range(loops):
			for i in range(len(self.theta)):
				ts=0
				for j in range(self.m):
					res=(self.hypo(it=j)-self.target[j])*self.feat[j][i]
					ts+=res
				self.theta[i]+=rate*(1/(self.m))*(ts)
			t=self.cost()
	def predict(self,x):
		x=np.array(x)
		labels=x.dot(self.theta[1:])+self.theta[0]
		w=np.full((labels.shape[0],),np.inf,dtype=int)
		k=labels.flat
		for i in range(len(list(k))):
			if k[i]>0:
				w[i]=0
			else:w[i]=1
		return w
				
		
		
		
		
		
