from math import log,e
from ..numc import *
from .lr import LR
class Multi():
	def fit(self,X,Y):
		LR.fit(self,X,Y)
		return self
	def M_class1(self):
		for i,n in enumerate(self.target):
			if i==2:
				self.target[n] = 1
		return np.array(self.target)
	def M_class2(self):
		for i,n in enumerate(self.target):
			if i==1:
				self.target[n]=0
			elif i==2:
				self.target[n]=1
		return np.array(self.target)

	def M_class3(self):
		for i,n in enumerate(self.target):
			if i==0:
				self.target[n]=1
			elif i==2:
				self.target[n]=0
		return np.array(self.target)
			
	
	def hypo(self,it):
		res=1/(1+e**(-LR.hyp(self,it)))
		return res		
	
	def cost(self):
		self.M_class1()
		s=0
		for _ in range(self.m):
			s+=self.target[_]*log(self.hypo(_))+(1-self.target[_])*log(1-self.hypo(_))
		r=(-1/self.m)*s
		return r
	
	def gd(self,rate=0.001,loops=100):
		self.M_class1()
		for k in range(loops):
			for i in range(len(self.theta)):
				ts=0
				for j in range(self.m):
					res=(self.hypo(it=j)-self.target[j])*self.feat[j][i]
					ts+=res
				self.theta[i]-=rate*(1/(self.m))*(ts)
				#self.theta[i]-=rate*(ts)
			t=self.cost()
	def predict(self,x):
		x=np.array(x)
		return 1/(1+np.exp(-(x.dot(self.theta[1:])+self.theta[0])))
		
		
		
