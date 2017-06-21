import numpy as np
class nx:
	"""a class for custom methods in numpy"""
	def __init__(self,ob):
		if not isinstance(ob,np.ndarray):
			raise("expected np.array object")
		else:
			self.ob=ob
	def add_col(self,col_no,val=np.inf):
		"""this method returns an numpy.ndarray
		which consists of an extra column (with a same value) to the original array object passed in __init__.
		col_no=column number to add
		val=value to be passed""" 
		new_shape=self.ob.shape[0],self.ob.shape[1]+1
		b=np.full(new_shape,np.inf)
		for i in range(len(self.ob)):
			temp=np.insert(self.ob[i],col_no,val)
			b[i]=temp
		return b
				
def split_train_test(X,Y,per=20):
		class_set=set(Y)
		l=list(Y)
		print("l",l)
		count=l.count(list(class_set)[0])
		print("coutn",count)
		dim_1=int((per/100)*count)*len(list(class_set))
		dim_2=X.shape[1]
		print(dim_1,dim_2)
		n_train_f=np.full([dim_1,dim_2],np.inf)
		n_test_f=np.full([len(X)-dim_1,dim_2],np.inf)
		n_train_t=np.full([dim_1,],np.inf)
		n_test_t=np.full([len(X)-dim_1,],np.inf)	
		f_i=0
		ins=f_i
		s_i=int((per/100)*count)
		ind=s_i
		for i in range(len(list(class_set))):
			n_train_f[ins:ins+ind]=X[f_i:s_i]
			n_test_f[s_i:f_i+50]=X[s_i:f_i+50]
			n_train_t[f_i:s_i]=Y[f_i:s_i]
			n_test_t[s_i:f_i+50]=Y[s_i:f_i+50]
			f_i+=count
			s_i+=count
			ins+=ind
			print(n_train_f)
			print(f_i,s_i)
		return n_train_f,n_train_t,n_test_f,n_test_t		
