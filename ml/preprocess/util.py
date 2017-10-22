import numpy as np
def checkfit(X,Y):
	if not isinstance(X,np.ndarray):
		X=np.array(X)
	if not isinstance(Y,np.ndarray):
		Y=np.array(Y)
	if len(X)!=len(Y):
		raise ValueError("The length of the features vector and target vector does not match")
	elif len(X.shape)!=2:
			raise ValueError("all the values in the feature vector do not have same dimensions")
	else:return True
def one_hot_encoding(y):
	"""creates one-hot vector for the target variables
	for example,if a training set consists of multiple classes y=(0,1,2...n)
	one-hot vector for y(0)=[1,0,0,..n],y(2)=[0,0,1,0,..n]"""
	no_=y.shape[0]
	class_set=set(y)
	temp=np.full([no_,len(class_set)],0.0)
	for i in range(len(y)):
		temp[i][y[i]]=1
	return temp
def normalize(X):
	"""	This function normalizes the feature values:
		new_x=(x-min(x))/(max(x)-min(x))"""
	mi=np.min(X,axis=0)
	ma=np.max(X,axis=0)
	r=np.full(X.shape,np.inf)
	for i in range(len(X)):
		q=(X[i]-mi)/(ma-mi)
		r[i]=q
	return r
def split_train_test(X,Y,per=20):
			"""slpits training and testing data
			X=feature vector
			Y=target vector
			per=percentage of data the training set should take
			Example:
				from sklearn.datasets import load_iris
				a=load_iris()
				train_f,train_l,test_f,test_l=split_train_test(a.data,a.target)
			"""
			#if checkfit(X,Y):
			class_set=set(Y)
			l=list(Y)
			count=l.count(list(class_set)[0])
			dim_1=int((per/100)*count)*len(list(class_set))
			dim_2=X.shape[1]
			n_train_f=np.full([dim_1,dim_2],np.inf)
			n_test_f=np.full([len(X)-dim_1,dim_2],np.inf)
			n_train_t=np.full([dim_1,],0)
			n_test_t=np.full([len(X)-dim_1,],0)	
			f_i=0
			ins=f_i
			s_i=int((per/100)*count)
			ind=s_i
			temp=0
			for i in range(len(class_set)):
				n_train_f[ins:ins+ind]=X[f_i:s_i]
				n_test_f[temp:(count-ind)+temp]=X[s_i:f_i+count]
				n_train_t[ins:ins+ind]=Y[f_i:s_i]
				n_test_t[temp:(count-ind)+temp]=Y[s_i:f_i+count]
				f_i+=count
				s_i+=count
				ins+=ind
				temp+=count-ind
			return n_train_f,n_train_t,n_test_f,n_test_t
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
		try: new_shape=self.ob.shape[0],self.ob.shape[1]+1
		except IndexError:
			new_shape=self.ob.shape[0],1
		b=np.full(new_shape,np.inf)
		for i in range(len(self.ob)):
			temp=np.insert(self.ob[i],col_no,val)
			b[i]=temp
		return b
		
		
