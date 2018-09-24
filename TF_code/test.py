
#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
from scipy.sparse import dia_matrix
from scipy.fftpack import dst
import tensorflow as tf
from time import time

'''
J = 5
e = np.ones((J-1)*(J-1))
data = np.asarray([-e,-e,4*e,-e,-e])
offsets = [-(J-1), -1, 0, 1, J-1]

spmat = dia_matrix((data, offsets), shape=((J-1)**2, (J-1)**2)).toarray()
print(spmat)
print(spmat.shape)

for i in range(1,J-1):
	# print(i)
	# print(i*(J-1), i*(J-1)-1)
	# print(i*(J-1)-1, i*(J-1))
	spmat[ i*(J-1), i*(J-1)-1 ] = 0
	spmat[ i*(J-1)-1, i*(J-1) ] = 0;

print(spmat)
print(spmat.shape)
'''

#------------------------------- Testing code for cylindrical noise generation:-------------------------------------------------------------------------
# J = 6
# Xi = np.random.randn(J-1, J-1)

# dW = np.zeros((J-1, J-1))
# for k1 in range(1,J):
# 	for j1 in range(1,J):
# 		x_n = np.sin(np.pi*j1*k1/J) * Xi[j1-1,:]
# 		dW[k1-1,:] = dW[k1-1,:] + 0.5*dst(x_n, type=1)
# print(dW)
# print("\n")

# dW_1 = np.transpose(0.5*dst(np.transpose(0.5*dst(Xi, type=1)), type=1))
# print(dW_1)
# print("\n")


# A = np.asarray([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24]])
# print(np.transpose(0.5*dst(np.transpose(0.5*dst(A, type=1)), type=1)))

# A_tf_const = tf.constant(A, dtype=tf.complex64, shape=[4,6])

# def tf_dst(A_tf):
# 	n, m = A_tf.get_shape().as_list()
# 	A_tf_rev = tf.reverse(-A_tf, [0])
# 	y = tf.concat([tf.zeros([1,m], dtype=tf.complex64), A_tf, tf.zeros([1,m], dtype=tf.complex64), A_tf_rev], axis=0)
# 	y_t = tf.transpose(y, perm=[1,0])
# 	yy = tf.transpose(tf.fft(y_t), perm=[1,0])
# 	deno = tf.complex(0.0,-2.0)
# 	return yy[1:n+1,:]/deno


# ddst = tf.real(tf.transpose( tf_dst( tf.transpose( tf_dst(A_tf_const), perm=[1,0] ) ), perm=[1,0] ))

# with tf.Session() as sess:

# 	# A_, Ar_, y_, yy_, dst_output_, y_t_ = sess.run([A_tf, A_tf_rev, y, yy, dst_output, y_t])
# 	# print(A_,"\n")
# 	# print(Ar_,"\n")
# 	# print(y_,"\n")
# 	# print(y_t_,"\n")
# 	# print(yy_,"\n")
# 	# print(dst_output_,"\n")
	
# 	ddst_ = sess.run([ddst])
# 	print(ddst_)

#------------------------------------------------------------------------------------------------------------------------------------------------------------

# A = np.asarray([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
# A_tf = tf.constant(A, dtype=tf.float32, shape=[4,4])
# # A_vec = tf.reshape(A_tf, [-1,1]) # WRONG (This does not do vec. It stacks the elements row-wise)
# A_vec = tf.reshape(tf.transpose(A_tf, perm=[1,0]), [-1,1])
# A_mat = tf.transpose(tf.reshape(A_vec,[4,4]), perm=[1,0])

# B = tf.random_normal(shape=[6,6], mean=0.0, stddev=1.0, dtype=tf.float32)
# range_x1 = range(1,3) 
# range_y1 = range(1,3)

# range_x2 = range(5,8)
# range_y2 = range(5,8)

# range_x = range_x1 + range_x2
# range_y = range_y1 + range_y2

# r_x1 = [1, 4]
# r_x2 = [3, 6]
# r_y1 = [1, 4]
# r_y2 = [3, 6]

# # range_indicies = tf.constant([X1,Y1]) 
# temp = []
# for i in range(len(r_x1)):
# 	temp.append( B[ r_x1[i]:r_x2[i], r_y1[i]:r_y2[i] ] )
# B_s = tf.concat(temp, axis=0)

# with tf.Session() as sess:
# 	B_, B_s_ = sess.run([B, B_s])
# 	print(B_,"\n")
# 	print(B_s_,"\n")
# 	# A_, A_vec_, A_mat_, A_sliced_, B_, B_s_ = sess.run([A_tf, A_vec, A_mat, A_sliced, B, B_s])
	# print(A_,"\n")
	# print(A_vec_,"\n")
	# print(A_mat_,"\n")
	# print(A_sliced_,"\n")
	# print(B_,"\n")
	# print(B_s_,"\n")

#---------------------------------------------- Tensordot------------------------------------------------------------------------------------------------------
r = 10
T = 5
J = 16
N = 3

Xi = np.random.randn(r,T,J-1,J-1)
U = np.random.randn(T,N)
curly_M = np.random.randn(J-1,J-1,N)
capital_M = np.random.randn(N,N)

Xi_tf = tf.constant(Xi, dtype=tf.float32, shape=[r,T,J-1,J-1])
U_tf = tf.constant(U, dtype=tf.float32, shape=[T,N])
U_tf_T = tf.transpose(U_tf, perm=[1,0])
M_tf = tf.constant(curly_M, dtype=tf.float32, shape=[J-1,J-1,N])
CM_tf = tf.constant(capital_M, dtype=tf.float32, shape=[N,N])

#---------------------------------------------------ZETA_1------------------------------------------------------------------------------------------------

# # (1.) Using nested for-loops:
with tf.device('/CPU:0'):

	zeta_list = []
	for r_ in range(r):
		temp_list =[]
		for T_ in range(T):
			m_temp = tf.expand_dims( tf.reduce_sum(tf.multiply(M_tf, tf.expand_dims(tf.squeeze(Xi_tf[r_,T_,:,:]),-1)), axis=[0,1], keep_dims=False), -1)
			# print(U_tf[T_,:].get_shape())
			temp_list.append( tf.matmul(tf.expand_dims(U_tf[T_,:],0), m_temp) )
		zeta_list.append(tf.reduce_sum(temp_list))

	zeta_1 = tf.stack(zeta_list,0)

	# computation of zeta_1 and zeta_2:
	def m_horizon_split(Xi_horizon): # each input will be of shape (J-1, J-1)
		return tf.reduce_sum(tf.multiply(M_tf, tf.expand_dims(Xi_horizon,-1)), axis=[0,1]) 

	def m_rolls_split(Xi_roll): # each input will be of shape (T, J-1, J-1)
		return tf.map_fn(fn=m_horizon_split, elems=Xi_roll, back_prop=False, dtype=tf.float32, parallel_iterations=r) # split over time horizon T	

	m_i = tf.map_fn(fn=m_rolls_split, elems=Xi_tf, back_prop=False, dtype=tf.float32, parallel_iterations=r) # split over rollouts & final shape : (r,T,N)

	# (2.) Using tensordot after computing m_i's:
	# zeta_1_td = tf.tensordot(m_i, U_tf_T, axes=[[1,2],[1,0]]) # input shapes: (r,T,N) and (N,T) 
	# ----------OR---------------

	zeta_1_td = tf.tensordot(m_i, U_tf, axes=[[1,2],[0,1]]) # input shapes: (r,T,N) and (T,N)

	# (3.) Full tensordot:
	zeta_1_fast = tf.tensordot(tf.tensordot(Xi_tf, M_tf, axes=[[2,3],[0,1]]), U_tf, axes=[[1,2],[0,1]])

	# (4.) Using 1 map and for-loops:

	def m_rolls_split_1(Xi_roll): # each input will be of shape (T, J-1, J-1)
		# print(Xi_roll.get_shape())
		temp_list = []
		for t_ in range(T):
			temp_list.append( tf.expand_dims( tf.reduce_sum(tf.multiply(M_tf, tf.expand_dims(tf.squeeze(Xi_roll[t_,:,:]),-1)), axis=[0,1]), 0) )
		
		return tf.concat(temp_list, axis=0)
	
	m_ii = tf.map_fn(fn=m_rolls_split_1, elems=Xi_tf, back_prop=False, dtype=tf.float32, parallel_iterations=r) # split over rollouts & final shape : (r,T,N)
	print(m_ii.get_shape())
	zeta_1_td_1 = tf.tensordot(m_ii, U_tf, axes=[[1,2],[0,1]]) # input shapes: (r,T,N) and (T,N)

# # Why above 2 are equivalent (using U_tf or U_tf_T)??
# # Tensordot will contract the tensors along PAIRS OF AXES. 
# # axis a_axes[i] of a must have the same dimension as axis b_axes[i] of b for all i in range(0, len(a_axes))


#---------------------------------------------------ZETA_2------------------------------------------------------------------------------------------------

# # (1.) for-loop method:
# temp_list1 = []
# temp_list2 = []
# for T_ in range(T):
# 	temp_list1.append( tf.tensordot(tf.tensordot(tf.expand_dims(U_tf[T_,:],0), CM_tf, axes=[[1],[0]]), tf.expand_dims(U_tf_T[:,T_],-1), axes=[[1],[0]]) )
# 	temp_list2.append( tf.matmul(tf.matmul(tf.expand_dims(U_tf[T_,:],0), CM_tf), tf.expand_dims(U_tf_T[:,T_],-1)) )
# total_1 = tf.reduce_sum(temp_list1)
# total_11 = tf.reduce_sum(temp_list2)

# # (2.)  functional equivalent of for-loop:
# def CM_Tsplit(U_t):
# 	U_t = tf.expand_dims(U_t,0)
# 	return tf.matmul(tf.matmul(U_t, CM_tf), U_t, transpose_b=True)
# total_2 = tf.reduce_sum(tf.map_fn(fn=CM_Tsplit, elems=U_tf, back_prop=False, dtype=tf.float32, parallel_iterations=r)) # split over rollouts	

# #  (3.)  Using tensordot and matrix tricks:
# total_21 = tf.tensordot(tf.tensordot(U_tf, CM_tf, axes=[[1],[0]]), U_tf, axes=[[0,1],[0,1]])
# total_22 = tf.reduce_sum(tf.multiply(tf.matmul(U_tf, CM_tf), U_tf))

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Control Update------------------------------------------------------------------------------------------------
# CM_tf_inv = tf.matrix_inverse(CM_tf)
# U_new_list = []
# for t_ in range(T):
# 	temp_list = []
# 	for r_ in range(r):
# 		temp_list.append( tf.expand_dims( tf.reduce_sum(tf.multiply(M_tf, tf.expand_dims(tf.squeeze(Xi_tf[r_,t_,:,:]),-1)), axis=[0,1], keep_dims=False), -1) )
# 	u_update = tf.reduce_sum( tf.concat(temp_list, -1), axis=-1, keep_dims=True)
# 	U_new_list.append( tf.transpose(tf.matmul(CM_tf_inv, u_update), perm=[1,0]) )

# U_new_forloop = tf.concat(U_new_list, 0)

# # (2.) using tensordot:
# U_new_td = tf.transpose(tf.matmul(CM_tf_inv, tf.transpose(tf.reduce_sum(tf.tensordot(Xi_tf, M_tf, axes=[[2,3],[0,1]]), axis=0, keep_dims=False), perm=[1,0])), perm=[1,0])

with tf.Session() as sess:

# 	U1 = sess.run([U_new_forloop])
# 	print(U1,"\n")

# 	U2 = sess.run([U_new_td])
# 	print(U2,"\n")	

# z1_, z2_, z1f_ = sess.run([zeta_1, zeta_1_td, zeta_1_fast])
# print(z1_,"\n")
# print(z2_,"\n")
# print(z1f_, "\n")

	s = time()
	z1_ = sess.run([zeta_1])
	print(time()-s)
	print(z1_,"\n")

	s = time()
	z2_ = sess.run([zeta_1_td])
	print(time()-s)
	print(z2_,"\n")

	s = time()
	z1f_ = sess.run([zeta_1_fast])
	print(time()-s)
	print(z1f_,"\n")	

	s = time()
	z1_1 = sess.run([zeta_1_td_1])
	print(time()-s)
	print(z1_1,"\n")	

# t1, t11, t2, t21, t22 = sess.run([total_1, total_11, total_2, total_21, total_22])
# print(t1, t11, t2, t21, t22)