
#!/usr/bin/env python

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import ConfigParser
import math
from precompute_matricies import compute_curly_M, compute_curly_v_tilde, compute_capital_M
from scipy.sparse import dia_matrix, identity, csc_matrix
from scipy.sparse.linalg import inv
from scipy.fftpack import dst
from time import time

np.random.seed(int(time()))
config = ConfigParser.RawConfigParser()
config.read('2Dheat.cfg')

dt = np.float32(config.getfloat('sec1', 'dt'))
Tsim = config.getfloat('sec1', 'Tsim')
Tf = config.getfloat('sec1', 'Tf')
a = config.getfloat('sec1', 'a')
J = config.getint('sec1', 'J')
iters = config.getint('sec1', 'iters')
rollouts = config.getint('sec1', 'rollouts')
rho = np.float32(config.getfloat('sec1', 'rho'))
scale_factor = config.getfloat('sec1', 'scale_factor')
epsilon = np.float32(config.getfloat('sec1', 'epsilon'))
desired_temperature = np.float32(config.getfloat('sec1', 'desired_temperature'))

Tsim_steps = np.int32(Tsim/dt) # total number of simulation timesteps
T = np.int32(Tf/dt) # total number of MPC timesteps
sigma = np.float32(1.0/np.sqrt(rho))
z = np.float32(a/J)

mu_x = np.asarray(config.get('sec2', 'mu_x').split(','), dtype=np.float32) * np.float32(a)
mu_y = np.asarray(config.get('sec2', 'mu_y').split(','), dtype=np.float32) * np.float32(a)
N = len(mu_x)
sig_val = np.float32(config.getfloat('sec2','sig_val') * a)
sig_val = np.square(sig_val) * np.ones(N)
sig_xx, sig_yy = sig_val, sig_val

print("\n")
print("-----------------------------Parameters-----------------------------")
print("time discretization, dt:", dt, " seconds")
print("total simulation time, Tsim:", Tsim, " seconds")
print("total simulation timesteps, Tsim_steps:", Tsim_steps)
print("MPC time horizon, Tf:", Tf, " seconds")
print("MPC timesteps:", T)
print("length of each side of the metal plate, a:", a)
print("number of spatial points, J:", J)
print("number of path-integral iterations per MPC step, iters:", iters)
print("number of rollouts, rollouts:", rollouts)
print("temperature parameter, rho:", rho)
print("noise standard-deviation, sigma:", sigma)
print("cost scaling factor, scale_factor:", scale_factor)
print("thermal diffusivity, epsilon:", epsilon)
print("desired_temperature:", desired_temperature)
print("--------------------------------------------------------------------")
print("\n")

# Pre-compute matricies:
print("Pre-computing curly_M, curly_v_tilde and capital_M matricies ... ")
curly_M = compute_curly_M()
curly_v_tilde = compute_curly_v_tilde()
M = compute_capital_M()
M_inv = np.linalg.inv(M)
print("Finished computing matricies.")

# Compute the finite-difference sparse matrix:
print("Computing finite-difference sparse matrix inverse ...")
e = np.ones((J-1)*(J-1), dtype=np.float32)
data = np.asarray([-e,-e,4*e,-e,-e])
offsets = [-(J-1), -1, 0, 1, J-1]
sp_A = (dt*epsilon/z**2) * dia_matrix((data, offsets), shape=((J-1)**2, (J-1)**2)).toarray()
for i in range(1,J-1):
	sp_A[ i*(J-1), i*(J-1)-1 ] = 0
	sp_A[ i*(J-1)-1, i*(J-1) ] = 0;
sp_I = identity((J-1)*(J-1), format='csc', dtype='float32')
EE = sp_I + csc_matrix(sp_A)
EE_inv = inv(EE)
# print( np.round( EE.dot(EE_inv).todense() ) ) # check if inverse was correct
EE_coo = EE_inv.tocoo()
indices = np.mat([EE_coo.row, EE_coo.col]).transpose() # COO format row and column index arrays of the matrix
print("Finished computing matrix inverse.")

# Set desired temperature profile:
# Each column represents the vertices (x1,x2,y1,y2) of each patch
r_x1 = [np.int32(0.45*(J-1)), np.int32(0.15*(J-1)), np.int32(0.75*(J-1))]
r_x2 = [np.int32(0.55*(J-1)), np.int32(0.25*(J-1)), np.int32(0.85*(J-1))]
r_y1 = [np.int32(0.45*(J-1)), np.int32(0.45*(J-1)), np.int32(0.45*(J-1))]
r_y2 = [np.int32(0.55*(J-1)), np.int32(0.55*(J-1)), np.int32(0.55*(J-1))]

h_d = np.zeros((J-1,J-1), dtype=np.float32)
for i in range(len(r_x1)):
	h_d[ r_x1[i]:r_x2[i], r_y1[i]:r_y2[i] ] = desired_temperature

#---------------------------------------------------------------------------------- TF code ----------------------------------------------------------------------------------

with tf.device('/CPU:0'):

	# Input placeholders for simulating "actual dynamics":
	h0_input_cpu = tf.placeholder(tf.float32, [J-1,J-1])
	U_input_cpu = tf.placeholder(tf.float32, [T, N])
	
	# Constants for cpu:
	EE_inv_tensor_cpu = tf.SparseTensor(indices, EE_coo.data, EE_coo.shape)
	curly_v_tilde_tensor_cpu = tf.constant(curly_v_tilde, dtype=tf.float32, shape=[N, (J-1)*(J-1)])	

	# Generate rollouts:
	def generate_rollouts_cpu(h_n, dW_nd_V): 
		dW_n = tf.expand_dims(dW_nd_V[0],-1) # ((J-1)^2, 1)
		V_n = tf.expand_dims(dW_nd_V[1],-1) # ((J-1)^2, 1)
		h_new = tf.sparse_tensor_dense_matmul(EE_inv_tensor_cpu, h_n + V_n + dW_n)
		return tf.reshape(h_new, [(J-1)**2, 1])

	def tf_dst_cpu(Xi_):
		n, m = Xi_.get_shape().as_list()
		A_tf_rev = tf.reverse(-Xi_, [0])
		y = tf.concat([tf.zeros([1,m], dtype=tf.complex64), Xi_, tf.zeros([1,m], dtype=tf.complex64), A_tf_rev], axis=0)
		y_t = tf.transpose(y, perm=[1,0])
		yy = tf.transpose(tf.fft(y_t), perm=[1,0])
		deno = tf.complex(0.0,-2.0)
		return yy[1:n+1,:]/deno

	def gen_noise(Xi_): # input shape: (T, J-1, J-1)
		temp_list = []
		for t_ in range(T):
			Xi_horizon = tf.squeeze(Xi_[t_,:,:])
			Xi_complex = tf.complex(Xi_horizon, tf.zeros([J-1,J-1]))
			dW_mat = tf.real(tf.transpose(tf_dst_cpu(tf.transpose(tf_dst_cpu(Xi_complex), perm=[1,0])), perm=[1,0]))
			temp_list.append(tf.reshape(tf.transpose(dW_mat, perm=[1,0]), [1,-1]))
		return tf.concat(temp_list, 0)

	Xi_cpu = tf.random_normal(shape=[T, J-1, J-1], mean=0.0, stddev=1.0, dtype=tf.float32)
	dW_cpu = gen_noise(Xi_cpu)
	dW_cpu = tf.multiply(dW_cpu, sigma*2.0*tf.sqrt(dt)/a) # Output shape is (T,(J-1)^2)

	curly_v_cpu = dt * tf.tensordot(U_input_cpu, curly_v_tilde_tensor_cpu, axes=[[1], [0]]) # output shape : (T, (J-1)^2)
	h0_vec_cpu = tf.reshape(tf.transpose(h0_input_cpu, perm=[1,0]), [-1,1]) # shape is ((J-1)^2,1)
	h_samples_cpu = tf.scan(fn=generate_rollouts_cpu, elems=(dW_cpu, curly_v_cpu), initializer=h0_vec_cpu, back_prop=False, parallel_iterations=rollouts)
	# (T,(J-1)^2,1)

	# Input placeholders for GPU:
	h0_input = tf.placeholder(tf.float32, [J-1,J-1])
	U_input = tf.placeholder(tf.float32, [T, N])
	
with tf.device('/GPU:0'):

	h_d_tensor = tf.constant(h_d, dtype=tf.float32, shape=[J-1,J-1])
	curly_M_tensor = tf.constant(curly_M, dtype=tf.float32, shape=[J-1,J-1,N])
	curly_v_tilde_tensor = tf.constant(curly_v_tilde, dtype=tf.float32, shape=[N, (J-1)*(J-1)])
	M_tensor = tf.constant(M, dtype=tf.float32, shape=[N,N])
	M_tensor_inv = tf.constant(M_inv, dtype=tf.float32, shape=[N,N])
	EE_inv_tensor = tf.SparseTensor(indices, EE_coo.data, EE_coo.shape)
	Xi = tf.random_normal(shape=[rollouts, T, J-1, J-1], mean=0.0, stddev=1.0, dtype=tf.float32)

	#------------- Generate the cylindrical noise for all rollouts --------------------------------------------------------------------------------------------------

	def tf_dst(Xi_):
		n, m = Xi_.get_shape().as_list()
		A_tf_rev = tf.reverse(-Xi_, [0])
		y = tf.concat([tf.zeros([1,m], dtype=tf.complex64), Xi_, tf.zeros([1,m], dtype=tf.complex64), A_tf_rev], axis=0)
		y_t = tf.transpose(y, perm=[1,0])
		yy = tf.transpose(tf.fft(y_t), perm=[1,0])
		deno = tf.complex(0.0,-2.0)
		return yy[1:n+1,:]/deno

	def rolls_split(Xi_roll): # each input will be of shape (T, J-1, J-1)
		temp_list = []
		for t_ in range(T):
			Xi_horizon = tf.squeeze(Xi_roll[t_,:,:])
			Xi_complex = tf.complex(Xi_horizon, tf.zeros([J-1,J-1]))
			dW_mat = tf.real(tf.transpose(tf_dst(tf.transpose(tf_dst(Xi_complex), perm=[1,0])), perm=[1,0]))
			temp_list.append(tf.reshape(tf.transpose(dW_mat, perm=[1,0]), [1,-1]))
		return tf.concat(temp_list, 0)

	dW = tf.map_fn(fn=rolls_split, elems=Xi, back_prop=False, dtype=tf.float32, parallel_iterations=rollouts) # split over rollouts
	dW = tf.multiply(dW, sigma*2.0*tf.sqrt(dt)/a) # Output shape is (r,T,(J-1)^2)

	#------------------------------------------------------------------------------------------------------------------------------------------------------------------

	#-------------------Propagate dynamics for all rollouts------------------------------------------------------------------------------------------------------------
	
	curly_v = dt * tf.tensordot(U_input, curly_v_tilde_tensor, axes=[[1], [0]]) # output shape : ( T, (J-1)^2 )
	h0_vec = tf.reshape(tf.transpose(h0_input, perm=[1,0]), [-1,1]) # shape is ( (J-1)^2, 1 )
	h0_vec_rolls = tf.concat([h0_vec]*rollouts, axis=1) # shape is ( (J-1)^2, rollouts )

	# Generate rollouts:
	def generate_rollouts(h_n, dW_nd_V): 
		dW_n = dW_nd_V[0] # ( (J-1)^2, rollouts )
		V_n = tf.expand_dims(dW_nd_V[1],-1) # ( (J-1)^2, 1 )
		h_new = tf.sparse_tensor_dense_matmul(EE_inv_tensor, h_n + V_n + dW_n)
		return tf.reshape(h_new, [(J-1)**2, rollouts])

	# use scan to split across horizon dimension:
	h_samples = tf.transpose( tf.scan(fn=generate_rollouts, elems=(tf.transpose(dW, perm=[1,2,0]), curly_v), initializer=h0_vec_rolls, back_prop=False, parallel_iterations=rollouts), perm=[2,0,1] )
	# ( T, (J-1)^2, rollouts ) to ( rollouts, T, (J-1)^2 )

	def h_r_split(h_n_r): # each is (T,(J-1)^2)
		temp_list = []
		for t_ in range(T):
			h_n_T = tf.expand_dims(h_n_r[t_,:], -1)
			temp_list.append(tf.expand_dims(tf.transpose(tf.reshape(h_n_T,[J-1,J-1]), perm=[1,0]),0))			
		return tf.concat(temp_list, 0)
			
	h_ns = tf.map_fn(fn=h_r_split, elems=h_samples, back_prop=False, dtype=tf.float32, parallel_iterations=rollouts) # split over T & final shape is (rollouts,T,J-1,J-1)

	#------------------------------------------------------------------------------------------------------------------------------------------------------------------

	#-------------------Cost Computation and Control Update------------------------------------------------------------------------------------------------------------

	temp_hd, temp_hns = [], []
	for i in range(len(r_x1)):
		temp_hd.append( h_d_tensor[ r_x1[i]:r_x2[i], r_y1[i]:r_y2[i] ] )
		temp_hns.append( h_ns[ :, :, r_x1[i]:r_x2[i], r_y1[i]:r_y2[i] ] )
	h_d_sliced = tf.concat(temp_hd, axis=0) # shape
	h_ns_sliced = tf.concat(temp_hns, axis=2) # (r,T,xxx,xxx)
	
	# Compute total state-cost:
	J_h = tf.expand_dims( tf.reduce_sum(tf.square(h_ns_sliced - tf.expand_dims(tf.expand_dims(h_d_sliced,0),0)), axis=[1,2,3], keep_dims=False), -1) # final shape : (rollouts,1)
	
	# computation of zeta_1 and zeta_2:
	Xi_td_curly_M_tensor = tf.tensordot(Xi, curly_M_tensor, axes=[[2,3],[0,1]]) # shape : (r,T,N)
	zeta_1 = tf.sqrt(dt/rho) * tf.expand_dims(tf.tensordot(Xi_td_curly_M_tensor, U_input, axes=[[1,2],[0,1]]),-1)
	zeta_2 = 0.5 * dt * tf.reduce_sum(tf.multiply(tf.matmul(U_input, M_tensor), U_input)) * tf.ones([rollouts,1], dtype=tf.float32)

	# Compute total cost and weights:
	J_h_tilde = J_h + zeta_1 + zeta_2 # each (rollouts,1)
	minCost = tf.reduce_min(J_h_tilde, keep_dims=False)
	maxCost = tf.reduce_max(J_h_tilde, keep_dims=False)
	J_h_tilde = J_h_tilde - minCost
	J_h_tilde = J_h_tilde / (maxCost - minCost)
	weights = tf.nn.softmax(-rho*J_h_tilde) # (rollouts,1)
	
	# Compute the control update:
	u_update_weighted =	tf.sqrt(dt) * tf.expand_dims(weights,-1) * Xi_td_curly_M_tensor  # (r,T,N)
	u_update_expected_value = tf.div(tf.reduce_sum(u_update_weighted, axis=0, keep_dims=False), rollouts) # (T,N)
	
	U_new = U_input + (1.0/(dt*tf.sqrt(rho))) * tf.transpose(tf.matmul(M_tensor_inv, u_update_expected_value, transpose_b=True), perm=[1,0])

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	h_current = np.zeros((J-1,J-1))
	U_current = np.random.randn(T,N)

	for t_sim in range(Tsim_steps):

		print("timestep: ", t_sim,"/", Tsim_steps)

		starttime = time()

		for iter_ in range(iters):

			U_current = sess.run(U_new, 
			feed_dict={
			h0_input: h_current,
			U_input: U_current,
			})

		
			
		endtime = time()
		print("Iterations time: ", endtime-starttime)

