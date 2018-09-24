
#!/usr/bin/env python

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import configparser
import math
from precompute_matricies import compute_curly_M, compute_curly_v_tilde, compute_capital_M
from scipy.sparse import dia_matrix, identity, csc_matrix
from scipy.sparse.linalg import inv
from scipy.fftpack import dst
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(int(time()))
config = configparser.RawConfigParser()
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

r_x1 = np.int32([np.round(0.48*(J-1)), np.round(0.18*(J-1)), np.round(0.78*(J-1)), np.round(0.48*(J-1)), np.round(0.48*(J-1))])
r_x2 = np.int32([np.round(0.52*(J-1)), np.round(0.22*(J-1)), np.round(0.82*(J-1)), np.round(0.52*(J-1)), np.round(0.52*(J-1))])
r_y1 = np.int32([np.round(0.48*(J-1)), np.round(0.48*(J-1)), np.round(0.48*(J-1)), np.round(0.18*(J-1)), np.round(0.78*(J-1))])
r_y2 = np.int32([np.round(0.52*(J-1)), np.round(0.52*(J-1)), np.round(0.52*(J-1)), np.round(0.22*(J-1)), np.round(0.82*(J-1))])

h_d = np.zeros((J-1,J-1), dtype=np.float32)
for i in range(len(r_x1)):
	h_d[ r_x1[i]:r_x2[i], r_y1[i]:r_y2[i] ] = desired_temperature

with tf.device('/CPU:0'):

	# Input placeholders for simulating "actual dynamics":
	h0_input_cpu = tf.placeholder(tf.float32, [J-1,J-1])
		
	# Constants for cpu:
	# EE_inv_tensor_cpu = tf.SparseTensor(indices, EE_coo.data, EE_coo.shape)
	EE_inv_tensor_cpu = tf.sparse_reorder(tf.SparseTensor(indices, EE_coo.data, EE_coo.shape))
	
	# Generate rollouts:
	def generate_rollouts_cpu(h_n, dW_): 
		dW_n = tf.expand_dims(dW_,-1) # ((J-1)^2, 1)
		h_new = tf.sparse_tensor_dense_matmul(EE_inv_tensor_cpu, h_n + dW_n)
		# h_new = tf.matmul(tf.sparse_tensor_to_dense(EE_inv_tensor_cpu), h_n + dW_n, a_is_sparse=True)
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
	# dW_cpu = tf.multiply(dW_cpu, sigma*2.0*tf.sqrt(dt)/a) # Output shape is (T,(J-1)^2)
	dW_cpu = tf.multiply(dW_cpu, 0.0*2.0*np.sqrt(dt)/a) # Output shape is (T,(J-1)^2)	

	h0_vec_cpu = tf.reshape(tf.transpose(h0_input_cpu, perm=[1,0]), [-1,1]) # shape is ((J-1)^2,1)
	h_samples_cpu = tf.scan(fn=generate_rollouts_cpu, elems=dW_cpu, initializer=h0_vec_cpu, back_prop=False, parallel_iterations=rollouts)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	print(sess.run([EE_inv_tensor_cpu]))

	h_traj = sess.run(h_samples_cpu, 
			feed_dict=
			{
			h0_input_cpu:h_d
			})

	h_actual = np.zeros((T,J-1,J-1))
	for t_ in range(T):
		h_actual[t_,:,:] = np.transpose(np.reshape(h_traj[t_,:,:], (J-1,J-1)), axes=[1,0])

	# Make data.
	X = np.arange(0, a, z)
	Y = np.arange(0, a, z)
	X, Y = np.meshgrid(X, Y)

	fig1 = plt.figure()
	ax = fig1.gca(projection='3d')
	surf = ax.plot_surface(X[1:J, 1:J], Y[1:J, 1:J], h_d, cmap=cm.hot, antialiased=True)
	# ax.set_zlim(0, 1.0)
	plt.title('inital condition')
	fig1.colorbar(surf)

	fig2 = plt.figure()
	ax = fig2.gca(projection='3d')
	surf = ax.plot_surface(X[1:J, 1:J], Y[1:J, 1:J], h_actual[np.int32(T*0.5),:,:], cmap=cm.hot, antialiased=True)
	# ax.set_zlim(0, 1.0)
	plt.title('half-way')
	fig2.colorbar(surf)	
	
	fig3 = plt.figure()
	ax = fig3.gca(projection='3d')
	surf = ax.plot_surface(X[1:J, 1:J], Y[1:J, 1:J], h_actual[-1,:,:], cmap=cm.hot, antialiased=True)
	# ax.set_zlim(0, 1.0)
	plt.title('end')
	fig3.colorbar(surf)		

	# plt.ion()
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# surf = ax.plot_surface(X[1:J, 1:J], Y[1:J, 1:J], h_d, cmap=cm.hot, antialiased=True)
	# ax.set_zlim(0, 1.0)
	# fig.colorbar(surf)
	# plt.pause(0.01)

	# for t_ in range(T):
	# 	print("timestep:",t_,"/",T)
	# 	plt.cla()
	# 	ax.plot_surface(X[1:J, 1:J], Y[1:J, 1:J], h_actual[t_,:,:], cmap=cm.hot, antialiased=True)
	# 	ax.set_zlim(0, 1.0)
	# 	plt.pause(0.01)

	# plt.ioff()    
	plt.show()

		

	
