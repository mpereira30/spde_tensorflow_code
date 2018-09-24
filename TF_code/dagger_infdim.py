
#!/usr/bin/env python

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from precompute_matricies import compute_curly_M, compute_curly_v_tilde, compute_capital_M
from scipy.sparse import dia_matrix, identity, csc_matrix
from scipy.sparse.linalg import inv
from scipy.fftpack import dst
from time import time
import configparser
import sys
from matplotlib import cm
import matplotlib as mpl 
from utils import extract_image_from_field

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
fnn_layer_size = config.getint('sec1', 'fnn_layer_size')
learning_rate = config.getfloat('sec1', 'learning_rate')
batch_size_l = config.getint('sec1', 'batch_size_l')

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

init_sigma = 0.5
num_gpus = 8
ops_device = '/GPU:0'
terminal_only = False
sigma_sys = 0.2
precompute = False

image_width = 5
image_height = 5
dpi_val = 100
height_pixels = image_height*dpi_val
width_pixels = image_width*dpi_val
channels = 3
num_filters = 8

savepath="/home/mpereira30/spdes/TF_code/save_n_log"
beta_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02,  0.00, 0.00]

print("")
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
print("Ops on device:", ops_device)
print("Considering terminal cost only? ", terminal_only)
print("Noise sigma for simulating action on real system:", sigma_sys)
print("Using", num_gpus, "GPUs")
print("Sigma for initial field profile perturbation:", init_sigma)
print("Learner batch size:", batch_size_l)
print("--------------------------------------------------------------------")
print("")

# Pre-compute matricies:
if precompute:
	print("Pre-computing curly_M, curly_v_tilde and capital_M matricies ... ")
	curly_M = compute_curly_M()
	curly_v_tilde = compute_curly_v_tilde()
	M = compute_capital_M()
	M_inv = np.linalg.inv(M)
	np.savez("precomputedMatricies.npz", curly_M=curly_M, M=M, M_inv=M_inv, curly_v_tilde=curly_v_tilde)
	print("Finished computing matricies. Saved in npz file.")
else:
	print("Loading precomputed matrices from npz file ...")
	allMatrices = np.load('precomputedMatricies.npz')
	curly_M = allMatrices['curly_M']
	curly_v_tilde = allMatrices['curly_v_tilde']
	M = allMatrices['M']
	M_inv = allMatrices['M_inv']

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
r_y1 = np.int32([np.round(0.48*(J-1)), np.round(0.48*(J-1)), np.round(0.48*(J-1)), np.round(0.18*(J-1)), np.round(0.78*(J-1))])
r_y2 = np.int32([np.round(0.52*(J-1)), np.round(0.52*(J-1)), np.round(0.52*(J-1)), np.round(0.22*(J-1)), np.round(0.82*(J-1))])
r_x1 = np.int32([np.round(0.48*(J-1)), np.round(0.18*(J-1)), np.round(0.78*(J-1)), np.round(0.48*(J-1)), np.round(0.48*(J-1))])
r_x2 = np.int32([np.round(0.52*(J-1)), np.round(0.22*(J-1)), np.round(0.82*(J-1)), np.round(0.52*(J-1)), np.round(0.52*(J-1))])

print("")
print("x and y coordinates of actuators are:")
print("r_y1:",r_y1)
print("r_y2:",r_y2)
print("r_x1:",r_x1)
print("r_x2:",r_x2)
print("")

h_d = np.zeros((J-1,J-1), dtype=np.float32)
# +1 has to be added because when referencing with numpy, the "upto" index is not considered
h_d[ r_y1[0]:r_y2[0]+1, r_x1[0]:r_x2[0]+1 ] = 0.5*desired_temperature
for i in range(1, len(r_x1)):
	h_d[ r_y1[i]:r_y2[i]+1, r_x1[i]:r_x2[i]+1  ] = desired_temperature

#---------------------------------------------------------------------------------- Expert ----------------------------------------------------------------------------------

with tf.device('/CPU:0'):

	# Input placeholders for GPU:
	h0_input_cpu = tf.placeholder(tf.float32, [num_gpus,J-1,J-1])
	U_input_cpu = tf.placeholder(tf.float32, [num_gpus,T,N])

	# Generate all noise on CPU first:
	Xi_cpu = tf.random_normal(shape=[num_gpus, rollouts, T, J-1, J-1], mean=0.0, stddev=1.0, dtype=tf.float32)

	def tf_dst(Xi_):
		return 0.5*dst(0.5*dst(Xi_, type=1).T, type=1).T

	def rolls_split(Xi_roll): # each input will be of shape (T, J-1, J-1)
		temp_list = []
		for t_ in range(T):
			dW_mat = tf.py_func(tf_dst, [tf.squeeze(Xi_roll[t_,:,:])], tf.float32)
			temp_list.append(tf.reshape(tf.transpose(dW_mat, perm=[1,0]), [1,(J-1)**2])) # Transpose is required before flattening reshape for vec operation
		return tf.concat(temp_list, 0)

	def batch_split(Xi_batch): # input shape (r,T,J-1,J-1)
		return tf.map_fn(fn=rolls_split, elems=Xi_batch, back_prop=False, dtype=tf.float32, parallel_iterations=rollouts) # split over rollouts

	dW_cpu = tf.map_fn(fn=batch_split, elems=Xi_cpu, back_prop=False, dtype=tf.float32, parallel_iterations=num_gpus) # split over batch
	dW_cpu = sigma * (2.0 * np.sqrt(dt)/a) * dW_cpu # Output shape is (bs,r,T,(J-1)^2)

	#----------------------------------------------------------------------------------------------------------------------------------
delta_U_list = []
for ng in range(num_gpus):
	with tf.device('/GPU:%d' % ng):

		h_d_tensor = tf.constant(h_d, dtype=tf.float32, shape=[J-1,J-1])
		curly_M_tensor = tf.constant(curly_M, dtype=tf.float32, shape=[J-1,J-1,N])
		curly_v_tilde_tensor = tf.constant(curly_v_tilde, dtype=tf.float32, shape=[N, (J-1)*(J-1)])
		M_tensor = tf.constant(M, dtype=tf.float32, shape=[N,N])
		M_tensor_inv = tf.constant(M_inv, dtype=tf.float32, shape=[N,N])
		EE_inv_tensor = tf.SparseTensor(indices, EE_coo.data, EE_coo.shape)

		#-------------------Propagate dynamics for all rollouts------------------------------------------------------------------------------------------------------------
		U_input = U_input_cpu[ng,:,:]	
		h0_input = h0_input_cpu[ng,:,:]
		Xi = Xi_cpu[ng,:,:,:,:]

		curly_v = dt * tf.tensordot(U_input, curly_v_tilde_tensor, axes=[[1], [0]]) # output shape : ( T, (J-1)^2 )
		h0_vec = tf.reshape(tf.transpose(h0_input, perm=[1,0]), [-1,1]) # shape is ( (J-1)^2, 1 ). Transpose is required before flattening reshape for vec operation
		h0_vec_rollouts = tf.concat([h0_vec]*rollouts, axis=1) # shape is ( (J-1)^2, rollouts )

		# Generate rollouts:
		h_samples_list = []
		dW = tf.transpose(dW_cpu[ng,:,:,:], perm=[1,2,0]) # change shape (r,T,(J-1)^2) ---> (T,(J-1)^2,r)
		curly_v_T = tf.transpose(curly_v, perm=[1,0]) # change shape (T,(J-1)^2) ---> ((J-1)^2,T)
		h_n = h0_vec_rollouts # Setting the current field state to initial state. shape is ((J-1)^2, r) 
		
		# Propagating rollouts (in parallel) with for-loop (not tf.scan()):
		for t_ in range(T):
			h_n = tf.sparse_tensor_dense_matmul(EE_inv_tensor, h_n + tf.expand_dims(curly_v_T[:,t_],-1) + dW[t_,:,:]) # shape is ((J-1)^2,r)
			h_samples_list.append(tf.expand_dims(tf.transpose(h_n, perm=[1,0]), 1)) # shape change : ((J-1)^2,r) ---> (r,(J-1)^2) ---> (r,1,(J-1)^2)
		h_samples = tf.concat(h_samples_list, 1) # shape : (r,T,(J-1)^2)

		# convert the propagated vec arrays into 2D matrices for cost computation:
		def h_r_split(h_n_r): # each is (T,(J-1)^2)
			temp_list = []
			for t_ in range(T):
				h_n_T = tf.expand_dims(h_n_r[t_,:], -1)
				temp_list.append(tf.expand_dims(tf.transpose(tf.reshape(h_n_T,[J-1,J-1]), perm=[1,0]),0))			
			return tf.concat(temp_list, 0)
		
		h_ns = tf.map_fn(fn=h_r_split, elems=h_samples, back_prop=False, dtype=tf.float32, parallel_iterations=rollouts) # split over T & final shape is (rollouts,T,J-1,J-1)

		#-------------------Cost Computation and Control Update------------------------------------------------------------------------------------------------------------

		# First extract the relevant patches of the desired field and actual realizations. This makes it easy to compute cost using broadcasted arithmetic.
		temp_hd, temp_hns = [], []
		for i in range(len(r_x1)):
			temp_hd.append( h_d_tensor[ r_y1[i]:r_y2[i]+1, r_x1[i]:r_x2[i]+1 ] )
			temp_hns.append( h_ns[ :, :, r_y1[i]:r_y2[i]+1, r_x1[i]:r_x2[i]+1 ] )
		h_d_sliced = tf.concat(temp_hd, axis=0) # shape
		h_ns_sliced = tf.concat(temp_hns, axis=2) # (r,T,xxx,xxx)
		
		# Compute total state-cost:
		print("Building graph for running and terminal cost on GPU#", ng+1)
		J_h = scale_factor * tf.expand_dims( tf.reduce_sum(tf.square(h_ns_sliced - tf.expand_dims(tf.expand_dims(h_d_sliced,0),0)), axis=[1,2,3], keepdims=False), -1) # final shape : (rollouts,1)

		# computation of zeta_1 and zeta_2:
		Xi_td_curly_M_tensor = tf.tensordot(Xi, curly_M_tensor, axes=[[2,3],[0,1]]) # shape : (r,T,N)
		# The above operation performs element-wise multiplication on the (J-1)*(J-1) dimensions and collapes them to 1 by reduce summing. This is done for each N. So, this results in (1xN) vectors.
		# This has to be done for each rollout and each timestep. Therefore, tensordot is performed on [2,3],[0,1] to give us (1xN) vectors for each rollout and timestep.  
		zeta_1 = np.sqrt(dt/rho) * tf.expand_dims(tf.tensordot(Xi_td_curly_M_tensor, U_input, axes=[[1,2],[0,1]]),-1)
		zeta_2 = 0.5 * dt * tf.reduce_sum(tf.multiply(tf.matmul(U_input, M_tensor), U_input)) * tf.ones([rollouts,1], dtype=tf.float32)

		# Compute total cost and weights:
		J_h_tilde = J_h + zeta_1 + zeta_2 # each (rollouts,1)
		minCost = tf.reduce_min(J_h_tilde, keepdims=False)
		maxCost = tf.reduce_max(J_h_tilde, keepdims=False)
		J_h_tilde = J_h_tilde - minCost
		J_h_tilde = J_h_tilde / (maxCost - minCost)
		weights = tf.exp(-rho*J_h_tilde) 
		weights = weights / tf.reduce_mean(weights) # (rollouts,1)
		
		# Compute the control update:
		u_update_weighted =	np.sqrt(dt) * tf.expand_dims(weights,-1) * Xi_td_curly_M_tensor  # (r,T,N)
		u_update_expected_value = tf.reduce_sum(u_update_weighted, axis=0, keepdims=False) / rollouts # (T,N)
		
		delta_U_gpu = (1.0/(dt*np.sqrt(rho))) * tf.transpose(tf.matmul(M_tensor_inv, u_update_expected_value, transpose_b=True), perm=[1,0])
		delta_U_list.append(tf.expand_dims(delta_U_gpu,0))

delta_U = tf.concat(delta_U_list,0)
U_new = U_input_cpu + delta_U

#---------------------------------------------------------------------------------- Learner ----------------------------------------------------------------------------------

with tf.device('/CPU:0'):

	expert_targets = tf.placeholder(tf.float32, [num_gpus, batch_size_l, N])
	image_inputs_learner = tf.placeholder(tf.float32, [num_gpus, batch_size_l, height_pixels, width_pixels, channels]) 
	
	# unstack :
	expert_targets_ = tf.unstack(expert_targets, axis=0)
	image_inputs_learner_ = tf.unstack(image_inputs_learner, axis=0)

tower_grads= []
average_gpu_whole_loss = []
gpu_output_control_sequences = []

with tf.variable_scope("CNN_learner", reuse=tf.AUTO_REUSE) as scope_PI :

	for gpu_i in range(num_gpus):
		
		with tf.device('/GPU:%d' % gpu_i):

			# Some notes on conv nets:
			# strides = [1, 1, 1, 1] applies the filter to a patch at every offset, strides = [1, 2, 2, 1] applies the filter to every other image patch in each dimension

			def conv_relu(input, kernel_shape, bias_shape):
			    # Create variable named "weights".
			    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
			    # Create variable named "biases".
			    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
			    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
			    return tf.nn.relu(conv + biases)

			def process_batch_input_images(images, cnn_string):
				with tf.variable_scope(cnn_string+"conv_1"):
				    # Variables created here will be named "conv1/weights", "conv1/biases".
				    relu1 = conv_relu(images, [3,3,channels,num_filters], [num_filters]) # (-1,hpx,wpx,nf)
				    pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=[2, 2], strides=2) # (-1,hpx/2,wpx/2,nf)    
				with tf.variable_scope(cnn_string+"conv_2"):
				    # Variables created here will be named "conv2/weights", "conv2/biases".
					relu2 = conv_relu(pool1, [2,2,num_filters,num_filters], [num_filters]) # (-1,hpx/2,wpx/2,nf)
					pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2) # (-1, hpx/4, wpx/4, nf)	
					flat_size = int(height_pixels/4) * int(width_pixels/4) * num_filters 		    		
					pool2_flat = tf.reshape(pool2, [-1, flat_size])
					dense_W = tf.get_variable(name="denselayer_weights", shape=[flat_size, fnn_layer_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer())	
					dense_b = tf.get_variable(name="denselayer_biases", shape=[fnn_layer_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))	
				return tf.nn.relu(tf.matmul(pool2_flat, dense_W) + dense_b)

			dense_output = process_batch_input_images(image_inputs_learner_[gpu_i], "CNN")
	
			state_estimate_output_W = tf.get_variable(name="state_estimate_output_weights", shape=[fnn_layer_size, fnn_layer_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
			state_estimate_output_b = tf.get_variable(name="state_estimate_output_biases", shape=[fnn_layer_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))			
			state_estmate = tf.matmul(dense_output, state_estimate_output_W) + state_estimate_output_b

			# formula for pooling layer : 
			# w2 = (w1 - f)/S + 1
			# h2 = (h1 - f)/S + 1

			def FNN_policy(data): # shape of data is same as state_inputs_learner 

				# No split:
				L1_W = tf.get_variable(name="layer_1_weights", shape=[fnn_layer_size, fnn_layer_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
				L1_B = tf.get_variable(name="layer_1_biases", shape=[fnn_layer_size], dtype=tf.float32, initializer=tf.zeros_initializer())
				O_W = tf.get_variable( name="output_layer_weights", shape=[fnn_layer_size, N], dtype=tf.float32, initializer=tf.truncated_normal_initializer() )
				O_B = tf.get_variable( name="output_layer_biases", shape=[1, N], dtype=tf.float32, initializer=tf.zeros_initializer() )	
				
				l1 = tf.tanh(tf.matmul(data, L1_W) + L1_B)
				return tf.matmul(l1, O_W) + O_B	
				
			output = FNN_policy(state_estmate)
			scope_PI.reuse_variables()
			gpu_output_control_sequences.append(output)

			total_loss = tf.reduce_mean( tf.reduce_sum( tf.square( tf.subtract(expert_targets_[gpu_i], output) ), axis=-1) )
			adam_optimizer = tf.train.AdamOptimizer(learning_rate)
			tower_grads.append(adam_optimizer.compute_gradients(total_loss)) # returns a list of (gradient, variable) tuples
			average_gpu_whole_loss.append(total_loss)
			print("Built optimizer on GPU:", gpu_i)			

numvars = len(tower_grads[0])
averaged_grads = [None]*numvars
output_leaf = gpu_output_control_sequences

tvars = tf.trainable_variables()
print("Length of trainable variables list is:", len(tvars))
total_variables = 0
for _var in tvars:
    print(_var)
    mul = 1
    vals = _var.get_shape().as_list()
    for v in vals:
        mul *= v
    total_variables += mul
print("")
print("Total number of trainable variables are = ", total_variables)

#----------------------------------------------------------------------------------TF session------------------------------------------------------------------------------------------------------

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:

	sess.run(tf.global_variables_initializer())
	print("Initialize trainable variables")

	for _iter_ in range(len(beta_values)):

		h0s = []
		expert_Us = []

		beta = beta_values[_iter_]
		print("Iteration number:", _iter_+1, ", beta = ", beta)
		f = open(savepath+"train_logs.txt","a")
		f.write("\n")
		f.write("Iteration number:"+str(_iter_)+", beta = "+str(beta)+"\n")
		f.close()		

		np.random.seed(int(time()))

		h_current = init_sigma * np.random.rand(num_gpus,J-1,J-1)
		U_current = np.random.randn(num_gpus,T,N)
		print("")

		for t_sim in range(Tsim_steps):

			starttime = time()

			# Query the expert:
			for _ in range(iters):

				U_current = sess.run(U_new,
				feed_dict={
					h0_input_cpu: h_current,
					U_input_cpu: U_current,
				})

			# Apply control on actual system:	
			batch_costs = []
			for bs in range(num_gpus):
				h_vec = np.reshape(h_current[bs,:,:].T, [-1,1]) # vec operation
				v_n = np.dot(np.expand_dims(np.squeeze(U_current[bs,0,:]),0), curly_v_tilde).T # operation : (1 x N) x (N x (J-1)*(J-1)) ---(after transpose)---> ((J-1)*(J-1) x 1)
				dW_actual = sigma_sys * (2.0 * np.sqrt(dt)/a) * 0.5 * dst(0.5*dst(np.random.randn(J-1,J-1), type=1).T, type=1).T
				dW_vec = np.reshape(dW_actual.T, [-1,1])
				h_new = EE_inv.dot(h_vec + v_n*dt + dW_vec) # Propagate system 
				h_current[bs,:,:] = np.reshape(h_new, (J-1,J-1)).T # Convert from vec to 2D mat for next timestep
				U_current[bs,:,:] = np.concatenate((U_current[bs,1:,:], np.expand_dims(U_current[bs,-1,:],0)), axis=0) # Warm-start

				cost = 0
				for i in range(len(r_x1)):
					cost = cost + scale_factor * np.sum(np.square(h_d[ r_y1[i]:r_y2[i]+1, r_x1[i]:r_x2[i]+1 ] - h_current[bs, r_y1[i]:r_y2[i]+1, r_x1[i]:r_x2[i]+1 ]))
				batch_costs.append(cost)
			batch_costs = np.asarray(batch_costs)
			
			endtime_2 = time()

			sys.stdout.write("step:{} of {}, c1: {:3f}, c2: {:3f}, c3: {:3f}, c4: {:3f}, c5: {:3f}, c6: {:3f}, c7: {:3f}, c8: {:3f}, iter time: {:.5f}\r".format(t_sim+1, Tsim_steps, 
							 	    batch_costs[0],\
							 	    batch_costs[1],\
							 	    batch_costs[2],\
							 	    batch_costs[3],\
							 	    batch_costs[4],\
							 	    batch_costs[5],\
							 	    batch_costs[6],\
							 	    batch_costs[7],\
							 	    endtime_1 - starttime))
			sys.stdout.flush()
			
