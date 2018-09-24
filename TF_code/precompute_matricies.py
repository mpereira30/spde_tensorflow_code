#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np

from scipy import integrate

import configparser
config = configparser.RawConfigParser()

config.read('2Dheat.cfg')

dt = config.getfloat('sec1', 'dt')
Tsim = config.getfloat('sec1', 'Tsim')
Tf = config.getfloat('sec1', 'Tf')
a = config.getfloat('sec1', 'a')
J = config.getint('sec1', 'J')
iters = config.getint('sec1', 'iters')
rollouts = config.getint('sec1', 'rollouts')
rho = config.getfloat('sec1', 'rho')
scale_factor = config.getfloat('sec1', 'scale_factor')
epsilon = config.getfloat('sec1', 'epsilon')

Tsim_steps = int(np.ceil(Tsim/dt)) # total number of simulation timesteps
T = int(np.ceil(Tf/dt)) # total number of MPC timesteps
sigma = np.float32(1.0/np.sqrt(rho))

# Get actuator related data:
mu_x = np.asarray(config.get('sec2', 'mu_x').split(','), dtype=np.float32) * np.float32(a)
mu_y = np.asarray(config.get('sec2', 'mu_y').split(','), dtype=np.float32) * np.float32(a)
N = len(mu_x)
sig_val = np.float32(config.getfloat('sec2','sig_val') * a)
sig_val = np.square(sig_val) * np.ones(N)
sig_xx, sig_yy = sig_val, sig_val

num_int_delta = 0.001
x_vals = np.arange(0,a,num_int_delta)
y_vals = np.arange(0,a,num_int_delta)

def compute_curly_M():
	curly_M = np.zeros((J-1,J-1,N), dtype=np.float32)
	for j2 in range(J-1):
		for j1 in range(J-1):
			for n in range(N):

				# From MATLAB code (for reference):
				# fun_x = @(x) exp( -0.5 .* ((x - mu_x(n)).^2)./sig_xx(n) ) .* sin( (j1*pi/a) .* x ); 
				# fun_y = @(y) exp( -0.5 .* ((y - mu_y(n)).^2)./sig_yy(n) ) .* sin( (j2*pi/a) .* y ); 

				fun_x = lambda x: np.sqrt(2.0/a) * np.exp( -0.5 * ((x - mu_x[n])**2)/sig_xx[n] ) * np.sin( ((j1+1)*np.pi/a) * x )
				fun_y = lambda y: np.sqrt(2.0/a) * np.exp( -0.5 * ((y - mu_y[n])**2)/sig_yy[n] ) * np.sin( ((j2+1)*np.pi/a) * y )
				curly_M[j2,j1,n] = np.float32( np.trapz(fun_x(x_vals), x_vals) * np.trapz(fun_y(y_vals), y_vals) )

	return curly_M 	

def compute_curly_v_tilde():
	temp = np.zeros((N,J-1,J-1), dtype=np.float32) # shape is [ (number of actuators) x (y_dim) x (x_dim) ]
	curly_v_tilde = np.zeros((N, (J-1)*(J-1)), dtype=np.float32) # (N x y_dim * x_dim)
	for n in range(N):
		for i in range(J-1): # y_dim (Dirichlet B.C.s)
			for j in range(J-1): # x_dim (Dirichlet B.C.s)
				y = (i+1)*a/J
				x = (j+1)*a/J
				temp[n,i,j] = np.float32( np.exp( -0.5 * ((x - mu_x[n])**2)/sig_xx[n] ) * np.exp( -0.5 * ((y - mu_y[n])**2)/sig_yy[n] ) )
		temp_1 = np.squeeze(temp[n,:,:])
		curly_v_tilde[n,:] = np.reshape(temp_1.T,(1,-1)) # We need to take transpose before reshaping so that we get the correct vec operation with numpy arrays
	return curly_v_tilde

def compute_capital_M():
	M = np.zeros((N,N), dtype=np.float32)
	for i in range(N):
		for j in range(N):
			mu_x_i = mu_x[i]
			mu_y_i = mu_y[i]
			mu_x_j = mu_x[j]
			mu_y_j = mu_y[j]
			sig_xx_i = sig_xx[i]
			sig_yy_i = sig_yy[i]
			sig_xx_j = sig_xx[j]
			sig_yy_j = sig_yy[j]

			fun_x = lambda x:  np.exp( -0.5 * ((x - mu_x_i)**2)/sig_xx_i ) * np.exp( -0.5 * ((x - mu_x_j)**2)/sig_xx_j )
			fun_y = lambda y:  np.exp( -0.5 * ((y - mu_y_i)**2)/sig_yy_i ) * np.exp( -0.5 * ((y - mu_y_j)**2)/sig_yy_j ) 
			M[i,j] = np.float32( np.trapz(fun_x(x_vals), x_vals) * np.trapz(fun_y(y_vals), y_vals) )
	return M