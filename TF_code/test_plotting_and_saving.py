#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import extract_image_from_field
import matplotlib as mpl 

# np.savez('data2plot.npz', h_actual=h_actual, X=X, Y=Y, vmax=vmax, vmin=vmin)
data = np.load('data2plot.npz')
h_actual = data['h_actual']
X = data['X']
Y = data['Y']
vmax = data['vmax']
vmin = data['vmin']

num_contour_levels = 200
Tsim_steps = h_actual.shape[0]
image_width = 5
image_height = 5
dpi_val = 100
height_pixels = image_height*dpi_val
width_pixels = image_width*dpi_val

print(h_actual.shape)
h_images = extract_image_from_field(h_actual, image_height, image_width, dpi_val, X, Y, num_contour_levels, vmin, vmax)
print(h_images.shape)

#--------------------------------------------------------------------------------------------------------------------
# plt.figure()
# h_temp = np.pad(h_actual[0,:,:], ((1,1), (1,1)), 'constant', constant_values=0)
# plt.contourf(X, Y, h_temp, num_contour_levels, cmap=cm.jet, vmin=vmin, vmax=vmax)
# plt.title('starting profile')
# # plt.colorbar()
# # plt.clim(vmin, vmax)

# plt.figure()
# h_temp = np.pad(h_actual[int((Tsim_steps+1)/2),:,:], ((1,1), (1,1)), 'constant', constant_values=0)
# plt.contourf(X, Y, h_temp, num_contour_levels, cmap=cm.jet, vmin=vmin, vmax=vmax)
# plt.title('half-way profile')
# # plt.colorbar()
# # plt.clim(vmin, vmax)

# mpl.rcParams['toolbar']='None'
# image_width = 5
# image_height = 5
# dpi_val = 100

# fig = plt.figure(figsize=(image_width, image_height), dpi=dpi_val)
# plt.axis('off')

# h_temp = np.pad(h_actual[-1,:,:], ((1,1), (1,1)), 'constant', constant_values=0)
# plt.contourf(X, Y, h_temp, num_contour_levels, cmap=cm.jet, vmin=vmin, vmax=vmax)

# fig.canvas.draw()
# height_pixels = image_height*dpi_val
# width_pixels = image_width*dpi_val
# img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height_pixels, width_pixels,3)

# plt.imshow(img)
# plt.imsave('temp_img', img, format="png")

# plt.show()

#-----------------------------------------------Testing------------------------------------------------------

# np.set_printoptions(precision=5, linewidth=1000)

# def remove_whitespaces(img):
# 	height_pixels = img.shape[0]
# 	width_pixels = img.shape[1]

# 	x1 = 0
# 	y = int(height_pixels/2)
# 	while(img[y,x1,0]>0):
# 		x1+=1
	
# 	x2 = width_pixels -1 
# 	while(img[y,x2,0]>0):
# 		x2-=1

# 	y1 = 0
# 	x = int(width_pixels/2)
# 	while(img[y1,x,0]>0):
# 		y1+=1

# 	y2 = height_pixels -1
# 	while(img[y2,x,0]>0):
# 		y2-=1

# 	return img[y1:y2+1, x1:x2+1,:]

# # Create all field data to plot/convert to images:
# hs_to_plot = [h_actual[0,:,:], h_actual[int((Tsim_steps+1)/2),:,:], h_actual[-1,:,:]]

# mpl.rcParams['toolbar']='None'


# fig = plt.figure(figsize=(image_height, image_width), dpi=dpi_val)
# plt.axis('off')

# for i in range(len(hs_to_plot)):
# 	print(i)
# 	h_temp = np.pad(hs_to_plot[i], ((1,1), (1,1)), 'constant', constant_values=0) # Zero padding 
# 	plt.contourf(X, Y, h_temp, num_contour_levels, cmap=cm.jet, vmin=vmin, vmax=vmax)

# 	fig.canvas.draw()
# 	img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height_pixels, width_pixels,3)	
# 	print("Image shape before removing whitespaces:", img.shape)
# 	new_img = remove_whitespaces(img)
# 	print("Image shape after removing whitespaces:", new_img.shape)	
# 	plt.imsave('temp_img_'+str(i), new_img, format="png")
# 	plt.clf()




