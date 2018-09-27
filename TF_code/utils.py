from matplotlib import cm
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np

def remove_whitespaces(img):
	height_pixels = img.shape[0]
	width_pixels = img.shape[1]

	x1 = 0
	y = int(height_pixels/2)
	while(img[y,x1,0]>0):
		x1+=1
	
	x2 = width_pixels -1 
	while(img[y,x2,0]>0):
		x2-=1

	y1 = 0
	x = int(width_pixels/2)
	while(img[y1,x,0]>0):
		y1+=1

	y2 = height_pixels -1
	while(img[y2,x,0]>0):
		y2-=1

	# return img[y1:y2+1, x1:x2+1,:]
	return y1,y2,x1,x2

def extract_image_from_field(h, image_height, image_width, dpi_val, X, Y, num_contour_levels, vmin, vmax, greyscale):
	height_pixels = int(image_height*dpi_val)
	width_pixels = int(image_width*dpi_val)	
	mpl.rcParams['toolbar']='None'
	fig = plt.figure(figsize=(image_height, image_width), dpi=dpi_val)
	plt.axis('off')	
	bs = h.shape[0] # batch size (= num_gpus)
	
	image_list = []
	for i in range(bs):
		h_temp = np.pad(h[i,:,:], ((1,1), (1,1)), 'constant', constant_values=0) # Zero padding 
		plt.contourf(X, Y, h_temp, num_contour_levels, cmap=cm.jet, vmin=vmin, vmax=vmax)

		fig.canvas.draw()
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height_pixels, width_pixels, 3)	
		if i == 0:
			y1,y2,x1,x2 = remove_whitespaces(img)
		# new_img = img[y1:y2+1, x1:x2+1,:]
		new_img = img[y1+1:y2, x1+1:x2,:]

		# Make image greyscale using formula for luminosity (0.21 R + 0.72 G + 0.07 B) :
		if greyscale:
			new_img = ( 0.21 * new_img[:,:,0] + 0.72 * new_img[:,:,1] + 0.07 * new_img[:,:,2]).astype('uint8')
		image_list.append(np.expand_dims(new_img, 0))
		plt.clf()

	plt.close()
	image_list = np.concatenate(image_list, 0)
	return image_list

