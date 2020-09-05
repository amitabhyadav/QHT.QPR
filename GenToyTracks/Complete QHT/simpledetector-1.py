import numpy as np
import matplotlib.pyplot as plt

class SimpleDetector(object):
	def __init__(self):
		pass
	

	def plot_detector_layers(self):
    		###################################################################################################
   		# Function: Generates a simple detector with 10 layers spaced equidistant in the range(-1000,1000)
    		# Input: N/A
    		# Output: Plots the circular layers of the ATLAS Inner Detector
    		###################################################################################################

    		layer_1 = plt.Circle((0, 0), 100, color='gray', linewidth=.1, fill=False)
    		layer_2 = plt.Circle((0, 0), 200, color='gray', linewidth=.1, fill=False)
    		layer_3 = plt.Circle((0, 0), 300, color='gray', linewidth=.1, fill=False)
    		layer_4 = plt.Circle((0, 0), 400, color='gray', linewidth=.1, fill=False)
    		layer_5 = plt.Circle((0, 0), 500, color='gray', linewidth=.1, fill=False)
    		layer_6 = plt.Circle((0, 0), 600, color='gray', linewidth=.1, fill=False)
    		layer_7 = plt.Circle((0, 0), 700, color='gray', linewidth=.1, fill=False)
    		layer_8 = plt.Circle((0, 0), 800, color='gray', linewidth=.1, fill=False)
    		layer_9 = plt.Circle((0, 0), 900, color='gray', linewidth=.1, fill=False)
    		layer_10 = plt.Circle((0, 0), 1000, color='gray', linewidth=.1, fill=False)
   
    		fig, ax = plt.subplots(num=None, figsize=(6,6), dpi=80) # with an existing figure # fig = plt.gcf() # ax = fig.gca()
    
    		# change default range so that new circles will work
    		ax.set_xlim((-1100, 1100))
    		ax.set_ylim((-1100, 1100))
    		
    		#plot layers
    		ax.add_artist(layer_1)
    		ax.add_artist(layer_2)
    		ax.add_artist(layer_3)
    		ax.add_artist(layer_4)
    		ax.add_artist(layer_5)
    		ax.add_artist(layer_6)
    		ax.add_artist(layer_7)
    		ax.add_artist(layer_8)
    		ax.add_artist(layer_9)
    		ax.add_artist(layer_10)
    		fig.savefig('plotcircles.png')
    
	def get_track_coordinates(self):
    		#####################################################################################################
    		# Function: Plot Cylindrical Space
    		# Input: R and Phi cylindrical coordinates of tracks, expressed as a python List.
    		# Output: Scatter Plot
    		# Remarks: The values for different tracks can be generated using: https://www.desmos.com/calculator
    		#####################################################################################################
    		# Method 1: Generate tracks from discrete data. The following data is for 10 circular tracks.
    
    		track_val_x = [5, 20, 45, 80, 125, 180, 245, 320, 405, 500,
                  		8.333, 33.333, 75, 133.333, 208.333, 300, 408.333, 533.333, 675, 833.333,
                  		99.745, 197.949, 293.031, 383.329, 467.025, 542.105, 606.218, 656.521, 689.387, 699.854,
                  		99.875, 198.997, 296.606, 391.918, 484.123, 572.364, 655.725, 733.212, 803.726, 866.025,
                 		-0.714, -2.857, -6.429, -11.429, -17.857, -25.714, -35, -45.714, -57.857, -71.429,
                  		-7.143, -28.571, -64.286, -114.286, -178.571, -257.143, -350, -457.143, -578.571, -714.286,
                 		-7.143, -28.571, -64.286, -114.286, -178.571, -257.143, -350, -457.143, -578.571, -714.286,
                  		0.714, 2.857, 6.429, 11.429, 17.857, 25.714, 35, 45.714, 57.857, 71.429,
                  		4, 16, 36, 64, 100, 144, 196, 256, 324, 400,
                  		9.091, 36.364, 81.818, 145.455, 227.273, 327.273, 445.455, 581.818, 736.364, 909.091]
    
    		track_val_y = [99.875, 198.997, 296.606, 391.918, 484.123, 572.364, 655.725, 733.212, 803.726, 866.025,
                  		99.625, 197.203, 290.474, 377.124, 454.530, 519.615, 568.563, 596.285, 595.294, 552.771,
                  		7.143, 28.571, 64.286, 114.286, 178.571, 257.143, 350, 457.143, 578.571, 714.286,
                  		5, 20, 45, 80, 125, 180, 245, 320, 405, 500,
                  		99.997, 199.980, 299.931, 399.837, 499.681, 599.449, 699.124, 798.693, 898.138, 997.446,
                  		99.745, 197.949, 293.031, 383.329, 467.025, 542.105, 606.218, 656.521, 689.387, 699.854,
                  		-99.745, -197.949, -293.031, -383.329, -467.025, -542.105, -606.218, -656.521, -689.387, -699.854,
                  		-99.997, -199.980, -299.931, -399.837, -499.681, -599.449, -699.124, -798.693, -898.138, -997.446,
                  		-99.92, -199.359, -297.832, -394.847, -489.898, -582.464, -672, -757.934, -839.657, -916.515,
                  		-99.586, -196.666, -288.627, -372.616, -445.362, -502.884, -539.972, -549.079, -517.464, -416.598]
    
    		# Method 2: Generate tracks from equation of a circle.
    		# Not Required, here. For implementation, refer to the notebook: QHT-classical-pre-processing
    
    		return track_val_x, track_val_y

	def plot_tracks_from_array(self, x, y):
    		#################################################################################
    		# Function: Plot Cartesian Space
    		# Input: X and Y cartesian coordinates of tracks, expressed as a python List.
    		# Output: Scatter Plot
    		################################################################################

    		plt.scatter(x,y,s=10)
    		#plt.grid(b=True, which='major', axis='both', color='green', linestyle='-', linewidth=0.1)
    		plt.xlabel('X')
    		plt.ylabel('Y')
    		plt.show()