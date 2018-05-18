#!/usr/bin/env python

#############################################################
#						Revision History					#
#############################################################

# reads in a file containing CASA ispec spectral data and outputs a simple frequency-ordered ascii file in MHz

# 1.0 - Project start (5/18/2018)

#############################################################
#							Preamble						#
#############################################################

import sys

#Python version check

if sys.version_info.major != 3:

	print("This code is written in Python 3.  It will not execute in Python 2.7.  Exiting (sorry).")
	
	quit()

import numpy as np

version = 1.0

#############################################################
#							Functions						#
#############################################################
	
#read_cat reads the ispec file in

def read_file(ispec_file):

	'''
	Reads in a catalog file line by line
	'''

	my_array = []

	try:
		with open(ispec_file) as input:
	
			for line in input:
		
				my_array.append(line)	
	except TypeError:
		print('Specify an ispec file.')
		return			
			
	return my_array	
	
def splice_ispec(raw_array,outfile='sorted.ispec',GHz=False):
		
	#clean out the empty lines and the commented lines
	
	clean_array = [x for x in raw_array if x[0] != '#' and x[0] != '\n']
	
	#split the lines into frequency and intensity arrays
	
	frequency = np.arange(len(clean_array),dtype=np.float64)
	intensity = np.arange(len(clean_array),dtype=np.float64)
	
	for x in range(len(clean_array)):
	
		frequency[x] = np.float64(clean_array[x].split()[0].strip())
		intensity[x] = np.float64(clean_array[x].split()[1].strip())
		
	if GHz == True:
	
		frequency *= 1000
		
	#sort the arrays
	
	idx = frequency.argsort()
	
	frequency = frequency[idx]
	intensity = intensity[idx]
	
	with open(outfile,'w') as output:
	
		for x in range(len(frequency)-1):
		
			output.write('{:.4f} {}\n' .format(frequency[x],intensity[x]))
			
		#This way there's no extra return and a blank line at the end of the file
		
		output.write('{:.4f} {}' .format(frequency[-1],intensity[-1]))
			
	return

#meta function

def convert_ispec(infile,out='sorted.ispec',GHz=False):

	raw_array = read_file(infile)
	
	splice_ispec(raw_array,outfile=out,GHz=GHz)			