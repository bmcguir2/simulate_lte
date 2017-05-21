#!/usr/bin/env python

#############################################################
#						Revision History					#
#############################################################

# reads in an spcat catalog and displays a spectrum

# 1.0 - Project start
# 2.0 - ipython overhaul and simplification

#############################################################
#							Preamble						#
#############################################################

import os, sys, argparse, math

#Python version check

if sys.version_info.major != 3:

	print("This code is written in Python 3.  It will not execute in Python 2.7.  Exiting (sorry).")
	
	quit()

import numpy as np
from numpy import exp as exp
import time as tm
import random
import warnings
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import itertools
import matplotlib.lines as mlines
from datetime import datetime, date, time
#warnings.filterwarnings('error')

version = 2.0

h = 6.626*10**(-34) #Planck's constant in J*s
k = 1.381*10**(-23) #Boltzmann's constant in J/K
kcm = 0.69503476 #Boltzmann's constant in cm-1/K
ckm = 2.998*10**5 #speed of light in km/s

#############################################################
#							Warning 						#
#############################################################

print('\nWarning! This code is in beta.  While it is believed that the resulted produced from it are robust, it is HIGHLY recommended to verify the results against a secondary resource before using for anything important.') 

#############################################################
#							Defaults 						#
#############################################################

auto_update = False

first_run = True

GHz = False

quietflag = False #turn on to suppress warnings about how long the simulation will take.

rms = float('-inf') #rms noise.  the simulation won't simulate any lines 10x smaller than this value.

thermal = float('inf') #initial default cutoff for optically-thick lines (i.e. don't touch them unless thermal is modified.)

T = 300 #temperature for simulations.  Default is 300 K.

catalog_file = None #catalog file to load in.  Needs to be a string.

C = 1.0E13 #column density.  Units are cm-2.

vlsr = 0.0 #vlsr offset applied to simulation.  Default is 0 km/s.

ll = float('-inf') #lower limit for the simulation range.  Default is none.

ul = float('inf') #upper limit for the simulation range.  Default is none.

spec = None	#file of a laboratory or observational spectrum to load in for comparison.  Default is none.

dV = 5.0 #linewidth of the simulation.  Default is 5.0 km/s.
	
CT = 300.0 #temperature the catalog is simulated at.  Default is 300 K.

Tbg = 2.7 #background continuum temperature (K)

gauss = True #toggle for simulating Gaussians or a stick spectrum.  Default is True.

#mode = 'SD' #sets the simulation either for a single dish ('SD') facility, which calculates dynamic beam sizes, or an array ('A'), which uses a fixed synthesized beam size.  Set this option manually, with configure_telescope(), or with init_telescope().

#dish_size = 10.0 #dish diamater of a single dish telescope in meters.  Set this option manually, with configure_telescope(), or with init_telescope().

eta = 1.0 #beam efficiency of the telescope.  Set this option manually, with configure_telescope(), or with init_telescope(). 

#units = 'K' #either 'K' or 'Jy/beam'.  Set this option manually, with configure_telescope(), or with init_telescope().

#synth_beam = 1.0 #synthesized beam size for an array in arcseconds.  Only used if mode = 'A'.  Set this option manually, with configure_telescope(), or with init_telescope().

#column_sim = False #if this is false (default), then S is just a static scalar.  If this is True, S is the column density, and the telescope correction is applied.

#source_size = 200.0 #source size in arcseconds.  Set this option manually or with init_source().

npts_line = 15 #default is 15 points across each line

res_kHz = False #if res_kHz is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of kHz

res_kms = False #if res_kms is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of km/s

res = 0.0 #resolution used in Gaussian simulation if res_kHz or res_kms is set to True.

#############################################################
#							Functions						#
#############################################################
	
#read_cat reads the catalog file in

def read_cat(catalog_file):

	'''
	Reads in a catalog file line by line
	'''

	my_array = []

	try:
		with open(catalog_file) as input:
	
			for line in input:
		
				my_array.append(line)	
	except TypeError:
		print('Specify a catalog file with catalog_file = \'x\'')
		return			
			
	return my_array	
	
#fix_pm fixes +/- quantum number issues

def fix_pm(qnarray):

	if '+' or '-' in qnarray:
	
		qnarray[qnarray == ''] = '0'
		qnarray[qnarray == '+'] = '1'
		qnarray[qnarray == '-'] = '2'
		
	return qnarray
	
#fix_qn fixes quantum number issues

def fix_qn(qnarray,line,old_qn):

	'''
	fixes quantum number issues arising from the use of alphabet characters to represent numbers in spcat
	'''

	new_qn = 000
			
	if 'A' in old_qn:
		
		new_qn = 100 + int(old_qn[1])
		
	if 'B' in old_qn:
		
		new_qn = 110 + int(old_qn[1])	
		
	if 'C' in old_qn:
		
		new_qn = 120 + int(old_qn[1])		

	if 'D' in old_qn:
		
		new_qn = 130 + int(old_qn[1])
		
	if 'E' in old_qn:
		
		new_qn = 140 + int(old_qn[1])
		
	if 'F' in old_qn:
		
		new_qn = 150 + int(old_qn[1])
		
	if 'G' in old_qn:
		
		new_qn = 160 + int(old_qn[1])
		
	if 'H' in old_qn:
		
		new_qn = 170 + int(old_qn[1])				
		
	if 'I' in old_qn:
		
		new_qn = 180 + int(old_qn[1])	
		
	if 'J' in old_qn:
		
		new_qn = 190 + int(old_qn[1])
		
	if 'K' in old_qn:
		
		new_qn = 200 + int(old_qn[1])
		
	if 'L' in old_qn:
		
		new_qn = 210 + int(old_qn[1])
		
	if 'M' in old_qn:
		
		new_qn = 220 + int(old_qn[1])	
		
	if 'N' in old_qn:
		
		new_qn = 230 + int(old_qn[1])	
		
	if 'O' in old_qn:
		
		new_qn = 240 + int(old_qn[1])
		
	if 'P' in old_qn:
		
		new_qn = 250 + int(old_qn[1])
		
	if 'Q' in old_qn:
		
		new_qn = 260 + int(old_qn[1])	
		
	if 'R' in old_qn:
		
		new_qn = 270 + int(old_qn[1])
		
	if 'S' in old_qn:
		
		new_qn = 280 + int(old_qn[1])
		
	if 'T' in old_qn:
		
		new_qn = 290 + int(old_qn[1])	
		
	if 'U' in old_qn:
		
		new_qn = 300 + int(old_qn[1])	
		
	if 'V' in old_qn:
		
		new_qn = 310 + int(old_qn[1])
		
	if 'W' in old_qn:
		
		new_qn = 320 + int(old_qn[1])	
		
	if 'X' in old_qn:
		
		new_qn = 330 + int(old_qn[1])	
		
	if 'Y' in old_qn:
		
		new_qn = 340 + int(old_qn[1])	
		
	if 'Z' in old_qn:
		
		new_qn = 350 + int(old_qn[1])
		
	if 'a' in old_qn:
		
		new_qn = 100 + int(old_qn[1])
		
	if 'b' in old_qn:
		
		new_qn = 110 + int(old_qn[1])	
		
	if 'c' in old_qn:
		
		new_qn = 120 + int(old_qn[1])		

	if 'd' in old_qn:
		
		new_qn = 130 + int(old_qn[1])
		
	if 'e' in old_qn:
		
		new_qn = 140 + int(old_qn[1])
		
	if 'f' in old_qn:
		
		new_qn = 150 + int(old_qn[1])
		
	if 'g' in old_qn:
		
		new_qn = 160 + int(old_qn[1])
		
	if 'h' in old_qn:
		
		new_qn = 170 + int(old_qn[1])				
		
	if 'i' in old_qn:
		
		new_qn = 180 + int(old_qn[1])	
		
	if 'j' in old_qn:
		
		new_qn = 190 + int(old_qn[1])
		
	if 'k' in old_qn:
		
		new_qn = 200 + int(old_qn[1])
		
	if 'l' in old_qn:
		
		new_qn = 210 + int(old_qn[1])
		
	if 'm' in old_qn:
		
		new_qn = 220 + int(old_qn[1])	
		
	if 'n' in old_qn:
		
		new_qn = 230 + int(old_qn[1])	
		
	if 'o' in old_qn:
		
		new_qn = 240 + int(old_qn[1])
		
	if 'p' in old_qn:
		
		new_qn = 250 + int(old_qn[1])
		
	if 'q' in old_qn:
		
		new_qn = 260 + int(old_qn[1])	
		
	if 'r' in old_qn:
		
		new_qn = 270 + int(old_qn[1])
		
	if 's' in old_qn:
		
		new_qn = 280 + int(old_qn[1])
		
	if 't' in old_qn:
		
		new_qn = 290 + int(old_qn[1])	
		
	if 'u' in old_qn:
		
		new_qn = 300 + int(old_qn[1])	
		
	if 'v' in old_qn:
		
		new_qn = 310 + int(old_qn[1])
		
	if 'w' in old_qn:
		
		new_qn = 320 + int(old_qn[1])	
		
	if 'x' in old_qn:
		
		new_qn = 330 + int(old_qn[1])	
		
	if 'y' in old_qn:
		
		new_qn = 340 + int(old_qn[1])	
		
	if 'z' in old_qn:
		
		new_qn = 350 + int(old_qn[1])																																									
				
	qnarray[line] = int(new_qn)			
	
# splices the catalog file appropriately, then populates a numpy array with the data

def splice_array(x):

	'''
	splices the catalog file appropriately, then populates a numpy array with the data
	'''

	frequency = np.arange(len(x),dtype=np.float)
	error = np.arange(len(x),dtype=np.float)
	logint = np.arange(len(x),dtype=np.float)
	dof = np.arange(len(x),dtype=np.int)
	elower = np.arange(len(x),dtype=np.float)
	gup = np.arange(len(x),dtype=np.int)
	tag = np.arange(len(x),dtype=np.int)
	qnformat = np.arange(len(x),dtype=np.int)
	qn1 = np.arange(len(x),dtype=object)
	qn2 = np.empty(len(x),dtype=object)
	qn3 = np.empty(len(x),dtype=object)
	qn4 = np.empty(len(x),dtype=object)
	qn5 = np.empty(len(x),dtype=object)
	qn6 = np.empty(len(x),dtype=object)
	qn7 = np.empty(len(x),dtype=object)
	qn8 = np.empty(len(x),dtype=object)
	qn9 = np.empty(len(x),dtype=object)
	qn10 = np.empty(len(x),dtype=object)
	qn11 = np.empty(len(x),dtype=object)
	qn12 = np.empty(len(x),dtype=object)
	
	parity = ['+','-']

	for line in range(len(x)):
	
		frequency[line] = float(str(x[line][:13]).strip())
		error[line] = float(str(x[line][13:21]).strip())
		logint[line] = float(str(x[line][21:29]).strip())
		dof[line] = int(str(x[line][29:31]).strip())
		elower[line] = float(str(x[line][31:41]).strip())
		try:
			gup[line] = int(str(x[line][41:44]).strip()) if str(x[line][41:44]).strip() else ''
		except ValueError:
			fix_qn(gup,line,str(x[line][41:44]))
		tag[line] = int(str(x[line][44:51]).strip())
		qnformat[line] = int(str(x[line][51:55]).strip())

		qn1[line] = str(x[line][55:57]).strip()
		qn2[line] = str(x[line][57:59]).strip()
		qn3[line] = str(x[line][59:61]).strip()
		qn4[line] = str(x[line][61:63]).strip()
		qn5[line] = str(x[line][63:65]).strip()
		qn6[line] = str(x[line][65:67]).strip()
		qn7[line] = str(x[line][67:69]).strip()
		qn8[line] = str(x[line][69:71]).strip()
		qn9[line] = str(x[line][71:73]).strip()
		qn10[line] = str(x[line][73:75]).strip()
		qn11[line] = str(x[line][75:77]).strip()
		qn12[line] = str(x[line][77:]).strip()
		
	if '+' in qn1 or '-' in qn1:
	
		qn1 = fix_pm(qn1)
		
	if '+' in qn2 or '-' in qn2:
	
		qn2 = fix_pm(qn2)	

	if '+' in qn3 or '-' in qn3:
	
		qn3 = fix_pm(qn3)
		
	if '+' in qn4 or '-' in qn4:
	
		qn4 = fix_pm(qn4)		
			
	if '+' in qn5 or '-' in qn5:
	
		qn5 = fix_pm(qn5)
	
	if '+' in qn6 or '-' in qn6:
	
		qn6 = fix_pm(qn6)
		
	if '+' in qn7 or '-' in qn7:
	
		qn7 = fix_pm(qn7)
		
	if '+' in qn8 or '-' in qn8:
	
		qn8 = fix_pm(qn8)
		
	if '+' in qn9 or '-' in qn9:
	
		qn9 = fix_pm(qn9)
		
	if '+' in qn10 or '-' in qn10:
	
		qn10 = fix_pm(qn10)
		
	if '+' in qn11 or '-' in qn11:
	
		qn11 = fix_pm(qn11)
		
	if '+' in qn12 or '-' in qn12:
	
		qn12 = fix_pm(qn12)														

	for line in range(len(qn1)):
	
		try:
			qn1[line] = int(qn1[line])
		except ValueError:
			fix_qn(qn1,line,qn1[line])
			
	for line in range(len(qn2)):
	
		try:
			qn2[line] = int(qn2[line])
		except ValueError:
			fix_qn(qn2,line,qn2[line])
			
	for line in range(len(qn3)):
	
		try:
			qn3[line] = int(qn3[line])
		except ValueError:
			fix_qn(qn3,line,qn3[line])						
			
	for line in range(len(qn4)):
	
		try:
			qn4[line] = int(qn4[line])
		except ValueError:
			fix_qn(qn4,line,qn4[line])

	for line in range(len(qn5)):
	
		try:
			qn5[line] = int(qn5[line])
		except ValueError:
			fix_qn(qn5,line,qn5[line])
			
	for line in range(len(qn6)):
	
		try:
			qn6[line] = int(qn6[line])
		except ValueError:
			fix_qn(qn6,line,qn6[line])
			
	for line in range(len(qn7)):
	
		try:
			qn7[line] = int(qn7[line])
		except ValueError:
			fix_qn(qn7,line,qn7[line])
			
	for line in range(len(qn8)):
	
		try:
			qn8[line] = int(qn8[line])
		except ValueError:
			fix_qn(qn8,line,qn8[line])
			
	for line in range(len(qn9)):
	
		try:
			qn9[line] = int(qn9[line])
		except ValueError:
			fix_qn(qn9,line,qn9[line])
						
	for line in range(len(qn10)):
	
		try:
			qn10[line] = int(qn10[line])
		except ValueError:
			fix_qn(qn10,line,qn10[line])
				
	for line in range(len(qn11)):
	
		try:
			qn11[line] = int(qn11[line])
		except ValueError:
			fix_qn(qn11,line,qn11[line])
			
	for line in range(len(qn12)):
	
		try:
			qn12[line] = int(qn12[line])
		except ValueError:
			fix_qn(qn12,line,qn12[line])	
																							
	return frequency,error,logint,dof,elower,gup,tag,qnformat,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12
	
#det_qns determines how many qns represent each state

def det_qns(qnformat):

	'''
	determines how many qns represent each state
	'''
	
	qns = int(str(qnformat[0])[-1:])

	return qns

#calc_q will dynamically calculate a partition function whenever needed at a given T.  The catalog file used must have enough lines in it to fully capture the partition function, or the result will not be accurate for Q.
	
def calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file):

	'''
	Dynamically calculates a partition function whenever needed at a given T.  The catalog file used must have enough lines in it to fully capture the partition function, or the result will not be accurate for Q.  This is perfectly fine for the *relative* intensities of lines for a given molecule used by this program.  However, absolute intensities between molecules are not remotely accurate.  
	'''

	Q = np.float64(0.0) #Initialize a float for the partition function
	
	if catalog_file=='acetone.cat':
	
		Q = 2.91296*10**(-7)*T**6 - 0.00021050085*T**5 + 0.05471337*T**4 - 5.5477*T**3 + 245.28*T**2 - 2728.3*T + 16431 #Hard code for Acetone
		
	elif catalog_file=='sh.cat':
	
		Q = 0.000000012549467*T**4 - 0.000008528126823*T**3 + 0.002288160909445*T**2 + 0.069272946237033*T + 15.357239728157400
 #Hard code for SH.  Completely unreliable below 2.735 K or above 300 K.

	elif catalog_file=='h2s.cat':
	
		Q = -0.000004859941547*T**3 + 0.005498622332982*T**2 + 0.507648423477309*T - 1.764494755639740 #Hard code for H2S.  Completely unreliable below 2.735 K or above 300 K.
	
	elif catalog_file=='hcn.cat':
	
		Q = -1.64946939*10**-9*T**4 + 4.62476813*10**-6*T**3 - 1.15188755*10**-3*T**2 + 1.48629408*T + .386550361
		
	elif catalog_file=='hc9n.cat':
	
		Q = 71.730808*T + 0.040659
		
	elif catalog_file=='hc7n.cat':
	
		Q = 36.949992*T + 0.135605
		
	elif 'methanol.cat' in catalog_file.lower() or 'ch3oh.cat' in catalog_file.lower() or 'ch3oh_v0.cat' in catalog_file.lower() or 'ch3oh_v1.cat' in catalog_file.lower() or 'ch3oh_v2.cat' in catalog_file.lower() or 'ch3oh_vt.cat' in catalog_file.lower():
	
		Q = 4.83410*10**-11*T**6 - 4.04024*10**-8*T**5 + 1.27624*10**-5*T**4 - 1.83807*10**-3*T**3 + 2.05911*10**-1*T**2 + 4.39632*10**-1*T -1.25670
		
	elif catalog_file.lower()=='13methanol.cat' or catalog_file.lower()=='13ch3oh.cat':
	
		Q = 0.000050130*T**3 + 0.076540934*T**2 + 4.317920731*T - 31.876881967
		
	elif catalog_file.lower()=='c2n.cat' or catalog_file.lower()=='ccn.cat':
		
		Q = 1.173755*10**(-11)*T**6 - 1.324086*10**(-8)*T**5 + 5.99936*10**(-6)*T**4 - 1.40473*10**(-3)*T**3 + 0.1837397*T**2 + 7.135161*T + 22.55770
		
	elif catalog_file.lower()=='ch2nh.cat':
	
		Q = 1.2152*T**1.4863
	
	else:
	
		nstates = elower.size #Determine the number of total states in the raw cat file
	
		combined_array = np.empty(shape=(nstates,qns+1)) #Set up an array that has [J, ka, kc, Elower]

		if (qns == 1):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = elower[i]

		if (qns == 2):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = elower[i]
	
		if (qns == 3):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = elower[i]
			
		if (qns == 4):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = elower[i]			

		if (qns == 5):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = qn11[i]
				combined_array[i][5] = elower[i]	
			
		if (qns == 6):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, QN12, Elower]
		
				try:
					combined_array[i][0] = qn7[i]
				except ValueError:
					print('I choked at index {}.' .format(i))
					quit()
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = qn11[i]
				combined_array[i][5] = qn12[i]
				combined_array[i][6] = elower[i]									
		
		temp = list(set(map(tuple,combined_array))) #Don't know HOW this works, but it does: sorts through the array and removes all duplicate entries, so that only a single entry remains for each set of quantum numbers.
	
		ustates = len(temp) #Number of unique lower states
	
		e_index = qns
	
		for i in range(ustates):
	
			J = temp[i][0] #Extract a J value from the list
			E = temp[i][qns] #Extract its corresponding energy
		
			Q += (2*J+1)*exp(np.float64(-E/(kcm*T))) #Add it to Q
			
		#result = [Q,ustates] #can enable this function to check the number of states used in the calculation, but note that this will break calls to Q further down that don't go after element 0.
	
	return Q

#scale_temp scales the simulated intensities to the new temperature

def scale_temp(int_sim,qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,CT,catalog_file):

	'''
	Converts linear intensities at one temperature to another.
	'''

	scaled_int = np.empty_like(int_sim)
	
	Q_T = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file)
	Q_CT = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file)

	
	scaled_int = int_sim * (Q_CT/Q_T) * (CT/T) * exp(-(((1/T)-(1/CT))*elower)/0.695)
	
# 	for i in range(len(scaled_int)):
# 	
# 		if catalog[1][i] > error_limit:
# 		
# 			scaled_int[i] = 0.0

	return scaled_int

#convert_int converts the intensity to not log units

def convert_int(logint):

	'''
	Converts catalog logarithmic intensity units to linear ones
	'''

	intensity = np.empty_like(logint)	
	
	intensity = 10**(logint)
	
	return intensity
	
#simulates Gaussian profiles after intensities are simulated.

def sim_gaussian(int_sim,freq,linewidth):

	'''
	Simulates Gaussian profiles for lines, after the intensities have been calculated.  Tries to be smart with how large a range it simulates over, for computational resources.  Includes a thermal cutoff for optically-thick lines.
	'''
	
	freq_gauss = []

	for x in range(int_sim.shape[0]):
	
		l_f = linewidth*freq[x]/ckm #get the FWHM in MHz
	
		min_f = freq[x] - 10*l_f #get the frequency 10 FWHM lower
		max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
		
		if res_kHz==True and res_kms==True:
		
			print('You have both the res_kHz and res_kms flags set to true.  The program will default back to the points per line value, currently {}, until you set one of these flags to False.' .format(npts_line))
	
			res_pnts = l_f / npts_line #determine a resolution element (15 points across the line)
			
		elif res_kHz == True and res_kms == False:
		
			res_pnts = res/1000 
			
		elif res_kms == True and res_kHz == False:
		
			res_pnts = (res*freq[x]/ckm) #get the frequency resolution from the desired velocity
		
		else:
		
			res_pnts = l_f / npts_line #determine a resolution element (15 points across the line)	
	
		freq_line = np.arange(min_f,max_f,res_pnts)
	
		freq_gauss.extend(freq_line)
	
	int_gauss = [0.0] * len(freq_gauss)
	
	freq_gauss.sort()
	
	start_time = tm.time()
	
	alerted = False

	for x in range(int_sim.shape[0]):
	
		if abs(int_sim[x]) < rms/10:
		
			continue
	
		telapsed = tm.time() - start_time
		
		if telapsed > 5 and alerted == False and quietflag == False:
		
			tstep = telapsed/x
			
			ttotal = (tstep * int_sim.shape[0])/60
		
			print('\nYou have asked for a computationally-expensive simulation.  Either wait for it to finish, narrow up your frequency range by setting ll or ul, or reduce the resolution.  Use quiet() to suppress further messages.\n')
			
			if ttotal < 0.2:
			
				print('Your simulation will probably finish in: a few seconds.')
				
			elif ttotal < 1.0:
			
				print('Your simulation will probably finish in: a minute or three.')
				
			elif ttotal < 2.0:
			
				print('Your simulation will probably finish in: a few minutes.')		
				
			elif ttotal < 5.0:
			
				print('Your simulation will probably finish in: go get a cup of coffee.')
				
			elif ttotal < 10.0:
			
				print('Your simulation will probably finish in: work on something else for a while.')
				
			elif ttotal < 30.0:
			
				print('Your simulation will probably finish in: watch an episode or two on Netflix.')	
				
			else:
			
				print('Your simulation will probably finish in: press ctrl+c and set some limits or lower the resolution.')																			
																						
			
			alerted = True 
	
		l_f = linewidth*freq[x]/ckm #get the FWHM in MHz
	
		c = l_f/2.35482

		int_gauss += int_sim[x]*exp(-((freq_gauss - freq[x])**2/(2*c**2)))
		
	try:
		int_gauss[int_gauss > thermal] = thermal
	except TypeError:
		pass	
	
	return(freq_gauss,int_gauss)

#write_spectrum writes out the current freq and intensity to output_file

def write_spectrum(x,output_file):

	'''
	Will write out the simulation frequency and intensity for the currently active simulation ('current'), the summed spectrum from sum_stored() ('sum'), or any stored simulation 'x' to output_file (which must be given as a string).  
	'''
	
	if x == 'current':
	
		freq_tmp = freq_sim
		int_tmp = int_sim
		
	elif x == 'sum':
	
		freq_tmp = freq_sum
		int_tmp = int_sum
		
	else:
	
		try:
			freq_tmp = sim[x].freq_sim
		except KeyError:
			print('\nOops: A spectrum called {} does not exist.  Either correct the typo (type sim to see the current spectrum labels), or remember that the syntax is write_spectrum(label,outputfile)' .format(x))
			return 
		int_tmp = sim[x].int_sim

	if gauss == True:
			
		with open(output_file, 'w') as output: 
					
			output.write('{} {}\n' .format(freq_tmp[0],int_tmp[0]))
				
		with open(output_file, 'a') as output:
					
			for h in range(len(freq_tmp)):
			
				output.write('{} {}\n' .format(freq_tmp[h],int_tmp[h]))					
	
	else:
				
		with open(output_file, 'w') as output: 
					
			output.write('{} {}\n' .format(freq_tmp[0],int_tmp[0]))
				
		with open(output_file, 'a') as output:
		
			for h in range(freq_tmp.shape[0]):
					
				output.write('{} {}\n' .format(freq_tmp[h],int_tmp[h]))									

#run_sim runs the simulation.  It's a meta routine, so that we can update later

def run_sim(freq,intensity,T,dV,C):

	'''
	Runs a full simulation accounting for the currently-active T, dV, S, and vlsr values, as well as any thermal cutoff for optically-thick lines
	'''
	
	np.seterr(under='ignore')
	np.seterr(over='ignore')
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file)

	numerator = (C)*(8*3.14159**3)*(frequency*1E6)*(sijmu)*(1-((exp((h*frequency*1E6)/(k*T))-1)/(exp((h*frequency*1E6)/(k*Tbg))-1)))*eta
	
	denominator = 1.06447*dV*Q*(exp(eupper/(kcm*T)))*(3*k)*1E48

	int_temp = numerator/denominator
		
	int_temp = trim_array(int_temp,frequency,ll,ul)		
	
	freq_tmp = trim_array(freq,frequency,ll,ul)
	
	int_temp[int_temp > thermal] = thermal	
	
	if gauss == True:

		freq_sim,int_sim = sim_gaussian(int_temp,freq_tmp,dV)
		
	else:
	
		#int_temp[int_temp > thermal] = thermal
		freq_sim = freq_tmp
		int_sim = int_temp
		
	return freq_sim,int_sim
	
#check_Q prints out Q at a given temperature x

def check_Q(x):

	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,x,catalog_file)
	
	print('Q({}) = {:.0f}' .format(x,Q))

#trim_array trims any given input array to the specified frequency ranges

def trim_array(array,frequency,ll,ul):

	'''
	trims any given input array to the specified frequency ranges.  Ranges can be 
	'''

	if type(ll) == int or type(ll) == float:
	
		tmp_ll = [ll]
		tmp_ul = [ul]
		
	else:
	
		tmp_ll = list(ll)
		tmp_ul = list(ul)
	
	foo = 0
	
	trimmed_array = np.array([])
		
	for z in range(len(tmp_ll)):
	
		try:
			i = np.where(frequency > tmp_ll[z])[0][0] 	#get the index of the first value above the lower limit
		except IndexError:
			if frequency[-1] < tmp_ll[z]:
				continue
			else:
				i = 0									#if the catalog begins after the lower limit
			
		try:
			i2 = np.where(frequency > tmp_ul[z])[0][0]	#get the index of the first value above the upper limit	
		except IndexError:
			i2 = len(frequency)							#if the catalog ends before the upper limit is reached		
			
		if foo == 0:
		
			trimmed_array = np.copy(array[i:i2])
			foo = 1
			
		else:
		
			trimmed_array = np.append(trimmed_array,array[i:i2])
	
	return trimmed_array		
	
#mod_T changes the temperature, re-simulates, and re-plots	
	
def modT(x):

	'''
	Modifies the temperature value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('x needs to be a number.')
		return
		
	global T,freq_sim,int_sim
		
	T = float(x)
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')
	
#modC changes the column density, re-simulates, and re-plots

def modC(x):

	'''
	Modifies the scaling value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('x needs to be a number.')
		return
		
	global C,freq_sim,int_sim
		
	C = x
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')
	
#moddV changes the velocity width, re-simulates, and re-plots

def moddV(x):

	'''
	Modifies the dV value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('dV needs to be a number.')
		return
		
	global dV,freq_sim,int_sim
	
	dV = x
	
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
		
	freq_sim,int_sim = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')
	
#modVLSR changes the LSR velocity, re-simulates, and re-plots

def modVLSR(x):

	'''
	Modifies the vlsr value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('vlsr needs to be a number.')
		return	
		
	global vlsr,freq_sim,int_sim,frequency
	
	vlsr = x
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')		

#modV is an alias for modVLSR

def modV(vlsr):

	'''
	Modifies the vlsr value of the current simulation to the value given
	'''

	modVLSR(vlsr)	
	
#make_plot makes the plot!

def make_plot():

	'''
	Generates a plot of the currently-active molecular simulation, as well as any laboratory data or observations which are loaded.  This will *not* restore any overplots from a previously-closed plot.
	'''

	global fig,ax,line1,line2,freq_sim,intensity,int_sim

	plt.ion()	

	fig = plt.figure()
	ax = fig.add_subplot(111)

	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Intensity (Probably Arbitrary)')

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0)
	except:
		pass

	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
		
	fig.canvas.draw()
	
	save_results('last.results')
	
#obs_off turns off the observations
	
def obs_off():

	'''
	turns off the laboratory data or observations on the plot
	'''

	try:
		clear_line('obs')
		save_results('last.results')
		return
	except:
		print('The observations are already off.  You can turn them on with obs_on()')
		return
	
	try:
		lines['obs']
		save_results('last.results')
	except:
		print('There are no observations loaded into the program to turn off.  Load in obs with read_obs()')
		return	
			
#obs_on turns on the observations
	
def obs_on():

	'''
	turns on the laboratory data or observations on the plot
	'''	
	
	try:
		lines['obs']
		return
	except:
		pass

	try:
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()
		save_results('last.results')
	except:
		print('There are no observations loaded into the program to turn on.  Load in obs with read_obs()')
		return
			
#read_obs reads in observations or laboratory spectra and populates freq_obs and int_obs

def read_obs(x):

	'''
	reads in observations or laboratory spectra and populates freq_obs and int_obs.  will detect a standard .ispec header from casaviewer export, and will apply a GHz flag if necessary, as well as populating the coords variable with the coordinates from the header.
	'''

	global spec, coords, GHz

	spec = x

	obs = read_cat(x)
	
	#check to see if these are casa spectra
	
	if obs[0].split(':')[0] == '#title':
	
		i = 0
		j = 0
	
		while i == 0:
		
			if obs[j].split(':')[0] == '#xLabel':
		
				if obs[j].split('[')[1].strip(']\n') == 'GHz':
				
					GHz = True
				
			if obs[j].split(':')[0] == '#region (world)':
			
				coords = obs[j].split(':')[1].strip('\n')
			
			if obs[j][0] != '#':
			
				i = 1
				
			j += 1		
		
		del obs[:j+1]
	
	global freq_obs,int_obs

	freq_obs = []
	int_obs = []

	for x in range(len(obs)):

		freq_obs.append(float(obs[x].split()[0]))
		int_obs.append(float(obs[x].split()[1].strip('\n')))
		
	if GHz == True:
	
		freq_obs[:] = [x*1000.0 for x in freq_obs]
		
	try:		
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0)
	except:
		return
		
#close closes the currently open plot

def close():

	'''
	Closes the currently open plot window.
	'''
	
	plt.close()	
	
	save_results('last.results')
	
#store saves the current simulation parameters for recall later.  *Not* saved as a Gaussian. 'x' must be entered as a string with quotes.

def store(x=None):

	'''
	saves the current simulation parameters for recall later.  *Not* saved as a Gaussian. 'x' must be entered as a string with quotes.  If used with just store(), it defaults the name to everything up to the first period in the catalog_file name.
	'''
	
	if x == None:
	
		x = '{}' .format(catalog_file.split('.')[0]) 
	
	sim[x] = Molecule(x,catalog_file,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim)
	
	if auto_update == True:
	
		sum_stored()
		overplot_sum()
		
	if any(line for line in ax.lines if line.get_label()==x):
		overplot(x)	
	
	save_results('last.results') 
	
#recall wipes the current simulation and re-loads a previous simulation that was stored with store(). 'x' must be entered as a string with quotes. This will close the currently-open plot.

def recall(x):

	'''
	wipes the current simulation and re-loads a previous simulation that was stored with store(). 'x' must be entered as a string with quotes. This will close the currently-open plot.
	'''

	save_results('last.results')

	global elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,S,dV,T,vlsr,frequency,freq_sim,intensity,int_sim,current,catalog_file

	current = sim[x].name
	elower = sim[x].elower
	eupper = sim[x].eupper
	qns = sim[x].qns
	logint = sim[x].logint
	qn7 = sim[x].qn7
	qn8 = sim[x].qn8
	qn9 = sim[x].qn9
	qn10 = sim[x].qn10
	qn11 = sim[x].qn11
	qn12 = sim[x].qn12
	S = sim[x].S
	dV = sim[x].dV
	T = sim[x].T
	vlsr = sim[x].vlsr
	frequency = sim[x].frequency
	freq_sim = sim[x].freq_sim
	intensity = sim[x].intensity
	int_sim = sim[x].int_sim
	catalog_file = sim[x].catalog_file

	try:
		clear_line('current')
	except:
		pass
		
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm
	
	freq_sim,int_sim=run_sim(tmp_freq,intensity,T,dV,C)	
		
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)	
		
	try:
		plt.get_fignums()[0]	
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()
	except:	
		make_plot()	
		
	save_results('last.results')	
	

#overplot overplots a previously-stored simulation on the current plot in a color other than red, but does not touch the simulation active in the main program. 'x' must be entered as a string with quotes.

def overplot(x,cchoice=None):

	'''
	overplots a previously-stored simulation on the current plot in a color other than red, but does not touch the simulation active in the main program. 'x' must be entered as a string with quotes.  Defaults to choosing a color from a large pool of random colors.  you can specify a color by giving it a string as the second variable.  The string is either a hex color code, or one of the few matplotlib defaults like 'red' 'green' 'blue'.
	'''

	global elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,S,dV,T,vlsr,frequency,freq_sim,intensity,int_sim,current,fig,ax

	#store the currently-active simulation in a temporary holding cell

	freq_temp = freq_sim 
	int_temp = int_sim 
	
	#pull the old simulation out of storage
	
	freq_sim = sim[x].freq_sim
	int_sim = sim[x].int_sim

	if any(line for line in ax.lines if line.get_label()==sim[x].name):
		if cchoice == None:
			line_color = [line.get_color() for line in ax.lines if line.get_label()==sim[x].name][0]
		else:
			line_color = cchoice
		line_style = [line.get_linestyle() for line in ax.lines if line.get_label()==sim[x].name][0]
		clear_line(sim[x].name)
	else:
		if cchoice == None:
			line_color = next(colors)
		else:
			line_color = cchoice
		line_style = '-'		

#Not working at the moment
		
# 	if any(line for line in ax.lines if line.get_color()==line_color):
# 		match = False
# 		i = 1
# 		while match == False:
# 			#first we just try changing the colors.
# 			line_color = next(colors)
# 			if any(line for line in ax.lines if line.get_color()==line_color) == False:
# 				match = True
# 			line = [line for line in ax.lines if line.get_color()==line_color][0]
# 			line_style_tmp = line.get_linestyle()
# 			line_style = next(styles)
# 			#if the line style hasn't been used before for that color, we have a good match and we exit
# 			if line_style != line_style_tmp:
# 				match = True
# 			#otherwise we go to the next line style and try again.  We loop long enough to try all the combinations, and then if we can't find any unique combinations, we go back and just re-use the color w/ the basic line.  Too bad.  Add more colors.
# 			else:
# 				line_style = next(styles)
# 				if i > 4:
# 					line_style = '-'
# 					match = True			
	
	lines[sim[x].name] = ax.plot(freq_sim,int_sim,color = line_color, linestyle=line_style, label = sim[x].name)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	freq_sim = freq_temp
	int_sim = int_temp
	
	save_results('last.results')
		
#load_mol loads a new molecule into the system.  Make sure to store the old molecule simulation first, if you want to get it back.  The current graph will be updated with the new molecule.  Catalog file must be given as a string.  Simulation will begin with the same T, dV, S, vlsr as previous, so change those first if you want.

def load_mol(x,format='spcat'):

	'''
	loads a new molecule into the system.  Make sure to store the old molecule simulation first, if you want to get it back.  The current graph will be updated with the new molecule.  Catalog file must be given as a string.  Simulation will begin with the same T, dV, S, vlsr as previous, so change those first if you want.
	'''

	global frequency,logint,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,first_run,tbg,sijmu
	
	
	if first_run == False:
		try:
			clear_line('current')
		except:
			pass
	
	current = x
	
	catalog_file = x
	
	raw_array = read_cat(catalog_file)

	catalog = splice_array(raw_array)

	frequency = np.copy(catalog[0])
	logint = np.copy(catalog[2])
	qn7 = np.asarray(catalog[14])
	qn8 = np.asarray(catalog[15])
	qn9 = np.asarray(catalog[16])
	qn10 = np.asarray(catalog[17])
	qn11 = np.asarray(catalog[18])
	qn12 = np.asarray(catalog[19])
	elower = np.asarray(catalog[4])
	qnformat = np.asarray(catalog[7])
	
	tbg = np.empty_like(elower)
	
	tbg.fill(2.7) #background temperature in the source, defaulting to CMB.  Can change this manually or with init_source()

	eupper = np.empty_like(elower)

	eupper = elower + frequency/29979.2458

	qns = det_qns(qnformat) #figure out how many qns we have for the molecule

	intensity = convert_int(logint)
	
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,300,catalog_file)
	
	print('\nWarning: Accurate absolute brightness temperatures depend on having a complete spectral catalog to calcualte  an accurate partition function.  Below is the value that is calcualted at 300 K from the current catalog.  If this is not accurate, the results need to be adjusted accordingly.\n')
	print('Q(300) = {:.0f}' .format(Q))
	
	sijmu = (exp(np.float64(-(elower/0.695)/300)) - exp(np.float64(-(eupper/0.695)/300)))**(-1) * ((10**logint)/frequency) * ((4.16231*10**(-5))**(-1)) * Q
	
	freq_sim,int_sim=run_sim(tmp_freq,intensity,T,dV,C)
	
	#check if a plot is already open.  If it is, nothing happens.  If there are no plots open, plt.get_fignums()[0] is empty and throws an IndexError, which the exception catches and just makes a fresh plot.  If it's the first time the program has been run since it was loaded, it ignores this check and just makes a new plot.  
	
	if first_run == True:
		make_plot()
		first_run = False
		return 
	else:
		try:
			plt.get_fignums()[0]
		except:	
			make_plot()
			return 
		
	#if there is a plot open, we just update the current simulation
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()	
	
	save_results('last.results')
	
	return
		
#clear_line removes a line labeled 'x' from the current plot window.  x must be a string

def clear_line(x):

	'''
	removes a line labeled 'x' from the current plot window.  x must be a string
	'''	
	
	try:
		line = lines.pop(x)
	except:
		return
	
	try:
		line.remove()
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()	

	except:
		line[0].remove()
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()	
		

#clear is an alias for clear_line

def clear(x):

	'''
	Alias for clear_line()
	'''

	clear_line(x)
	
#save_results prints out a file with all of the parameters for each molecule *which has been stored.*  It will not print the active molecule unless it has been stored. 'x' must be a string, and the output will go to there.

def save_results(x):

	'''
	save_results prints out a file that contains at the top the output of status(), followed by all of the parameters for each molecule which has been stored.  This file can then be used with restore() to restore the parameter space and keep going.
	'''

	with open(x, 'w') as output:
	
		output.write('viewspectrum.py version {}\n' .format(version))		
		output.write('saved: {}\n\n' .format(datetime.now().strftime("%m-%d-%y %H:%M:%S")))
		
		output.write('#### Active Simulation ####\n\n')
	
		output.write('molecule:\t{}\n' .format(current))
		output.write('obs:\t{}\n' .format(spec))
		output.write('T:\t{} K\n' .format(T))
		output.write('C:\t{:.2f}\n' .format(C))
		output.write('dV:\t{:.2f} km/s\n' .format(dV))
		output.write('VLSR:\t{:.2f} km/s\n' .format(vlsr))
		output.write('ll:\t{} MHz\n' .format(ll))
		output.write('ul:\t{} MHz\n' .format(ul))
		output.write('CT:\t{} K\n' .format(CT))
		output.write('gauss:\t{}\n' .format(gauss))
		output.write('catalog_file:\t{}\n' .format(catalog_file))
		output.write('thermal:\t{} K\n' .format(thermal))
		output.write('GHz:\t{}\n' .format(GHz))
		output.write('rms:\t{}\n\n' .format(rms))
	
		output.write('#### Stored Simulations ####\n\n')
		
		output.write('Molecule\tT(K)\tS\tdV\tvlsr\tCT\tcatalog_file\n')
	
		for molecule in sim:
		
			output.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n' .format(sim[molecule].name,sim[molecule].T,sim[molecule].C,sim[molecule].dV,sim[molecule].vlsr,sim[molecule].CT,sim[molecule].catalog_file))
			
		output.write('\n#### Active Graph Status ####\n\n')
		
		output.write('Label\tColor\tStyle\n')
		
		for line in ax.lines:
		
			output.write('{}\t{}\t{}\n' .format(line.get_label(),line.get_color(),line.get_linestyle()))	
	
#status prints the current status of the program and the various key variables.

def status():

	'''
	prints the current status of the program and the various key variables
	'''

	print('Current Molecule:\t {}' .format(current))
	print('Current Catalog:\t {}' .format(catalog_file))
	print('Current lab or observation: \t {}' .format(spec))
	print('T: \t {} K' .format(T))
	print('S: \t {}' .format(S))
	print('dV: \t {} km/s' .format(dV))
	print('VLSR: \t {} km/s' .format(vlsr))
	print('ll: \t {} MHz' .format(ll))
	print('ul: \t {} MHz' .format(ul))
	print('CT: \t {} K' .format(CT))
	print('gauss: \t {}' .format(gauss))
		
#sum_stored creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.  It's done from scratch because the frequency axes in freq_sim stored for each molecule will not necessarily be the same, so co-adding is difficult without re-gridding, which well, maybe later.	

def sum_stored():

	'''
	Creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.
	'''
	global freq_sum, int_sum
	
	freq_gauss = []

	for x in sim:
	
		tmp_freq = np.copy(sim[x].frequency)
		
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
		
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
		
		tmp_int = np.copy(sim[x].intensity)
		
		tmp_int_trimmed = trim_array(tmp_int,tmp_freq,ll,ul)

		for y in range(len(tmp_int_trimmed)):
	
			l_f = sim[x].dV*tmp_freq_trimmed[y]/ckm #get the FWHM in MHz
	
			min_f = tmp_freq_trimmed[y] - 10*l_f #get the frequency 10 FWHM lower
			max_f = tmp_freq_trimmed[y] + 10*l_f #get the frequency 10 FWHM higher
	
			res_pnts = l_f / 15 #determine a resolution element (15 points across the line)
	
			freq_line = np.arange(min_f,max_f,res_pnts)
	
			freq_gauss.extend(freq_line)
	
	int_gauss = [0.0] * len(freq_gauss)
	
	freq_gauss.sort()
	
	del tmp_freq, tmp_freq_trimmed, tmp_int, tmp_int_trimmed
		
	for x in sim:
	
		tmp_freq = np.copy(sim[x].frequency)
		
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
		
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
		
		tmp_int = np.copy(sim[x].intensity)
		
		tmp_int = scale_temp(tmp_int,sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,sim[x].T,sim[x].CT,sim[x].catalog_file)
		
		tmp_int_trimmed = trim_array(tmp_int,tmp_freq,ll,ul)
		
		tmp_int_trimmed *= sim[x].S
		
		tmp_int_trimmed[tmp_int_trimmed > thermal] = thermal
	
		for y in range(len(tmp_int_trimmed)):
		
			if abs(tmp_int_trimmed[y]) < rms/10:
			
				continue
		
			l_f = sim[x].dV*tmp_freq_trimmed[y]/ckm #get the FWHM in MHz
			
			c = l_f/2.35482

			int_gauss += tmp_int_trimmed[y]*exp(-((freq_gauss - tmp_freq_trimmed[y])**2/(2*c**2)))
			
	int_gauss[int_gauss > thermal] = thermal
	
	freq_sum = freq_gauss
	int_sum = int_gauss		

#overplot_sum overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.

def overplot_sum():

	'''
	Overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.
	'''	
	
	line_color = '#00FF00'
	
	if any(line for line in ax.lines if line.get_label()=='sum'):
		clear_line('sum')
	
	lines['sum'] = ax.plot(freq_sum,int_sum,color = line_color, label = 'sum', gid='sum', linestyle = '-',zorder=25)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()	

	
#restore restores the state of the program from a save file, loading all stored spectra into memory, loads the previously active simulation into current, and restores the last active graph. x is a string with the filename of the restore file. The catalog files must be present, and named to match those in the save file.

def restore(x):

	'''
	restores the state of the program from a save file, loading all stored spectra into memory, loads the previously active simulation into current, and restores the last active graph. x is a string with the filename of the restore file. The catalog files must be present, and named to match those in the save file.
	
	This procedure attempts to correct for any backward compatability issues with old versions of the program.  Usually the restore will proceed without issue and will warn the user if there were issues it corrected.  The simplest way to update a restore file is to save it with the latest version of the program after a successful load.  Most of the time, the backwards compatability issues are caused simply by missing meta-data that have been added in later version of the program.  In this case, the default values are simply used.
	'''

	global frequency,logint,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,T,dV,C,vlsr,ll,ul,CT,gauss,first_run,thermal,sim,GHz,rms
	
	#empty out the previously-stored simulations
	
	sim = {}

	#read in the save file as an array line by line
	
	restore_array = []
	
	GHz = False
	
	try:
		with open(x) as input:
			for line in input:
				restore_array.append(line)
	except TypeError:
		print('Input must be a string in \'\'.')
		return
	except FileNotFoundError:
		print('This is not the file you are looking for.')
		return
	
	#check if the file that was read it is actually a savefile for viewspectrum.py
	
	if restore_array[0].split()[0] != 'viewspectrum.py':
		print('The file is not a viewspectrum.py save file, has been altered, or was created with an older version of the program.  In any case, I can\'t read it, sorry.')
		return	
		
	#let's grab the date and time, so we can print some nice information to the terminal
	
	restore_date = restore_array[1].split()[1]
	restore_time = restore_array[1].split()[2]	
	
	#do a version check
	
	restore_version = float(restore_array[0].split()[2])
	
	if restore_version != version:
	
		print('Warning: {} was created using v{}.  The current version is v{}.  Type help(restore) for more information.' .format(x,restore_version,version))   
		
	#separate out the sections into their own arrays

	active_array = []	
	stored_array = []
	graph_array = []
	
	#figure out where each section starts
	
	active_index = restore_array.index('#### Active Simulation ####\n')
	stored_index = restore_array.index('#### Stored Simulations ####\n')
	graph_index = restore_array.index('#### Active Graph Status ####\n')
	
	for i in range(active_index+2,stored_index-1):
	
		active_array.append(restore_array[i])
		
	for i in range(stored_index+3,graph_index-1):
	
		stored_array.append(restore_array[i])
		
	for i in range(graph_index+3,len(restore_array)):
	
		graph_array.append(restore_array[i])
		
	#just to be safe, let's set the upper limits, lower limits, gaussian toggles, thermal values, and GHz flag now.
	
	#what is the rms level set at?
	
	try:
		rms = float(active_array[13].split('\t')[1].strip('\n'))
	except IndexError:
		print('The restore file does not have an rms value in it.  It was probably generated with a previous version of the program. The restore can proceed, but it is recommended that you re-save the restore file with the latest version of the program.')
	
	#are we simulating Gaussians?
	
	if active_array[9].split('\t')[1].strip('\n') == 'True':
		gauss = True
	else:
		gauss = False
		
	#are observations read in in GHz?
	
	if active_array[12].split('\t')[1].strip('\n') == 'True':
		GHz = True
	else:
		GHz = False	
	
	#set the lower limit.  requires some logical to differentiate between a single value and a list	
		
	try:
		ll = float(active_array[6].split('\t')[1].strip(' MHz\n'))
	except ValueError:
		ll = []
		tmp_str = active_array[6].split('\t')[1].strip(' MHz\n').strip(']').strip('[').split(',')
		for line in range(len(tmp_str)):
			ll.append(float(tmp_str[line]))
			
	#set the upper limit.  requires some logical to differentiate between a single value and a list		
	
	try:	
		ul = float(active_array[7].split('\t')[1].strip(' MHz\n'))
	except ValueError:
		ul = []
		tmp_str = active_array[7].split('\t')[1].strip(' MHz\n').strip(']').strip('[').split(',')
		for line in range(len(tmp_str)):
			ul.append(float(tmp_str[line]))
	
	
	thermal = float(active_array[11].split('\t')[1].strip(' K\n'))
	
	#OK, now time to do the hard part.  As one always should, let's start with the middle part of the whole file, and load and then store all of the simulations.
	
	for i in range(len(stored_array)):
	
		name = stored_array[i].split('\t')[0]
		T = float(stored_array[i].split('\t')[1])
		C = float(stored_array[i].split('\t')[2])
		dV = float(stored_array[i].split('\t')[3])
		vlsr = float(stored_array[i].split('\t')[4])
		CT = float(stored_array[i].split('\t')[5])
		catalog_file = str(stored_array[i].split('\t')[6]).strip('\n').strip()
		
# 		try:
		first_run = True
		try:
			foo = load_mol(catalog_file)
			if foo == False:
				raise ValueError()
		except ValueError:
			continue
# 		except FileNotFoundError:
# 			print('I was unable to locate the catalog file {} for molecule entry {}.  Sorry.' .format(catalog_file,name))
# 			return
			
		store(name)
		
		close()
	
	#Now we move on to loading in the currently-active molecule
	
	catalog_file = active_array[10].split('\t')[1].strip('\n')
	try:
		obs = active_array[1].split('\t')[1].strip('\n')
		read_obs(obs)
	except:
		pass
	T = float(active_array[2].split('\t')[1].strip(' K\n'))
	S = float(active_array[3].split('\t')[1].strip('\n'))
	dV = float(active_array[4].split('\t')[1].strip(' km/s\n'))
	vlsr = float(active_array[5].split('\t')[1].strip(' km/s\n'))
	CT = float(active_array[8].split('\t')[1].strip(' K\n'))
	current = active_array[0].split('\t')[1].strip('\n')
	
	try:
		first_run = True
		load_mol(catalog_file)
	except FileNotFoundError:
		print('I was unable to locate the catalog file {} for molecule entry {}.  Sorry.' .format(catalog_file,name))
		return
		
	#And finally, overplot anything that was on the plot previously
	
	for i in range(len(graph_array)):
	
		name = graph_array[i].split('\t')[0]
		line_color = graph_array[i].split('\t')[1]
		line_style = graph_array[i].split('\t')[2].strip('\n')
		
		if name == 'obs':
		
			continue
			
		elif name == 'current':
		
			continue
			
		elif name == 'sum':
		
			sum_stored()
			overplot_sum()
			continue			
			
		else:
			
			lines[name] = ax.plot(sim[name].freq_sim,sim[name].int_sim,color = line_color, linestyle=line_style, label = name)	
			
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()	
		
	#If we made it here, we were successful, so let's print out what we did
	
	print('Successfully restored from file {} which was saved on {} at {}.' .format(x,restore_date,restore_time))	
	
#fix_legend allows you to change the legend to meet its size needs.

def fix_legend(x,lsize):

	'''
	Modifies the legend to have x columns.  lsize can be {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or an int or float.
	'''

	plt.legend(ncol=x,prop={'size':lsize})
	
	fig.canvas.draw()	
	
#purge permanently removes a stored simulation from memory.  Can't undo this one, folks.

def purge(x):

	'''
	permanently removes a stored simulation x from memory.  Can't undo this one, folks.
	'''
	
	try:
		del sim[x]
	except KeyError:
		print('No simulation with that key in the simulation dictionary.  Type \'sim\' to see an (ugly) list of stored simulations.')
	
#use_GHz changes the read-in default for observations from MHz to GHz.  This will just convert the read-in frequencies to MHz.

def use_GHz():

	'''
	changes the read-in default for observations from MHz to GHz.  This will just convert the read-in frequencies to MHz.
	'''

	global GHz
	
	GHz = True

#calc_beam returns an array of beam sizes for a telescope at each point.  This is a constant value if mode = 'A' for the synthesized beam.

def calc_beam(frequency):

	'''
	returns an array of beam sizes for a telescope at each point.  This is a constant value if mode = 'A' for the synthesized beam.
	'''
	
	beam_size = np.copy(frequency)
	
	if mode == 'A':
	
		beam_size.fill(synth_beam)
		
	else:
	
		beam_size = (1.22*(3*10**8/(frequency*10**6))/dish_size) * 206265

	return beam_size
	
#calc_bcorr calculates and returns the beam dilution correction factor at each frequency, given the source size and the beam size

def calc_bcorr(frequency,beam_size):

	'''
	calculates and returns the beam dilution correction factor at each frequency, given the source size and the beam size
	'''
	
	bcorr = np.copy(frequency)
	
	bcorr = (source_size**2 + beam_size**2)/source_size**2
	
	return bcorr
	
#apply_telescope takes a simulation and applies a correction factor to the intensity based on a provided telescope configuration, column density, and source structure

def apply_telescope(eupper,T,freq,dV,eta,NT,tbg):

	'''
	takes a simulation and applies a correction factor to the intensity based on a provided telescope configuration, column density, and source structure
	'''
	
	Q_300 = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,300,catalog_file)
	
	sij = (exp(np.float64(-(elower/0.695)/300)) - exp(np.float64(-(eupper/0.695)/300)))**(-1) * ((10**logint)/frequency) * ((4.16231*10**(-5))**(-1)) * Q_300
	
	Q_T = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file)

	A = 3*k/(8*(np.pi**3))
	
	B = Q_T*np.exp(np.float64(eupper/T))/(freq*(1*10**6)*sij*(1*10**(-43)))
	
	C = 0.5*(np.pi/(np.log(2)))**0.5
	
	D1 = dV*(1*10**5)/eta

	D2 = (np.exp(h*freq*(1*10**6)/(k*T)) -1)
	
	D3 = (np.exp(h*freq*(1*10**6)/(k*tbg)) -1)
	
	D = D1/(1-(D2/D3))
	
	TA = NT/(A*B*C*D)	
	
	beam_size = calc_beam(frequency)
	
	bcorr = calc_bcorr(frequency,beam_size)
	
	TA /= bcorr

	return TA

#init_source initializes a pre-loaded set of conditions for specific sources

def init_source(source,size=0.0):

	'''
	Initalizes a pre-loaded set of conditions for specific sources.  Options are:
	SGRB2N: Requires the desired source size to be set with the size command.  Sets tbg according to Hollis et al. 2007 ApJ 660, L125 (probably not valid below 10 GHz) or other measurements at higher frequencies (XXX, YYY, ZZZ)
	'''
	global source_size,tbg,dish_size
	
	if source == 'SGRB2N':
	
		dish_size = 100.0
	
		source_size = 20.0  #The emitting region for the background continuum in SgrB2 is always 20"
		
		beam_size = calc_beam(frequency)
		
		bcorr_tbg = calc_bcorr(source_size,beam_size) #calculate the correction factor for the continuum
		
		tbg = (10**(-1.06*np.log10(frequency/1000) + 2.3))*bcorr_tbg
		
		for x in range(tbg.shape[0]):
	
			if frequency[x] > 60000:
		
				tbg[x] = 5.2
			
			if frequency[x] > 130000:
		
				tbg[x] = 6.5
			
			if frequency[x] > 230000:
		
				tbg[x] = 10.0
			
			if frequency[x] > 1000000:
		
				tbg[x] = 13.7		
		
		source_size = size

	return

#quiet suppresses warnings about computational time.  Can be used iteratively to turn it on and off.

def quiet():

	global quietflag

	if quietflag == False:
	
		quietflag = True
		
	elif quietflag == True:
	
		quietflag = False

#############################################################
#							Classes for Storing Results		#
#############################################################	

class Molecule(object):

	def __init__(self,name,catalog_file,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim):
	
		self.name = name
		self.catalog_file = catalog_file
		self.elower = elower
		self.eupper = eupper
		self.qns = qns
		self.logint = logint
		self.qn7 = qn7
		self.qn8 = qn8
		self.qn9 = qn9
		self.qn10 = qn10
		self.qn11 = qn11
		self.qn12 = qn12
		self.C = C
		self.dV = dV
		self.T = T
		self.CT = CT
		self.vlsr = vlsr
		self.frequency = frequency
		self.freq_sim = freq_sim
		self.intensity = intensity
		self.int_sim = int_sim

	
#############################################################
#							Run Program						#
#############################################################

sim = {} #dictionary to hold stored simulations

lines = {} #dictionary to hold matplotlib lines

freq_obs = [] #to hold laboratory or observational spectra
int_obs = []

freq_sim = [] #to hold simulated spectra
int_sim = [] 

freq_sum = [] #to hold combined spectra
int_sum = []

current = catalog_file

colors = itertools.cycle(['#ff8172','#514829','#a73824','#7b3626','#a8ac87','#8c8e64','#974710','#d38e20','#ce9a3a','#ae7018','#ac5b14','#64350f','#b18f59','#404040','#791304','#1f2161','#171848','#3082fe','#2c5b5e','#390083','#5c65f8','#6346fa','#3c3176','#1cf6ba','#c9bcf0','#90edfc','#3fb8ee','#b89b33','#e7d17b'])
styles = itertools.cycle(['-','--','-.',':'])
