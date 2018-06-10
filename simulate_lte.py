#!/usr/bin/env python

#############################################################
#						Revision History					#
#############################################################

# reads in an spcat catalog and displays a spectrum

# 1.0 - Project start
# 2.0 - ipython overhaul and simplification
# 2.1 - fixes bug with status and S/C conversion
# 2.2 - fixes bug with recall and sum_stored
# 2.3 - patch for 13ch3oh
# 2.4 - fixes bug with catalogs not at 300 K
# 2.6 - adds autoset functionality
# 2.7 - speeds up gaussian simulations
# 2.8 - further speeds up gaussian simulations and normalizes spectral resolution
# 2.9 - adds labeling ability (Eupper and Quantum Numbers) to current plot
# 3.0 - adds residual plotting ability
# 3.1 - adds residual saving ability
# 3.3 - removes labels (broken on some machines); adds ability to print out line information
# 3.4 - adds ability to do Gaussian fitting of data in the program
# 3.5 - adds ability to mass-generate a p-list for gauss fitting
# 3.6 - adds ability to convert the observations from Jy/beam to K, or K to Jy/beam.
# 3.7 - adds ability to simulate double doublets from cavity FTMW
# 3.8 - adds ability to read in frequencies to plot just as single intensity lines; or to do it manually with a list.  changes default plotting to steps, adds ability to switch back to lines.  
# 3.9 - adds ability to plot manual catalogs with a velocity offset
# 3.91 - minor change to how plots are titled
# 4.0 - changes the simulation to perform a proper 'radiative transfer' which first calculates an opacity, and then applies the appropriate optical depth correction
# 4.1 - added a K to Jy/beam converter that wasn't there before...
# 4.2 - adds Aij to print_lines() readout
# 4.3 - fixes edge case where two lines have same frequency in catalog for print_lines()
# 4.4 - minor warning fix
# 5.0 - adds ability to do velocity stacking
# 5.1 - adds ability to correct for beam dilution
# 5.2 - adds ability to save velocity stacked spectra
# 5.3 - bug fix for noisy data without lines in peak finding
# 5.4 - minor correction to calculation of optically thick lines
# 5.5 - added ability to cut out spectra around simulated stick spectra
# 5.6 - update to autoset_limits() to not simulate where there's no data between the absolute upper and lower bounds
# 5.7 - adds flag to run simulations using the Planck scale and a beam size to convert to Jy/beam
# 6.0 - major update to Tbg handling

#############################################################
#							Preamble						#
#############################################################

import sys

#Python version check

if sys.version_info.major != 3:

	print("This code is written in Python 3.  It will not execute in Python 2.7.  Exiting (sorry).")
	
	quit()

import numpy as np
from numpy import exp as exp
import time as tm
import warnings
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
from scipy.optimize import curve_fit
import peakutils
import math
#warnings.filterwarnings('error')

version = 6.0

h = 6.626*10**(-34) #Planck's constant in J*s
k = 1.381*10**(-23) #Boltzmann's constant in J/K
kcm = 0.69503476 #Boltzmann's constant in cm-1/K
ckm = 2.998*10**5 #speed of light in km/s
ccm = 2.998*10**10 #speed of light in cm/s
cm = 2.998*10**8 #speed of light in m/s

#############################################################
#							Warning 						#
#############################################################

print('\n--- MAJOR CHANGE STARTING IN VERSION 6.0 ---') 

print('\nThere has been a major update to the way the program treats background temperatures.  It now allows for different background temperatures at different frequencies, as well as permitting functionalized backgrounds.  As a result, the old system of simply setting Tbg = X will no longer function in Version 6.0+.  Please see the documentation for the new protocols.') 

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

gauss = True #toggle for simulating Gaussians or a stick spectrum.  Default is True.

dish_size = 100 #for use if beam corrections are desired; given in meters

source_size = 1E20 #for use if beam corrections are desired; given in arcseconds

eta = 1.0 #beam efficiency of the telescope.  Set this option manually, with configure_telescope(), or with init_telescope(). 

npts_line = 15 #default is 15 points across each line

res_kHz = False #if res_kHz is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of kHz

res_kms = False #if res_kms is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of km/s

res = 0.01 #resolution used in Gaussian simulation if res_kHz or res_kms is set to True.

cavity_ftmw = False #if set to True, simulates doubler doublets from the cavity FTMW. 

cavity_dV = 0.13 #sets the default cavity linewidth to 0.2 km/s

cavity_split = 0.826 #sets the default doppler splitting in the cavity to 0.826 km/s in each direction.

draw_style = 'steps' #can be toggled on and off for going between drawing steps and drawing lines between points using use_steps() and use_lines()

planck = False #flag to use planck scale.  If planck = True is enabled, a synthesized beam size must also be provided using synth_beam = [bmaj,bmin] below.

synth_beam = ['bmaj','bmin'] #to be used with planck = True conversions.  Will throw an error if you don't set it in the program.

sim = {} #dictionary to hold stored simulations

lines = {} #dictionary to hold matplotlib lines

tbg = [] #to hold background temperatures

############ Tbg Parameters ##############

#tbg_params is to hold the actual parameters used to calculate Tbg.  

	#If it is a constant, then it can be passed an integer.  tbg_type must be 'poly' and tbg_order must be an integer 0.  These are the defaults.  Other possibilities are described below.

tbg_params = 2.7

#tbg_type can be the following:

	#'poly' is a polynomial of order set by tbg_order = X, where X is the order and an integer.  If tbg_order = 0, tbg_params can be a float or a list with one value.  If tbg_order is greater than 0, then tbg_params must be a list of length = X+1.  So a first order polynomial needs two values [A,B] in the tbg_params: y = Ax + B.
	
	#'power' is a power law of the form Y = Ax^B + C.  tbg_params must be a list with three values [A,B,c]

tbg_type = 'poly'

#tbg_range can contain a list of paired upper and lower limits, themselves a length 2 list, for the sets of parameters in tbg_params to be used within.  If ranges are defined, any bit of the simulation not in the defined range defaults to 2.7 K.  float('-inf') or float('inf') are valid range values.

tbg_range = []

#Some examples:

#To have three different frequency ranges (100000-120000 MHz, 150000-160000, and 190000-210000 MHz), all with their own tbg constants of 27, 32, and 37, the following must be input:

	#tbg_params = [27,32,37]
	#tbg_range = [[100000,120000],[150000,160000],[190000,210000]]
	
#To have a power law across the entire simulation, with Y = Ax^B = C and A = 2, B = 1.2, and C = 0:

	#tbg_params = [2,1.2,0]
	#tbg_type = 'power'

#To have three polynomials, of orders 1, 3, and 4, over the three different ranges above, you'd need:

	#tbg_params = [[1.2,5],[1.7,1.3,2,4],[2,4,-0.7,0.8,1.2]]
	#tbg_range = [[100000,120000],[150000,160000],[190000,210000]]	

##########################################

vel_stacked = [] #to hold velocity-stacked spectra
int_stacked = []

freq_obs = [] #to hold laboratory or observational spectra
int_obs = []

freq_sim = [] #to hold simulated spectra
int_sim = [] 

freq_sum = [] #to hold combined spectra
int_sum = []

freq_man = [] #to hold manual (frequency only) spectra
int_man = []

freq_resid = [] #to hold residual spectra
int_resid = []
current = catalog_file

obs_name = ''

colors = itertools.cycle(['#ff8172','#514829','#a73824','#7b3626','#a8ac87','#8c8e64','#974710','#d38e20','#ce9a3a','#ae7018','#ac5b14','#64350f','#b18f59','#404040','#791304','#1f2161','#171848','#3082fe','#2c5b5e','#390083','#5c65f8','#6346fa','#3c3176','#1cf6ba','#c9bcf0','#90edfc','#3fb8ee','#b89b33','#e7d17b'])
styles = itertools.cycle(['-','--','-.',':'])

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
	
	if qns > 6:
	
		qns = 6

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
		
	elif '13ch3oh.cat' in catalog_file.lower() or 'c033502.cat' in catalog_file.lower():
	
		Q = 0.399272*T**1.756329
	
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

	
		for i in range(ustates):
	
			J = temp[i][0] #Extract a J value from the list
			E = temp[i][qns] #Extract its corresponding energy
			
			if 'benzonitrile.cat' in catalog_file.lower() or 'bn_global.cat' in catalog_file.lower():
			
				#Goes through and does the adjustments for the nuclear hyperfine splitting (it's being overcounted in the catalog, needs to be divide by 3), and the spin-statistic degeneracies.
			
				if (temp[i][1] % 2 == 0):
				
					 Q += (1/3)*(5/8)*(2*J+1)*exp(np.float64(-E/(kcm*T)))
					 
				else:
				
					Q += (1/3)*(3/8)*(2*J+1)*exp(np.float64(-E/(kcm*T)))					
					
			else:
		
				Q += (2*J+1)*exp(np.float64(-E/(kcm*T))) #Add it to Q
			
		#result = [Q,ustates] #can enable this function to check the number of states used in the calculation, but note that this will break calls to Q further down that don't go after element 0.
	
	return Q

#scale_temp scales the simulated intensities to the new temperature

def scale_temp(int_sim,qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,CT,catalog_file):

	'''
	Converts linear intensities at one temperature to another.
	'''

	scaled_int = np.copy(int_sim)
	
	scaled_int *= 0.0
	
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

	intensity = np.copy(logint)	
	
	intensity = 10**(logint)
	
	return intensity
	
#simulates Gaussian profiles after intensities are simulated.			

def sim_gaussian(int_sim,freq,linewidth):

	'''
	Simulates Gaussian profiles for lines, after the intensities have been calculated.  Tries to be smart with how large a range it simulates over, for computational resources.  Includes a thermal cutoff for optically-thick lines.
	'''
	
	freq_gauss_tmp = []
	
	x = 0
	
	if cavity_ftmw == True:
	
		linewidth = cavity_dV

	while (x < len(int_sim)):
	
		l_f = linewidth*freq[x]/ckm #get the FWHM in MHz
	
		min_f = freq[x] - 10*l_f #get the frequency 10 FWHM lower
		
		max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
		
		if x < len(int_sim)-2:
		
			while (freq[x+1] < max_f and x < len(int_sim)-2):
		
					x += 1
			
					max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
	
		freq_line = np.arange(min_f,max_f,res) #generate a chunk of spectra at resolution res
	
		freq_gauss_tmp.extend(freq_line)
		
		x+= 1
	
	freq_gauss_tmp.sort()
	
	freq_gauss = np.asarray(freq_gauss_tmp)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0
	
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
		
		#if set up to simulate cavity FTMW doppler doublets, divide the intensity by half, shift each line up and down by the splitting, and make sure to use the correct fwhm.
		
		if cavity_ftmw == True:
		
			freq_l = freq[x] - (cavity_split*freq[x]/ckm)
			freq_h = freq[x] + (cavity_split*freq[x]/ckm)
			
			int_gauss += 0.5*int_sim[x]*exp(-((freq_gauss - freq_l)**2/(2*c**2)))
			int_gauss += 0.5*int_sim[x]*exp(-((freq_gauss - freq_h)**2/(2*c**2)))
						
		else:
		
			Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_gauss)
		
			J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*T))) -1)**-1
			J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*Tbg))) -1)**-1

			int_gauss += int_sim[x]*exp(-((freq_gauss - freq[x])**2/(2*c**2)))
	
	Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_gauss)
	
	J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*T))) -1)**-1
	J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*Tbg))) -1)**-1
	
	int_gauss_tau = (J_T - J_Tbg)*(1 - np.exp(-int_gauss))
	
	return(freq_gauss,int_gauss_tau)

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
		
	elif x == 'residual':
	
		freq_tmp = freq_resid
		int_tmp = int_resid
		
	elif x == 'obs':
	
		freq_tmp = freq_obs
		int_tmp = int_obs
		
	elif x == 'stacked':
	
		freq_tmp = vel_stacked
		int_tmp = int_stacked
		
	elif x == 'tbg':
	
		freq_tmp = freq_sim
		int_tmp = tbg
				
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

#apply_beam applies a beam dilution correction factor

def apply_beam(frequency,intensity,source_size,dish_size):

	#create a wave to hold wavelengths, fill it to start w/ frequencies

	wavelength = np.copy(frequency)
	
	#Convert those frequencies to Hz
	
	wavelength *= 1.0E6
	
	#Convert to meters
	
	wavelength = cm/wavelength
	
	#create an array to hold beam sizes
	
	beam_size = np.copy(wavelength)
	
	#fill it with beamsizes
	
	beam_size *= 206265 * 1.22 / dish_size
	
	#create an array to hold beam dilution factors
	
	dilution_factor = np.copy(beam_size)
	
	dilution_factor = source_size**2/(beam_size**2 + source_size**2)
	
	intensity_diluted = np.copy(intensity)
	
	intensity_diluted *= dilution_factor
	
	return intensity_diluted
	
#run_sim runs the simulation.  It's a meta routine, so that we can update later

def run_sim(freq,intensity,T,dV,C):

	'''
	Runs a full simulation accounting for the currently-active T, dV, S, and vlsr values, as well as any thermal cutoff for optically-thick lines
	'''
	
	np.seterr(under='ignore')
	np.seterr(over='ignore')
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file)
	
	Nl = C * (2*qn7 + 1) * np.exp(-elower/(0.695 * T)) / Q
	
	tau_numerator = np.asarray((ccm/(frequency * 10**6))**2 * aij * gup * Nl * (1 - np.exp(-(h * frequency * 10**6)/(k * T))),dtype=float)
	
	tau_denominator = np.asarray(8 * np.pi * (dV * frequency * 10**6 / ckm) * (2*qn7 + 1),dtype=float)

	tau =tau_numerator/tau_denominator
	
	int_temp = tau
		
	int_temp = trim_array(int_temp,frequency,ll,ul)		
	
	freq_tmp = trim_array(freq,frequency,ll,ul)
	
	int_temp = apply_beam(freq_tmp,int_temp,source_size,dish_size)
	
	if gauss == True:

		freq_sim,int_sim = sim_gaussian(int_temp,freq_tmp,dV)
		
	else:
	
		
		freq_sim = freq_tmp

		Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_sim)
		
		J_T = (h*freq_sim*10**6/k)*(np.exp(((h*freq_sim*10**6)/(k*T))) -1)**-1
		J_Tbg = (h*freq_sim*10**6/k)*(np.exp(((h*freq_sim*10**6)/(k*Tbg))) -1)**-1
		
		int_sim = (J_T - J_Tbg)*(1 - np.exp(-int_temp))
		
	if planck == True:
	
		#calculate the beam solid angle, and throw an error if it hasn't been set.
		
		try:
			omega = synth_beam[0]*synth_beam[1]*np.pi/(4*np.log(2))	
		except TypeError:
			print('You need to set a beam size to use for this conversion with synth_beam = [bmaj,bmin]')
			print('Your simulation is still in Kelvin.')
			return freq_sim,int_sim
		
		#create an array that's a copy of int_sim to work with temporarily
			
		int_jansky = np.copy(int_sim)
		
		#create a mask so we only work on non-zero values
		
		mask = int_jansky != 0
		
		#do the conversion
		
		int_jansky[mask] = (3.92E-8 * (freq_sim[mask]*1E-3)**3 *omega/ (np.exp(0.048*freq_sim[mask]*1E-3/int_sim[mask]) - 1))
		
		int_sim = int_jansky
		
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

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
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

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')
	
#modS is a legacy command from viewspectrum.py that just calls modC, the new command.

def modS(x):

	modC(x)

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

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
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

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
	
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
	plt.ylabel('Intensity (K)')
	plt.title(obs_name)
	plt.rcParams['axes.labelweight'] = 'bold'

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass

	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
		
	fig.canvas.draw()
	
# 	if labels_flag == True:
# 	
# 		labels_on()
	
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

	global spec, coords, GHz, res, obs_name, draw_style

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
	
	freq_tmp = list(freq_obs)
	int_tmp = list(int_obs)
	
	freq_obs = [freq_tmp for freq_tmp,int_tmp in sorted(zip(freq_tmp,int_tmp))]
	int_obs = [int_tmp for freq_tmp,int_tmp in sorted(zip(freq_tmp,int_tmp))]	
		
	if GHz == True:
	
		freq_obs[:] = [x*1000.0 for x in freq_obs]
		
	res = abs(freq_obs[1]-freq_obs[0])	
	
	if res == 0.0:
	
		res = abs(freq_obs[2]-freq_obs[1])
		
	if res == 0.0:
	
		print('First three frequency data points for the observations are identical; resolution could not be automatically determined and has been set to 10 kHz by default.  Modify this by issuing res = X, where X is the desired resolution in MHz')
	
	clear_line('obs')
		
	try:		
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0, drawstyle=draw_style)
	except:
		return
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()	
	
	if len(spec.split('.')) > 1:
		
		tmp_str = str(spec.split('.')[-1])
	
		obs_name = str(spec.strip(tmp_str).strip('.').split('/')[-1])
		
	else: 
	
		obs_name = str(spec)
		
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
	
		x = '{}' .format(catalog_file.split('.')[0].strip('\n').split('/')[-1]) 
	
	sim[x] = Molecule(x,catalog_file,tag,gup,dof,error,qn1,qn2,qn3,qn4,qn5,qn6,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim,aij,sijmu)
	
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

	global elower,eupper,qns,logint,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12,S,dV,T,vlsr,frequency,freq_sim,intensity,int_sim,current,catalog_file,sijmu,C,tag,gup,error,aij
	
	current = sim[x].name
	elower = sim[x].elower
	eupper = sim[x].eupper
	
	qns = sim[x].qns
	logint = sim[x].logint
	gup = sim[x].gup
	tag = sim[x].tag
	error = sim[x].error
	qn1 = sim[x].qn1
	qn2 = sim[x].qn2
	qn3 = sim[x].qn3
	qn4 = sim[x].qn4
	qn5 = sim[x].qn5
	qn6 = sim[x].qn6
	qn7 = sim[x].qn7
	qn8 = sim[x].qn8
	qn9 = sim[x].qn9
	qn10 = sim[x].qn10
	qn11 = sim[x].qn11
	qn12 = sim[x].qn12
	C = sim[x].C
	dV = sim[x].dV
	T = sim[x].T
	vlsr = sim[x].vlsr
	frequency = sim[x].frequency
	freq_sim = sim[x].freq_sim
	intensity = sim[x].intensity
	int_sim = sim[x].int_sim
	catalog_file = sim[x].catalog_file
	aij = sim[x].aij
	sijmu = sim[x].sijmu

	try:
		clear_line('current')
	except:
		pass
		
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file)
	
	freq_sim,int_sim=run_sim(tmp_freq,intensity,T,dV,C)	
		
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)	
		
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
	loads a new molecule into the system.  Make sure to store the old molecule simulation first, if you want to get it back.  The current graph will be updated with the new molecule.  Catalog file must be given as a string.  Simulation will begin with the same T, dV, C, vlsr as previous, so change those first if you want.
	'''

	global frequency,logint,error,dof,gup,tag,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,first_run,tbg,sijmu,gauss,aij		
	
	current = x
	
	try:
		clear_line('current')
	except:	
		pass	
	
	catalog_file = x
	
	raw_array = read_cat(catalog_file)

	catalog = splice_array(raw_array)

	frequency = np.copy(catalog[0])
	error = np.copy(catalog[1])
	logint = np.copy(catalog[2])
	dof = np.copy(catalog[3])
	elower = np.asarray(catalog[4])
	gup = np.asarray(catalog[5])
	tag = np.asarray(catalog[6])
	qnformat = np.asarray(catalog[7])		
	qn1 = np.asarray(catalog[8])
	qn2 = np.asarray(catalog[9])
	qn3 = np.asarray(catalog[10])
	qn4 = np.asarray(catalog[11])
	qn5 = np.asarray(catalog[12])
	qn6 = np.asarray(catalog[13])
	qn7 = np.asarray(catalog[14])
	qn8 = np.asarray(catalog[15])
	qn9 = np.asarray(catalog[16])
	qn10 = np.asarray(catalog[17])
	qn11 = np.asarray(catalog[18])
	qn12 = np.asarray(catalog[19])
	
	tbg = np.copy(elower)
	
	tbg.fill(2.7) #background temperature in the source, defaulting to CMB.  Can change this manually or with init_source()

	eupper = np.copy(elower)

	eupper = elower + frequency/29979.2458

	qns = det_qns(qnformat) #figure out how many qns we have for the molecule

	intensity = convert_int(logint)
	
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file)

	sijmu = (exp(np.float64(-(elower/0.695)/CT)) - exp(np.float64(-(eupper/0.695)/CT)))**(-1) * ((10**logint)/frequency) * ((4.16231*10**(-5))**(-1)) * Q
	
	#aij formula from CDMS.  Verfied it matched spalatalogue's values
	
	aij = 1.16395 * 10**(-20) * frequency**3 * sijmu / gup
	
	freq_sim,int_sim=run_sim(tmp_freq,intensity,T,dV,C)
	
	if gauss == True:
	
		gauss = False
		
		freq_stick,int_stick=run_sim(tmp_freq,intensity,T,dV,C)
		
		gauss = True
		
	else:
	
		freq_stick = np.asarray([freq_sim])
		int_stick = np.asarray([int_sim])
		

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

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)	

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
		
		output.write('Molecule\tT(K)\tC\tdV\tvlsr\tCT\tcatalog_file\n')
	
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
	print('C: \t {}' .format(C))
	print('dV: \t {} km/s' .format(dV))
	print('VLSR: \t {} km/s' .format(vlsr))
	print('ll: \t {} MHz' .format(ll))
	print('ul: \t {} MHz' .format(ul))
	print('CT: \t {} K' .format(CT))
	print('gauss: \t {}' .format(gauss))
	print('thermal: \t {}' .format(thermal))
		
#sum_stored creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.  It's done from scratch because the frequency axes in freq_sim stored for each molecule will not necessarily be the same, so co-adding is difficult without re-gridding, which well, maybe later.	

def sum_stored():

	'''
	Creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.
	'''
	global freq_sum, int_sum
	
	freq_gauss_tmp = []
	
	freq_sim_total = []

	for x in sim:
	
		tmp_freq = np.copy(sim[x].frequency)
		
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
		
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
		
		freq_sim_total.extend(tmp_freq_trimmed)

	freq_sim_total.sort()
		
	y = 0
		
	while (y < len(freq_sim_total)):
	
		l_f = sim[x].dV*freq_sim_total[y]/ckm #get the FWHM in MHz

		min_f = freq_sim_total[y] - 10*l_f #get the frequency 10 FWHM lower
		max_f = freq_sim_total[y] + 10*l_f #get the frequency 10 FWHM higher
	
		if y < len(freq_sim_total)-2:
	
			while (freq_sim_total[y+1] < max_f and y < len(freq_sim_total)-2):
	
				y += 1

		freq_line = np.arange(min_f,max_f,res)

		freq_gauss_tmp.extend(freq_line)
		
		y += 1
	
	freq_gauss_tmp.sort()
	
	freq_gauss = np.asarray(freq_gauss_tmp)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0
	
	del tmp_freq, tmp_freq_trimmed
		
	for x in sim:
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,sim[x].T,sim[x].catalog_file)
	
		Nl = sim[x].C * (2*sim[x].qn7 + 1) * np.exp(-sim[x].elower/(0.695 * sim[x].T)) / Q
	
		tau_numerator = np.asarray((ccm/(sim[x].frequency * 10**6))**2 * sim[x].aij * sim[x].gup * Nl * (1 - np.exp(-(h * sim[x].frequency * 10**6)/(k * sim[x].T))),dtype=float)
	
		tau_denominator = np.asarray(8 * np.pi * (sim[x].dV * sim[x].frequency * 10**6 / ckm) * (2*sim[x].qn7 + 1),dtype=float)

		tau =tau_numerator/tau_denominator
		
		int_tmp = trim_array(tau,sim[x].frequency,ll,ul)		
	
		freq_tmp = trim_array(sim[x].frequency,sim[x].frequency,ll,ul)
		
		freq_tmp += (-sim[x].vlsr)*freq_tmp/ckm	
	
		#tmp_freq_trimmed,tmp_int_trimmed = sim_gaussian(int_tmp,freq_tmp,sim[x].dV)
	
		for y in range(len(int_tmp)):
		
			l_f = sim[x].dV*freq_tmp[y]/ckm #get the FWHM in MHz
			
			c = l_f/2.35482
			
			int_tmp_tau = (sim[x].T - Tbg)*(1 - np.exp(-int_tmp[y]))
			
			int_gauss += int_tmp_tau*exp(-((freq_gauss - freq_tmp[y])**2/(2*c**2)))
		
		#for y in range(len(tmp_int_trimmed)):
		
		#	if abs(tmp_int_trimmed[y]) < 0.001:
			
		#		continue
		
		#	l_f = sim[x].dV*tmp_freq_trimmed[y]/ckm #get the FWHM in MHz
			
		#	c = l_f/2.35482
			
		#	int_tmp_tau = (sim[x].T - Tbg)*(1 - np.exp(-tmp_int_trimmed[y]))

		#	int_gauss += int_tmp_tau*exp(-((freq_gauss - tmp_freq_trimmed[y])**2/(2*c**2)))

			#int_gauss += tmp_int_trimmed[y]*exp(-((freq_gauss - tmp_freq_trimmed[y])**2/(2*c**2)))
			
	#int_gauss_tau = (T - Tbg)*(1 - np.exp(-int_gauss))
	
	int_gauss_tau = int_gauss
	
	int_gauss[int_gauss > (sim[x].T - Tbg)] = (sim[x].T - Tbg)
	
	freq_sum = freq_gauss
	int_sum = int_gauss_tau		
	
	overplot_sum()

#overplot_sum overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.

def overplot_sum():

	'''
	Overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.
	'''	
	
	line_color = '#00FF00'
	
	if any(line for line in ax.lines if line.get_label()=='sum'):
		clear_line('sum')
	
	lines['sum'] = ax.plot(freq_sum,int_sum,color = line_color, label = 'sum', gid='sum', linestyle = '-',drawstyle=draw_style,zorder=25)
	
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

	global frequency,logint,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,T,dV,C,vlsr,ll,ul,CT,gauss,first_run,thermal,sim,GHz,rms,res
	
	#close the old graph
	
# 	make_plot()
# 	close()
	
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
	
		print('Warning: {} was created using v{}.  The current version is v{}.  Type help(restore) for more information.' .format(x,restore_version,str(version)))   
		
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
	
	try:
		obs = active_array[1].split('\t')[1].strip('\n')
		read_obs(obs)
	except:
		res = 0.1
		pass
	
	#OK, now time to do the hard part.  As one always should, let's start with the middle part of the whole file, and load and then store all of the simulations.
	
	for i in range(len(stored_array)):
	
		name = stored_array[i].split('\t')[0].strip('\n')
		T = float(stored_array[i].split('\t')[1])
		C = float(stored_array[i].split('\t')[2])
		dV = float(stored_array[i].split('\t')[3])
		vlsr = float(stored_array[i].split('\t')[4])
		CT = float(stored_array[i].split('\t')[5])
		catalog_file = str(stored_array[i].split('\t')[6]).strip('\n').strip()
		

		first_run = True
		load_mol(catalog_file)

# 		try:	
# 			load_mol(catalog_file)
# 		except FileNotFoundError:
# 			continue
			
		store(name)
		
		close()
			
	#Now we move on to loading in the currently-active molecule
	
	try:
		obs = active_array[1].split('\t')[1].strip('\n')
		read_obs(obs)
	except:
		pass
	name = active_array[0].split('\t')[1].strip('\n')

	try:
		recall(name)
	except KeyError:
		catalog_file = active_array[10].split('\t')[1].strip('\n')
		T = float(active_array[2].split('\t')[1].strip(' K\n'))
		C = float(active_array[3].split('\t')[1].strip('\n'))
		dV = float(active_array[4].split('\t')[1].strip(' km/s\n'))
		vlsr = float(active_array[5].split('\t')[1].strip(' km/s\n'))
		CT = float(active_array[8].split('\t')[1].strip(' K\n'))
		current = active_array[0].split('\t')[1].strip('\n')
		name = active_array[0].split('\t')[1]
		
		first_run = True
		load_mol(catalog_file)

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

#quiet suppresses warnings about computational time.  Can be used iteratively to turn it on and off.

def quiet():

	global quietflag

	if quietflag == False:
	
		quietflag = True
		
	elif quietflag == True:
	
		quietflag = False
		
#autoset_limits() automatically sets the upper and lower limits to 25 MHz above and below the lowest limits of the loaded spectra.

def autoset_limits(spacing_tolerance=100):

	global ll,ul
	
	if len(freq_obs) == 0:
	
		print('First, load a spectrum with read_obs()')
		
	#run through the data and try to find gaps
	#first, calculate the average spacing of the datapoints
	
	spacing = abs(freq_obs[0]-freq_obs[10])/10
	
	#find all values where the next point is more than spacing_tolerance * spacing away
	
	#initialize a list to hold these, with the first point being in there automatically
	
	values = [freq_obs[0]]
	
	for x in range(len(freq_obs)-1):
	
		if abs(freq_obs[x+1] - freq_obs[x]) > spacing_tolerance*spacing:
		
			values.append(freq_obs[x])
			values.append(freq_obs[x+1])
			
	#make sure the last point gets in there too
	
	values.append(freq_obs[-1])
			
	ll = values[0::2] 
	ul = values[1::2]

	ll = [x-25 for x in ll]
	ul = [x+25 for x in ul]

#plot_residuals will make a new plot and show the residual spectrum after the total simulation is subtracted from the observations

def plot_residuals():

	#first, make a simulation that has exactly the same frequency points as the observations
	
	global freq_resid,int_resid
	
	freq_gauss = np.asarray(freq_obs)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0	
	
	for x in sim:

		tmp_freq = np.copy(sim[x].frequency)
	
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
	
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,CT,sim[x].catalog_file)

		sijmu = (exp(np.float64(-(sim[x].elower/0.695)/CT)) - exp(np.float64(-(sim[x].eupper/0.695)/CT)))**(-1) * ((10**sim[x].logint)/sim[x].frequency) * ((4.16231*10**(-5))**(-1)) * Q
	
		tmp_int = np.copy(sim[x].intensity)
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,sim[x].T,sim[x].catalog_file)
	
		numerator = (sim[x].C)*(8*3.14159**3)*(tmp_freq*1E6)*(sijmu)*(1-((exp((h*tmp_freq*1E6)/(k*sim[x].T))-1)/(exp((h*tmp_freq*1E6)/(k*Tbg))-1)))*eta

		denominator = 1.06447*sim[x].dV*Q*(exp(sim[x].eupper/(kcm*sim[x].T)))*(3*k)*1E48

		tmp_int = numerator/denominator
	
		tmp_int_trimmed = trim_array(tmp_int,tmp_freq,ll,ul)
	
		tmp_int_trimmed[tmp_int_trimmed > thermal] = thermal

		for y in range(len(tmp_int_trimmed)):
	
			if abs(tmp_int_trimmed[y]) < 0.001:
		
				continue
	
			l_f = sim[x].dV*tmp_freq_trimmed[y]/ckm #get the FWHM in MHz
		
			c = l_f/2.35482

			int_gauss += tmp_int_trimmed[y]*exp(-((freq_gauss - tmp_freq_trimmed[y])**2/(2*c**2)))
			
	int_gauss[int_gauss > thermal] = thermal
	
	freq_sum = freq_gauss
	int_sum = int_gauss		
	
	freq_resid = np.asarray(freq_sum)
	int_resid = np.asarray(int_sum)
	
	int_obs_tmp = np.asarray(int_obs)
	
	int_resid = int_obs_tmp - int_sum
	
	f, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=False)
	
	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Frequency (MHz)')
	ax1.set_ylabel('Intensity (K)')
	ax2.set_ylabel('Intensity (K)')
	
	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax1.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax1.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax1.get_xaxis().get_major_formatter().set_useOffset(False)
		
	ax2.plot(freq_obs,int_obs,color = 'black',label='obs')
	ax2.plot(freq_sum,int_sum,color = 'green',label='simulation')
	ax1.set_ylim(ax2.get_ylim())
	ax1.plot(freq_resid,int_resid,color = 'red',label='residual')
	
	ax2.legend()
	ax1.legend()
	
	fig.canvas.draw()

#print_lines will print out the catalog info on lines that are above a certain threshold for a molecule.  The default just prints out the current lines above a standard 1 mK threshold.		

def print_lines(mol='current',thresh=0.0001,rest=True):

	global gauss
	
	gflag = False

	if gauss == True:
	
		gauss = False
		
		gflag = True

	#prep an array to hold the output string for each line
	
	print_array = []

	#If we're working with the current simulation, re-run it to get the non-Gaussian lines
	
	if mol == 'current':
	
		name = catalog_file.split('.')[0].strip('\n').split('/')[-1]
		
		qns_length = 4*qns*2 + 15 #length of the qn string
		
		if qns_length > 15:
		
			qn_length = qns_length	#if the length of the string is going to be longer than the label, we need to pad the label.
	
		print_array.append('Molecule: {}' .format(name))
		print_array.append('Column Density: {:.2e} cm-2 \t Temperature: {} K \t Linewidth: {} km/s \t vlsr: {} km/s' .format(C,T,dV,vlsr))
		print_array.append('Frequency \t Intensity (K) \t {{:<{}}} \t Eu (K) \t gJ \t log(Aij)' .format(qn_length).format('Quantum Numbers'))
	
		#gotta re-read-in the molecule to get back the quantum numbers
		
		raw_array = read_cat(catalog_file)

		catalog = splice_array(raw_array)
		
		qn1 = np.asarray(catalog[8])
		qn2 = np.asarray(catalog[9])
		qn3 = np.asarray(catalog[10])
		qn4 = np.asarray(catalog[11])
		qn5 = np.asarray(catalog[12])
		qn6 = np.asarray(catalog[13])
		qn7 = np.asarray(catalog[14])
		qn8 = np.asarray(catalog[15])
		qn9 = np.asarray(catalog[16])
		qn10 = np.asarray(catalog[17])
		qn11 = np.asarray(catalog[18])
		qn12 = np.asarray(catalog[19])
			
		#run the simulation to get the intensities
	
		freq_tmp,int_tmp = run_sim(frequency,intensity,T,dV,C)
		
		old_f = np.nan
		
		i = 0

		for x in range(len(freq_tmp)):
		
			if int_tmp[x] > thresh:
		
				y = np.where(frequency == freq_tmp[x])[0]
		
				qn_string = ''
		
				#deal with the case where multiple transitions have the same frequency		
				
				if freq_tmp[x] == old_f:
				
					i += 1
					
				else:
				
					i = 0 

				if qns == 1:
   
					qn_string = '{:>2} -> {:>2}' .format(qn1[y][i],qn7[y][i])
	   
				if qns == 2:
   
					qn_string = '{:>2} {: >3} -> {:>2} {: >3}' .format(qn1[y][i],qn2[y][i],qn7[y][i],qn8[y][i])		
	   
				if qns == 3:
   
					qn_string = '{:>2} {: >3} {: >3} -> {:>2} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn7[y][i],qn8[y][i],qn9[y][i])							

				if qns == 4:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i])	
	   
				if qns == 5:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i])			
	   
				if qns == 6:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn6[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i],qn12[y][i])							
	
				gJ = 2*qn1[y][i] + 1
			
				if rest==False:
			
					frequency_tmp_shift = frequency[y][i] - vlsr*frequency[y][i]/3E5				
	
					print_array.append('{:} \t {:<13.3f} \t {} \t {:<9.3f} \t {} \t {:.2f}' .format(frequency_tmp_shift,int_tmp[x],qn_string,eupper[y][i]/0.695,gJ,np.log10(aij[y][i])))
				
				else:
			
					print_array.append('{:} \t {:<13.3f} \t {} \t {:<9.3f} \t {} \t {:.2f}' .format(frequency[y][i],int_tmp[x],qn_string,eupper[y][i]/0.695,gJ,np.log10(aij[y][i])))
					
				old_f = freq_tmp[x]	
						
	else:
	
		try:
			catalog_file_tmp = sim[mol].catalog_file
		except KeyError:
			print('No molecule stored with that name.')
			
		C_tmp = sim[mol].C
		T_tmp = sim[mol].T
		dV_tmp = sim[mol].dV
		vlsr_tmp = sim[mol].vlsr
		frequency_tmp = sim[mol].frequency
		intensity_tmp = sim[mol].intensity
		eupper_tmp = sim[mol].eupper
		aij_tmp = sim[mol].aij
		
		qns_tmp = sim[mol].qns
		
		qns_length = 4*(qns_tmp-2)*2 + 15 #length of the qn string
		
		if qns_length > 15:
		
			qn_length = qns_length	#if the length of the string is going to be longer than the label, we need to pad the label.
			
		print_array.append('Molecule: {}' .format(mol))
		print_array.append('Column Density: {:.2e} cm-2 \t Temperature: {} K \t Linewidth: {} km/s \t vlsr: {} km/s' .format(C_tmp,T_tmp,dV_tmp,vlsr_tmp))
		print_array.append('Frequency \t Intensity (K) \t {{:<{}}} \t Eu (K) \t gJ' .format(qn_length).format('Quantum Numbers'))
		
	
		#gotta re-read-in the molecule to get back the quantum numbers
		
		raw_array = read_cat(catalog_file_tmp)

		catalog = splice_array(raw_array)
		
		qn1 = np.asarray(catalog[8])
		qn2 = np.asarray(catalog[9])
		qn3 = np.asarray(catalog[10])
		qn4 = np.asarray(catalog[11])
		qn5 = np.asarray(catalog[12])
		qn6 = np.asarray(catalog[13])
		qn7 = np.asarray(catalog[14])
		qn8 = np.asarray(catalog[15])
		qn9 = np.asarray(catalog[16])
		qn10 = np.asarray(catalog[17])
		qn11 = np.asarray(catalog[18])
		qn12 = np.asarray(catalog[19])
		
		#run the simulation to get the intensities
	
		freq_tmp,int_tmp = run_sim(frequency_tmp,intensity_tmp,T_tmp,dV_tmp,C_tmp)
		
		old_f = np.nan
		
		i = 0
		
		for x in range(len(freq_tmp)):
		
			if int_tmp[x] > thresh:
		
				y = np.where(frequency_tmp == freq_tmp[x])
			
				qn_string = ''
				
				#deal with the case where multiple transitions have the same frequency		
				
				if freq_tmp[x] == old_f:
				
					i += 1
					
				else:
				
					i = 0 				
			
				if qns_tmp == 1:
			
					qn_string = '{:>2} -> {:>2}' .format(qn1[y][i],qn7[y][i])
				
				if qns_tmp == 2:
			
					qn_string = '{:>2} {: >3} -> {:>2} {: >3}' .format(qn1[y][i],qn2[y][i],qn7[y][i],qn8[y][i])		
				
				if qns_tmp == 3:
			
					qn_string = '{:>2} {: >3} {: >3} -> {:>2} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn7[y][i],qn8[y][i],qn9[y][i])							

				if qns_tmp == 4:
			
					qn_string = '{:>2} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i])	
				
				if qns_tmp == 5:
			
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i])			
				
				if qns_tmp == 6:
			
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn6[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i],qn12[y][i])							
		
				gJ = 2*qn1[y][i] + 1
				
				if rest==False:
				
					frequency_tmp_shift = frequency_tmp[y][i] - vlsr_tmp*frequency_tmp[y][i]/3E5
			
					print_array.append('{:} \t {:<13.3f} \t {} \t {:<9.3f} \t {} \t {:.2f}' .format(frequency_tmp_shift,int_tmp[x],qn_string,eupper_tmp[y][i]/0.695,gJ,np.log10(aij_tmp[y][i])))
				
				else:
				
					print_array.append('{:} \t {:<13.3f} \t {} \t {:<9.3f} \t {} \t {:.2f}' .format(frequency_tmp[y][i],int_tmp[x],qn_string,eupper_tmp[y][i]/0.695,gJ,np.log10(aij_tmp[y][i])))
					
				old_f = freq_tmp[x]		
			
	for x in range(len(print_array)):
	
		print('{}' .format(print_array[x]))		
		
	if gflag == True:
		
		gauss = True	
		
#gauss_func is a model gaussian function to be used with gauss_fit below

def gauss_func(x, dT, v, dV):

	#get the FWHM in frequency space here
		
	df = dV*v/ckm
	
	#convert to c
	
	c = df/2.35482
	
	#return the Gaussian value
	
	G = dT*np.exp(- ((x-v)**2/(2*c**2)))
	
	return G

#gauss_fit does a Gaussian fit on lines in the data, specified in tuples: p = [[dT1,v1,dV1],[dT2,v2,dV2],...] where dT1,v1,dV1 etc are the initial guesses for the intensity, line center, and fwhm of the lines.  dT is in whatever units are being used in the observations, v is in whatever units are being used in the observations, and dV is in km/s.  By default, the amplitude is unconstrained, the center frequency is constrained to within 5 MHz of the guess, and the linewidth is constrained to within 20% of the guess.  These can be changed.

def gauss_fit(p,plot=True,dT_bound=np.inf,v_bound=5.0,dV_bound=0.2):

	'''
	#gauss_fit does a Gaussian fit on lines in the data, specified in tuples: p = [[dT1,v1,dV1],[dT2,v2,dV2],...] where dT1,v1,dV1 etc are the initial guesses for the intensity, line center, and fwhm of the lines.  dT is in whatever units are being used in the observations, v is in whatever units are being used in the observations, and dV is in km/s.  By default, the amplitude is unconstrained, the center frequency is constrained to within 5 MHz of the guess, and the linewidth is constrained to within 20% of the guess.  These can be changed.
	'''

	data = [freq_obs,int_obs]
	
	coeff = []
	var_matrix = []	
	err_matrix = []
	fit = np.copy(data[0])
	fit *= 0.0
		
	for x in range(len(p)):
	
		temp = curve_fit(gauss_func, data[0], data[1], p0 = p[x], bounds=([-dT_bound,p[x][1]-v_bound,p[x][2]*(1-dV_bound)],[dT_bound,p[x][1]+v_bound,p[x][2]*(1+dV_bound)]))
		
		coeff.append(temp[0])
		var_matrix.append(temp[1])
		err_matrix.append(np.sqrt(np.diag(temp[1])))
		
		fit += gauss_func(data[0], coeff[x][0], coeff[x][1], coeff[x][2])

	if plot == True:
	
		try:
			plt.get_fignums()[0]
		except:	
			make_plot()
					
		clear_line('Gauss Fit')
	
		lines['Gauss Fit'] = ax.plot(data[0],fit,color='cyan',label='Gauss Fit',linestyle= '-', gid='gfit', zorder = 50000)
 		
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend()
		fig.canvas.draw()		
	
	print('Gaussian Fit Results to {} Lines' .format(len(p)))
	print('{:<20} \t {:<10} \t {:<10}' .format('Line Center','dT', 'dV'))
		
	for x in range(len(p)):
	
		dT_temp = coeff[x][0]
		v_temp = coeff[x][1]
		dV_temp = coeff[x][2]
		
		dT_err = err_matrix[x][0]
		v_err = err_matrix[x][1]
		dV_err = err_matrix[x][2]
		
		print('{:<.4f}({:<.4f}) \t {:^.3f}({:^.3f}) \t {:^.3f}({:^.3f})' .format(v_temp,v_err,dT_temp,dT_err,dV_temp,dV_err))
		
#make_gauss_params generates a parameters list for gaussian fitting.  Takes an input txt file which is first column frequencies and second column intensities.  

def make_gauss_params(file,vlsr,dV):

	p = []

	with open(file) as input:

		for line in input:
	
			p_indiv = []
	
			freq = float(line.split()[0])
			freq -= vlsr*freq/3E5
			
			p_indiv.append(float(line.split()[1].strip('\n')))
			p_indiv.append(freq)			
			p_indiv.append(float(dV))
		
			p.append(p_indiv)
		
	return p

#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out. 

def jy_to_k(bmaj,bmin,freq):

	'''
	#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out
	'''

	global freq_obs,int_obs
	
	for x in range(len(int_obs)):

		int_obs[x] = 1.224*10**6 * int_obs[x] / (freq**2 * bmaj * bmin)		
		
	clear_line('obs')
		
	try:		
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		return
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
#k_to_jy converts the current observations from K to Jy/beam, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out. 

def k_to_jy(bmaj,bmin,freq,sim=False):

	'''
	#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out
	'''
	
	if sim == False:

		global freq_obs,int_obs
	
		for x in range(len(int_obs)):

			int_obs[x] = int_obs[x] * (freq**2 * bmaj * bmin) / (1.224*10**6)	
		
		clear_line('obs')
		
		try:		
			lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
		except:
			return
			
	else:
	
		global freq_sim,int_sim
	
		for x in range(len(int_sim)):

			int_sim[x] = int_sim[x] * (freq**2 * bmaj * bmin) / (1.224*10**6)	
		
		clear_line('current')
		
		try:		
			lines['current'] = 	ax.plot(freq_sim,int_sim,color = 'black',label='obs',zorder=0)
		except:
			return		
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()	

#load_freqs will plot lines that are provided not from a standard spcat catalog, but rather just a set of frequencies.  The user can specify either a manual array OR a catalog file containing a single column of frequencies (not both), as well as an optional intensity for the lines.

def load_freqs(man_freqs='',peak=1.0,vlsr=vlsr,dV=dV):

	'''
	#load_freqs will plot lines that are provided not from a standard spcat catalog, but rather just a set of frequencies.  The user can specify either a manual array OR a catalog file containing a single column of frequencies.  Optional commands are an intensity for the lines (defaults to 1.0), a vlsr offset (defaults to current vlsr), and a linewidth (defaults to current linewidth).  Modifying any of these parameters requires re-issuing the entire command.
	'''	

	global int_man,freq_man
	
	try:
		clear_line('manual spectra')
	except:	
		pass	
	
	int_tmp = []
	freq_tmp = []

	if type(man_freqs) == list:
	
		for x in range(len(man_freqs)):
		
			freq_tmp.append(man_freqs[x])
			int_tmp.append(peak)
			
	else:
	
		tmp_array = read_cat(man_freqs)
		
		for x in range(len(tmp_array)):
		
			freq_tmp.append(float(tmp_array[x].split()[0].strip()))
			int_tmp.append(peak)	
			
	freq_tmp = np.asarray(freq_tmp)
	int_tmp = np.asarray(int_tmp)
	
	freq_tmp -= vlsr*freq_tmp/ckm			
			
	if gauss == True:

		freq_man,int_man = sim_gaussian(int_tmp,freq_tmp,dV)
		
	else:

		freq_man = freq_tmp
		int_man = int_tmp		
		
	if gauss == False:

		lines['manual spectra'] = ax.vlines(freq_man,0,int_man,linestyle = '-',color = 'cyan',label='manual spectra',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['manual spectra'] = ax.plot(freq_man,int_man,color = 'cyan',label='manual spectra',drawstyle=draw_style,zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()						

#use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.

def use_steps():

	'''
	use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.
	'''

	global draw_style
	
	draw_style = 'steps'
	
	clear_line('obs')
		
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()	
	
def use_lines():

	'''
	use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.
	'''

	global draw_style
	
	draw_style = 'default'
	
	clear_line('obs')
	
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass	
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()		

#baseline will subtract a polynomial baseline from the spectrum of whatever order is entered as an array.  So, for example, to subtract zeroth order baseline with a static offset of 2.5, issue baseline(2.5) or baseline([2.5]).  To subtract a line of y = mx + b, issue baseline([b,m]).  To go for a larger polynomial, y = ax^3 + bx^2 + cx + d, issue baseline([d,c,b,a]).

def baseline(constants):

	'''
	baseline will subtract a polynomial baseline from the spectrum of whatever order is entered as an array.  So, for example, to subtract zeroth order baseline with a static offset of 2.5, issue baseline(2.5) or baseline([2.5]).  To subtract a line of y = mx + b, issue baseline([b,m]).  To go for a larger polynomial, y = ax^3 + bx^2 + cx + d, issue baseline([d,c,b,a]).
	'''

	global int_obs
	
	if type(constants) == int or type(constants) == float:
	
		constants = [constants]
	
	int_base = [0] * len(int_obs)
	
	for x in range(len(constants)):
	
		for y in range(len(freq_obs)):
	
			int_base[y] += constants[x]*freq_obs[y]**x
			
	for x in range(len(int_obs)):
	
		int_obs[x] -= int_base[x]
		
	clear_line('obs')
	
	try:
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()				
	
# find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks as well as the measured RMS	
	
def find_peaks(frequency,intensity,fwhm,sigma=3,width_tweak=1.0):
		
		'''
		find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks and the rms
		'''
		
		#figure out how many channels a typical line will span
		
		#calculate fwhm in MHz
		
		fwhm_MHz = fwhm*np.median(frequency)/ckm
		
		#calculate channel spacing in MHz
		
		dMHz_chan = abs(frequency[-1] - frequency[0])/len(frequency)
		 			
		#calculate the number of channels per FHWM
		
		fwhm_chan = fwhm_MHz/dMHz_chan	
		
		#define the number of channels per line, modified by width_tweak:
		
		line_chan = int(fwhm_chan * 5 * width_tweak)
				
		#So this is going to be a multistep iterative process to find peaks, remove them temporarily, and then find the RMS, and repeat until the RMS isn't changing.  Then we find all the peaks above the threshold sigma level.
		
		intensity_tmp = np.asarray(intensity)
		intensity_mask = np.asarray(intensity)
		frequency_mask = np.asarray(frequency)
		
		converged = False
		
		rms = np.inf
		
		while converged == False:
		
			rms = np.sqrt(np.nanmean(intensity_tmp**2))
					
			peak_indices_tmp = peakutils.indexes(intensity_tmp,thres=0.99)
			
			if len(peak_indices_tmp) == 0:
			
				converged = True 
				
			if len(peak_indices_tmp) < 0.75*len(intensity):
			
				converged = True
				
			for x in range(len(peak_indices_tmp)):
			
				if intensity_tmp[peak_indices_tmp[x]] < 2*rms:
				
					converged = True
			
			#now we remove the channels that have those lines in them
			
			#first, figure out the channels to drop
			
			drops = []
			
			for x in range(len(peak_indices_tmp)):
			
					ll = peak_indices_tmp[x]-line_chan
					
					if ll < 0:
					
						ll = 0
						
					ul = peak_indices_tmp[x]+line_chan+1
					
					if ul > len(intensity_tmp):
						
						ul = len(intensity_tmp)
			
					for z in range(ll,ul):
					
						drops.append(z)
						
			#use those to make a mask			
			
			mask = np.ones(len(intensity_tmp), dtype=bool)
			
			mask[drops] = False			
			
			#mask out those peaks
			
			intensity_tmp = intensity_tmp[mask]
			
			intensity_mask = intensity_mask[mask]
			
			frequency_mask = frequency_mask[mask]
			
			new_rms = np.sqrt(np.nanmean(intensity_tmp**2))
			
			#if it is the first run, set the rms to the new rms and continue
						
			if rms == np.inf:
			
				rms = new_rms
			
			#otherwise if there are no lines left above x sigma, stop
			
			elif np.amax(intensity_tmp) < new_rms*5:
			
				rms = new_rms
			
				converged = True
				
			else:
			
				rms = new_rms
				
		#now that we know the actual rms, we can find all the peaks above the threshold, and we have to calculate a real threshold, as it is a normalized intensity (percentage).  
		
		intensity_tmp = np.asarray(intensity)
		
		max = np.amax(intensity_tmp)
		min = np.amin(intensity_tmp)
		
		#calc the percentage threshold that the rms sits at
		
		rms_thres = (sigma*rms - min)/(max - min)

		peak_indices = peakutils.indexes(intensity_tmp,thres=rms_thres,min_dist=int(fwhm_chan*.5))
		
		return peak_indices,rms,frequency_mask,intensity_mask		

# find peaks in velocity space spectra

def find_vel_peaks(velocity,intensity,fwhm,sigma=3,width_tweak=1.0):
		
		'''
		find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks and the rms
		'''
		
		#figure out how many channels a typical line will span
		
		line_chan = int(dV * 5 * width_tweak)
				
		#So this is going to be a multistep iterative process to find peaks, remove them temporarily, and then find the RMS, and repeat until the RMS isn't changing.  Then we find all the peaks above the threshold sigma level.
		
		intensity_tmp = np.asarray(intensity)
		intensity_mask = np.asarray(intensity)
				
		converged = False
		
		rms = np.inf
		
		while converged == False:
					
			peak_indices_tmp = peakutils.indexes(intensity_tmp,thres=0.99)
			
			if len(peak_indices_tmp) == 0:
			
				converged = True 
				
			if len(peak_indices_tmp) < 0.5*len(intensity):
			
				converged = True
			
			#now we remove the channels that have those lines in them
			
			#first, figure out the channels to drop
			
			drops = []
			
			for x in range(len(peak_indices_tmp)):
			
					ll = peak_indices_tmp[x]-line_chan
					
					if ll < 0:
					
						ll = 0
						
					ul = peak_indices_tmp[x]+line_chan+1
					
					if ul > len(intensity_tmp):
						
						ul = len(intensity_tmp)
			
					for z in range(ll,ul):
					
						drops.append(z)
						
			#use those to make a mask			
			
			mask = np.ones(len(intensity_tmp), dtype=bool)
			
			mask[drops] = False			
			
			#mask out those peaks
			
			intensity_tmp = intensity_tmp[mask]
			
			intensity_mask = intensity_mask[mask]
			
			new_rms = np.sqrt(np.nanmean(intensity_tmp**2))
			
			#if it is the first run, set the rms to the new rms and continue
						
			if rms == np.inf:
			
				rms = new_rms
			
			#otherwise if there are no lines left above x sigma, stop
			
			elif np.amax(intensity_tmp) < new_rms*5:
			
				rms = new_rms
			
				converged = True
				
			else:
			
				rms = new_rms
				
		#now that we know the actual rms, we can find all the peaks above the threshold, and we have to calculate a real threshold, as it is a normalized intensity (percentage).  
		
		intensity_tmp = np.asarray(intensity)
		
		max = np.amax(intensity_tmp)
		min = np.amin(intensity_tmp)
		
		#calc the percentage threshold that the rms sits at
		
		rms_thres = (sigma*rms - min)/(max - min)

		peak_indices = peakutils.indexes(intensity_tmp,thres=rms_thres,min_dist=int(fwhm*.5))
		
		return peak_indices,rms	
		
# find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks
	
def find_sim_peaks(frequency,intensity,fwhm):
		
		'''
		find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks and the rms
		'''
		
		#figure out how many channels a typical line will span
		
		#calculate fwhm in MHz
		
		fwhm_MHz = fwhm*np.median(frequency)/ckm
		
		#calculate channel spacing in MHz
		
		dMHz_chan = abs(frequency[3] - frequency[2])
		
		if dMHz_chan < 0.000000000001:
		
			dMHz_chan = abs(frequency[9] - frequency[8])
			
		if dMHz_chan < 0.000000000001:
		
			dMHz_chan = abs((freq_obs[0] - freq_obs[10])/10)			
			
		if dMHz_chan < 0.000000000001:
		
			print('The program determined the channel spacing was {} MHz.  Oops.  Please take a look and make sure your frequency input is correct.' .format(dMHz_chan))
			
		#calculate the number of channels per FHWM
		
		fwhm_chan = fwhm_MHz/dMHz_chan	
		
		#define the number of channels per line, modified by width_tweak:
		
		line_chan = int(fwhm_chan * 5)
				
		intensity_tmp = np.asarray(intensity)

		peak_indices = peakutils.indexes(intensity_tmp,0,min_dist=int(fwhm_chan*.5))
		
		return peak_indices		
		
# plot_peaks will plot the peaks, as well as the determined rms level and the baseline mask, optionally

def plot_peaks(frequency,intensity,peak_indices,rms,freq_mask=None,int_mask=None):

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
	
	ax.plot(frequency,intensity,color='black',label='obs',zorder=0)
	
	rms_x = np.copy(frequency)
	rms_y = np.copy(frequency)
	rms_y.fill(rms)
	
	ax.plot(rms_x,rms_y,color='red',label='rms',zorder=50)
	
	if freq_mask is None:
		pass
	else:
		ax.plot(freq_mask,int_mask,color='green',label='baseline',zorder=15)
	
	freq_peaks = []
	int_peaks = []
	
	for x in range(len(peak_indices)):
	
		freq_peaks.append(frequency[peak_indices[x]])
		int_peaks.append(intensity[peak_indices[x]])
	
	ax.scatter(freq_peaks,int_peaks,color='red',marker='x',label='peaks',zorder=25)
	
	ax.legend()
	fig.canvas.draw()		

def find_nearest(array,value):		

	idx = (np.abs(array-value)).argmin()
	
	return idx
	
#velocity_stack does a velocity stacking analysis using the current ll and ul, and the current simulation, using lines that are at least 0.1 sigma or larger, with the brightest simulated line scaled to be 1 sigma.

def velocity_stack(man_drops=[],plot_chunks=True):

	drops = []

	#find the simulation indices where the peaks are

	peak_indices = find_sim_peaks(freq_sim,int_sim,dV)
	
	#find the frequencies and intensities corresponding to those peaks
	
	peak_freqs = freq_sim[peak_indices]
	peak_ints = int_sim[peak_indices]
	
	#go find all the chunks in obs and extract them
	
	obs_chunks = []
	
	for x in range(len(peak_freqs)):
	
		#get the peak frequency
	
		freq = peak_freqs[x]
		
		#find the index in the observation closest to that frequency
		
		idx = find_nearest(freq_obs,freq)
				
		#calculate the resolution of the data there
		
		try:
			chan_MHz = abs((freq_obs[idx] - freq_obs[idx+10])/10)
		except IndexError:
			obs_chunks.append([[],[]])
			continue
		
		#convert that to km/s
		
		chan_kms = chan_MHz*ckm/freq
		
		#calculate how many channels to go 40 FHWM away
		
		nchan = int(40*dV/chan_kms)
		
		l_idx = idx - nchan
		u_idx = idx + nchan
		
		#slice chunks out of freq_obs and int_obs and place them into obs_chunks (as numpy arrays)
		
		obs_chunks.append([np.asarray(freq_obs[l_idx:u_idx]),np.asarray(int_obs[l_idx:u_idx])])
		
	#remove anything where we didn't have data.  Hard coded to need to be within 3 MHz
	
	for x in range(len(obs_chunks)):
		
		if len(obs_chunks[x][0]) == 0:
		
			drops.append(x)	
			
			continue
			
		mid_point = math.ceil(len(obs_chunks[x][0])/2)	
			
		if abs(obs_chunks[x][0][mid_point] - peak_freqs[x]) >  3:
		
			drops.append(x)
			
			continue
			
		total_span = obs_chunks[x][0][-1] - obs_chunks[x][0][0]
		
		freq = peak_freqs[x]
		
		dV_MHz = dV*freq/ckm
		
		expected_span = 80*dV_MHz	
			
		if total_span > 1.2*expected_span:
		
			drops.append(x)
			
			continue
		
		
			
	
	if len(man_drops) != 0:
	
		for x in range(len(man_drops)):
		
			drops.append(man_drops[x])	
			
		
	#ok, now drop anything that is in the drop array
	
	if len(drops) != 0:
	
		#sort the array in descending order, so that we remove items from the correct places
	
		drops = list(set(drops))
	
		drops.sort(reverse=True)
		
		for x in range(len(drops)):
		
			del obs_chunks[drops[x]]
			peak_freqs = np.delete(peak_freqs, drops[x])
			peak_ints = np.delete(peak_ints, drops[x])		
			
	if len(obs_chunks) == 0:
	
		print('There are no lines in the data to be averaged.  Sorry.')
		
		return		

	#now we go measure the rms of each of the obs chunks
	
	rms_chunks = []
	
	for x in range(len(obs_chunks)):
	
		freq_chunk = obs_chunks[x][0]
		int_chunk = obs_chunks[x][1]
	
		rms = find_peaks(freq_chunk,int_chunk,dV)[1]
		
		rms_chunks.append(rms)

	#how many figures will we have?
	
	n_figs = math.ceil(len(obs_chunks)/16)
	
	n_chunks_left = len(obs_chunks)
	
	for x in range(n_figs):
	
		if plot_chunks == False:
		
			break
	
		chunks_fig, axes = plt.subplots(4,4)
	
		n_chunks_left -= x*16
		
		#how many rows will we need?

		n_rows = math.ceil(n_chunks_left/4)
		
		if n_rows > 4:
		
			n_rows = 4
			
		for y in range(n_rows):
		
			for z in range(4):
			
				chunk_number = x*16 + y*4 + z
		
				try:
					axes[y,z].plot(obs_chunks[chunk_number][0],obs_chunks[chunk_number][1])
				except IndexError:
					break
				
				axes[y,z].annotate('[{}]' .format(chunk_number), xy=(0.1,0.8), xycoords='axes fraction')
				minorLocator = AutoMinorLocator(5)

				plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
				axes[y,z].xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

				axes[y,z].get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
				axes[y,z].get_xaxis().get_major_formatter().set_useOffset(False)
			
	#now need to resample so everything is on the same x-axis (velocity).  So, first, make velocity arrays.
	
	obs_chunks_vel = []
	
	for x in range(len(obs_chunks)):
	
		vel = np.asarray(obs_chunks[x][0])
		
		vel = (obs_chunks[x][0] - peak_freqs[x])*ckm/peak_freqs[x]
		
		obs_chunks_vel.append([vel,obs_chunks[x][1]])
		
	#now for the hard part, resampling these to all be on the same velocity 
	
	obs_chunks_vel_samp = []
	
	vel_ref = np.asarray(obs_chunks_vel[-1][0])
	
	for x in range(len(obs_chunks_vel)):
	
		vel_tmp = np.asarray(obs_chunks_vel[x][0])
		int_tmp = np.asarray(obs_chunks_vel[x][1])
		
		int_interp = np.interp(vel_ref,vel_tmp,int_tmp)
		
		obs_chunks_vel_samp.append([vel_ref,int_interp])
		
		#display these, for checking purposes

		#how many figures will we have?

# 	n_figs = math.ceil(len(obs_chunks_vel_samp)/16)
# 
# 	n_chunks_left = len(obs_chunks_vel_samp)
# 
# 	for x in range(n_figs):
# 
# 		if plot_chunks == False:
#    
# 			break
# 
# 		chunks_fig, axes = plt.subplots(4,4)
# 
# 		n_chunks_left -= x*16
#    
# 			#how many rows will we need?
# 
# 		n_rows = math.ceil(n_chunks_left/4)
#    
# 		if n_rows > 4:
#    
# 			n_rows = 4
# 	   
# 		for y in range(n_rows):
#    
# 			for z in range(4):
# 	   
# 				chunk_number = x*16 + y*4 + z
#    
# 				try:
#    
# 					axes[y,z].plot(obs_chunks_vel_samp[chunk_number][0],obs_chunks_vel_samp[chunk_number][1])
# 			   
# 				except IndexError:
# 		   
# 					break
# 		   
# 				axes[y,z].annotate('[{}]' .format(chunk_number), xy=(0.1,0.8), xycoords='axes fraction')
				
	#Need to now generate a weighting array for the line heights
	
	weights = np.asarray(peak_ints)
		
	#scale the weights array so that the largest value is 1
	
	max_int = np.amax(weights)
	
	weights /= max_int
	
	#now we do the average
	
	vel_avg = vel_ref
	
	int_avg = np.zeros_like(vel_ref)
	
	rms_chunks = np.asarray(rms_chunks)
	
	for x in range(len(obs_chunks_vel_samp)):
	
		int_avg += obs_chunks_vel_samp[x][1]*weights[x]/rms_chunks[x]**2
		
	int_avg /= np.sum(rms_chunks**2)
	
	final_rms = find_vel_peaks(vel_avg,int_avg,dV,width_tweak = 3)[1]
	
	int_avg /= final_rms
	
	plt.ion()	

	fig = plt.figure()
	ax = fig.add_subplot(111)

	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Velocity (km/s)')
	plt.ylabel('SNR')

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	
	ax.plot(vel_avg,int_avg,color='black',label='average',zorder=0)
	
	drops = []	
	
	global vel_stacked,int_stacked
	
	vel_stacked = np.copy(vel_ref)
	int_stacked = np.copy(int_avg)

def cut_spectra(write=False,outputfile='cut.txt',n_fwhm=30):

	if gauss == True:
	
		print('cut_spectra() does not yet work with gaussian simulated spectra, only stick spectra.  Please resimulate with gauss=False and try again.')
		
		return

	freq_cut = []
	int_cut = []

	for x in freq_sim:

		i = (np.abs(freq_obs - x)).argmin()
	
		if abs(freq_obs[i]-x) < 1:
	
			#get resolution in MHz near point
		
			res_tmp = abs((freq_obs[i]-freq_obs[i+10])/10)
		
			#get resolution in km/s near point
		
			vel_res_tmp = abs(res_tmp*ckm/freq_obs[i])
		
			#calculate number of points equivalent to <<n_fwhm>> * FHWM
		
			pts_tmp = int(n_fwhm*dV/vel_res_tmp)
		
			#find the indexes at +/- 30 FWHM
		
			i_low = i-pts_tmp
		
			i_high = i+pts_tmp
		
			for x in range(i_low,i_high):
			
				freq_cut.append(freq_obs[x])
				int_cut.append(int_obs[x])
			
	if len(freq_cut) == 0:
	
		print('Something is wrong - no cuts were made.  Either there\'s no matching data or the script is broken.')	
			
	if write==False:
	
		return freq_cut,int_cut
		
	if write==True:
	
		with open(outputfile,'w') as output:
		
			for x in range(len(freq_cut)):
			
				output.write('{} {}\n' .format(freq_cut[x],int_cut[x]))
				
		return

#calc_tbg generates the background temperature wave.  The user should never call this directly.  A meta function to update tbg and re-run the simulation is provided as update()

def calc_tbg(tbg_params,tbg_type,tbg_range,frequencies):
		
	'''
	calc_tbg generates the background temperature wave.  The user should never call this directly.  A meta function to update tbg and re-run the simulation is provided as update().  If ranges are defined, any chunk that is not in the defined range defaults to 2.7 K.
	'''
	
	#figure out how many ranges we're dealing with
	
	n_ranges = len(tbg_range)
	
	#initialize a numpy array for tbg that is the same length as the requested array of covered frequencies
	
	tbg = np.zeros_like(frequencies)
	
	tbg = np.float64(tbg)

	#first, make sure to get tbg_params into list form if it's a single integer or float
	
	if type(tbg_params) == int or type(tbg_params) == float:
		
		tbg_params = [tbg_params]
			
	#Will need to run several different possible scenarios here
	
	if tbg_type == 'poly':
	
		#if there's no range specified...
	
		if n_ranges == 0:

			#we'll cycle through each order individually

			for x in range(len(tbg_params)):
				
				#create a temporary array to handle what is going to be added to tbg for this order
				
				tmp_tbg = np.zeros_like(frequencies)
				
				tmp_tbg = np.float64(tmp_tbg)
				
				tmp_tbg = tbg_params[x]*frequencies**x
				
				tbg += tmp_tbg
				
			tbg[tbg == 0] = 2.7
		
			return tbg
		
		else:
		
			#before we can add to the tbg array, we need to find the indices we'll be wanting to work with for each range.  So.
			
			for i in range(n_ranges):
			
				#first, get the ll and ul for the range in question.
			
				ll = tbg_range[i][0]
				ul = tbg_range[i][1]
				
				#next, let's get the indexes i_low and i_high this range covers in our frequencies.
				
				try:
					i_low = np.where(frequencies > ll)[0][0]
				except IndexError:
				
					#first, check and make sure this range is actually in the simulation.  If the ll is higher than the highest frequency in frequencies, just move on
					
					if frequencies[-1] < ll:
						continue
						
					#otherwise, if the simulation starts after the lower limit, then we just take the first point in frequencies
					
					else:
						i_low = 0
						
				try:
					i_high = np.where(frequencies > ul)[0][0]
				except IndexError:
				
					#If we can't find a point in frequencies that's above the upper limit, then the last point in the simulation is the upper limit index
					i_high = len(frequencies)
						
				#now that we have the indices i_low and i_high we are going to want to apply our tbg to, we can do as above
				
				#first, lets just get the constants for this particular range
					
				constants = tbg_params[i]
				
				if type(constants) == float or type(constants) == int:
				
					constants = [constants]
				
				#now we cycle through the orders again
				
				for x in range(len(constants)):
				
					#create a temporary array to handle what is going to be added to tbg for this order
					
					tmp_tbg = np.zeros_like(frequencies)
					
					tmp_tbg = np.float64(tmp_tbg)
					
					tmp_tbg = constants[x]*freq_sim**x
					
					tbg[i_low:i_high] += tmp_tbg[i_low:i_high]		
					
			tbg[tbg == 0] = 2.7
			
			return tbg			
						
	elif tbg_type == 'power':
	
		#if there's no range specified...
	
		if n_ranges == 0:

			#create a temporary array to handle what is going to be added to tbg for this order
				
			tmp_tbg = np.zeros_like(frequencies)
			
			tmp_tbg = np.float64(tmp_tbg)
		
			tmp_tbg = tbg_params[0]*freq_sim**tbg_params[1] + tbg_params[3]
		
			tbg += tmp_tbg
		
			tbg[tbg == 0] = 2.7

			return tbg
		
		else:
		
			#before we can add to the tbg array, we need to find the indices we'll be wanting to work with for each range.  So.
			
			for i in range(n_ranges):
			
				#first, get the ll and ul for the range in question.
			
				ll = tbg_range[i][0]
				ul = tbg_range[i][1]
				
				#next, let's get the indexes i_low and i_high this range covers in our freq_sim.
				
				try:
					i_low = np.where(frequencies > ll)[0][0]
				except IndexError:
				
					#first, check and make sure this range is actually in the simulation.  If the ll is higher than the highest frequency in frequencies, just move on
					
					if frequencies[-1] < ll:
						continue
						
					#otherwise, if the simulation starts after the lower limit, then we just take the first point in frequencies
					
					else:
						i_low = 0
						
				try:
					i_high = np.where(frequencies > ul)[0][0]
				except IndexError:
				
					#If we can't find a point in the simulation that's above the upper limit, then the last point in the simulation is the upper limit index
					
					i_high = len(frequencies)
						
				#now that we have the indices i_low and i_high we are going to want to apply our tbg to, we can do as above
				
				#first, lets just get the constants for this particular range
				
				constants = tbg_params[i]
			
				tmp_tbg = np.zeros_like(frequencies)
				
				tmp_tbg = np.float64(tmp_tbg)
		
				tmp_tbg = constants[0]*frequencies**constants[1] + constants[3]				
					
				tbg[i_low:i_high] += tmp_tbg[i_low:i_high]		
					
			tbg[tbg == 0] = 2.7
			
			return tbg					
	
	else:
	
		print('Your Tbg calls are not set properly. This is likely because you have tbg_type set to something other than poly or power. Tbg has been defaulted to the CMB value of 2.7 K across your entire simulation.  Please see the Tbg documentation if this is not what you desire.')
		
		tbg = np.zeros_like(frequencies)
		
		tbg = np.float64(tbg)
		
		tbg += 2.7
		
		return tbg
				
				
#update is a general call to just re-run the simulation, if the user has modified any generalized variables themselves like Tbg stuff, or updated vlsr or dV, etc, without using mod functions.

def update():

	'''
	A general call to just re-run the simulation, if the user has modified any generalized variables themselves like Tbg stuff, or updated vlsr or dV, etc, without using mod functions.
	'''

	global freq_sim,int_sim
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend()
	fig.canvas.draw()
	
	save_results('last.results')

#reset_tbg just resets all the tbg parameters to the defaults and calls update()

def reset_tbg():

	global tbg_params,tbg_type,tbg_range
	
	tbg_params = 2.7
	tbg_type = 'poly'
	tbg_range = []
	update()

#############################################################
#							Classes for Storing Results		#
#############################################################	

class Molecule(object):

	def __init__(self,name,catalog_file,tag,gup,dof,error,qn1,qn2,qn3,qn4,qn5,qn6,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim,aij,sijmu):
	
		self.name = name
		self.catalog_file = catalog_file
		self.tag = tag
		self.gup = gup
		self.dof = dof
		self.error = error
		self.qn1 = qn1
		self.qn2 = qn2
		self.qn3 = qn3
		self.qn4 = qn4
		self.qn5 = qn5
		self.qn6 = qn6
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
		self.aij = aij
		self.sijmu = sijmu

	
#############################################################
#							Run Program						#
#############################################################