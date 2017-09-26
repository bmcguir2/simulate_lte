#####################################################
#                                                   #
#               viewspectrum.py README              #
#                                                   #
#####################################################

Last Updated: Jan 10, 2017

This program is designed to read molecular data from an SPCAT formatted catalog file, and simulate a spectrum of the molecule, given a variety of parameters.  It can provide stick spectra or Gaussian simulations (at some computational expense for large numbers of lines).  It can plot those spectra over a laboratory or observational spectrum.  Simulations can be stored into memory, and a combined simulation of all molecules can be generated and displayed.  Simulations can be written out to and ascii file.  The current state of the program is written to a (human-readable) output file (default: last.results), and manual saves can also be performed.  The program can restore to the state of any save file, if the appropriate catalogs are present as well.

The default simulation parameters are:

Temperature = 300 K
LSR Velocity = 0.0 km/s
Intensity Scaling Factor: 1.0
Lower Limit Cut-off for Simulation: none
Upper Limit Cut-off for Simulation: none
Linewidth: 5.0 km/s
Temperature Catalog was Simulated At: 300 K
Simulate Gaussian Profiles? True
Thermal Continuum: none

For the Gaussian simulations, the program is hard-coded to provide simulations at +/- 10 FWHM from the line center, and with a resolution that provides 15 points across the FWHM.

Be warned!  The lower limits and upper limits will need to be adjusted for any sufficiently-complex catalog.  You can simulate the full CO catalog, for example, but the full glycolaldehyde catalog would take a few hours.  If the simulation takes more than ~30 seconds, there's a warning that pops up and gives an estimate for how much longer it will take.  This estimate is probably pretty poor.  Also bear in mind that this simulation will likely be performed many times as you adjust parameters, so anything over a few seconds could get pretty tedious.

It is imperative for accurate temperature scaling that the catalogs used either be simulated at 300 K (the default if you get them from JPL or CDMS) OR that you set CT to the appropriate value BEFORE loading in the catalog.

For a select few catalogs which use odd notation for quantum numbers (+/-), the program won't be able to parse them.  You can get around this by manually adding an exception for that particular catalog, along with a calculation of the partition function from a polynomial fit.  You can see how this is done in calc_q.

Finally, note that the program is calculating partition functions on demand from the catalogs, and only considering lines within the cutoffs.  That means that the absolute value of these partition functions is almost certainly not correct.  This has no effect on the accuracy of the relative intensities of the lines for a single molecule. However, one cannot compare the intensities between two molecules accurately using this method.

#####################################################
#                                                   #
#                   Getting Started                 #
#                                                   #
#####################################################

This program is designed to run from inside iPython.  You can get some help on any of these functions from within the program with help(function).

Load the program into iPython.

> %run -i viewspectrum.py

Set the lower and upper limits (no, seriously) to x and y MHz.  These can be single values (either int or float), or can be an array of start and stop value pairs.  i.e. for 50 - 55 GHz and 60 - 65 GHz, specify ll = [50000,60000] ul = [55000,65000]. 

> ll = x
> ul = y

Load a molecule into the program from catalog x. This will pop up a graph with the simulation.  

> load_mol('x')

Compare the simulation to some observations or laboratory data you have in ascii format (frequency intensity) from file 'x'.  Delimiter doesn't matter, as long as it is consistent.  Will detect a standard .ispec header from casaviewer export, and will apply a GHz flag (see below) automatically if necessary, as well as populating the 'coords' variable with the coordinates from the header.

> read_obs('x')

If your observations are in GHz, you can set a flag in the program so that they are read in and appropriately converted to MHz.  This flag will be saved, and any restore will then know to restore properly as well.  Must do this before calling read_obs().  

> use_GHz()

You can toggle these observations on and off in the plot:

> obs_off()
> obs_on()

Now you can modify the parameters and see their effects on your simulation:

> modT(x)	#temperature
> modS(x)	#linear intensity scaling factor
> modV(x)	#vlsr
> moddV(x)	#linewidth

If you have optically-thick lines, you can set a thermal continuum cutoff to x K.  Note that if the molecules are coming from two different spatial regions, or are masing, the program will not handle this correctly and the overall intensity will be lower than it should be:

> thermal = x

If you want to see a summary of the active simulation parameters:

> status()

Once you are happy with the way your simulation looks, store it into memory with a name x:

> store('x')

Or, call it empty and it will save the name using the basename of the catalog file (everything up to the first period):

> store()

You can see a (not terribly nicely formatted, yet) summary of what you have stored:

> sim

If you want to recall what temperature, dV, etc. the stored simulation was at:

> sim['x'].T	#temperature
> sim['x'].S	#scaling factor
etc.

Once stored, you can overplot that simulation back onto the current plot:

> overplot('x')

You can close any plot either by just x-ing out of the window, or, if you fancy it (only works for the most recent plot):

> close()

If you accidentally close the plot, or just want another one, you can do that.  Note that only the last graph opened is ever editable again.

> make_plot()

Opening a new plot this way will wipe out any over-plotted spectra, however.  If you want to get back to where you were, you can restore from the autosave file:

> restore('last.results')

You can also save your progress at any point to file 'x' (remember the auto-save only goes back to your last graph):

> save_results('x')

If you decide you want to go back to working directly with an old simulation, you can recall it.  This will wipe out whatever the currently-active simulation is, so be sure to store that one if you want it for later:

> recall('x')

Remember, recall and load_mol both wipe out the current simulation and there is no way to recover that data.  Store it or lose it!

Once you have a few stored simulations, it can be useful to see a co-added spectrum of them.  You can do this!  First, add them all together:

> sum_stored()

Then, overplot the result with a special function, which ensure it's in a color that will never ever be used for anything else (just as black is always observations and red is always the active simulation):

> overplot_sum()

If you need to change the legend on the figure, you can do that, too, by making it x number of columns, and with a font-size lsize, which can be a float, int, or a string (see help file for possible options for strings).

> fix_legend(x,lsize)

If you want to clear an over-plotted spectrum x back off the plot, you can do that:

> clear_line('x')

Finally, if you want to save a simulation to an output ascii file output_file, you can do that.  Use 'current' for the active simulation, 'sum' for the summation, or the name of the stored simulation for y:

> write_spectrum('y','output_file')

That's it!  If you find any issues, please let me know.
