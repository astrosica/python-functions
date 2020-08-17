import aplpy
import numpy as np
import matplotlib.pyplot as plt
import astropy.wcs as wcs
from astropy.io import fits
from licpy.lic import runlic

def astroLIC(I_dir,Q_dir,U_dir,length=0.25,toIAU=False,wcs=True):
	'''
	Runs the astro-LIC package.

	Input
	I_dir  : path to Stokes I file                                                  [str]
	Q_fir  : path to Stokes Q file                                                  [str]
	U_dir  : path to Stokes U file                                                  [str]
	length : fraction of image length to compute LIC across                         [default=0.25]
	toIAU  : if True, convert Stokes Q and U to IAU convention                      [default=False]
	wcs    : if True, plot results using WCS as defined by the Stokes I FITS header [default=True]

	'''

	# extract Stokes images
	I_data = fits.getdata(I_dir)
	Q_data = fits.getdata(Q_dir)
	U_data = fits.getdata(U_dir)

	# compute the plane-of-sky magnetic field orientation in IAU convention
	Bangle = fBangle(Q_data,U_data,toIAU=toIAU)

	# compute the LIC texture
	texture = fLICtexture(Bangle,length)

	# plot!
	fplotLIC(I_dir,texture,wcs)


def fBangle(Q_data,U_data,toIAU=False):
	'''
	Computes the plane-of-sky magnetic field angle. Stokes maps must be in IAU convention!
	
	Input
	Q_data : Stokes Q map                                                       [float]
	U_data : Stokes U map                                                       [float]
	toIAU  : if True, converts Stokes Q and U maps from COSMO to IAU convention [default=False]

	Output
	Bangle : magnetic field angles between 0 and pi                             [radians]
	'''

	# copy to ensure we're not rotating original data
	Q = np.copy(Q_data)
	U = np.copy(U_data)

	# flip both Q and U to convert from electric field to magnetic field orientation
	Q *= -1.
	U *= -1.

	if toIAU==True:
		# if converting from COSMOS to IAU convention, flip sign of Stokes U
		U *= -1.

	# compute magnetic field angle on the domain [0,pi)
	Bangle = np.mod(0.5*np.arctan2(U,Q), np.pi)

	return Bangle

def fLICtexture(Bangle,length=0.25):
	'''
	Computes the LIC texture to be overplotted on an image using LicPy.

	Input
	Bangle : plane-of-sky magnetic field angles on the domain 0 and pi [radians]
	length : fraction of image length to compute LIC across            [default=0.25]

	Output
	texture : LIC texture                                              [float]
	'''

	# LIC measures angles from the horizontal while IAU polarization angles are 
	# measured from the vertical; this translates the magnetic field angle accordingly
	Bangle += (np.pi/2.)

	# x- and y-components of magnetic field
	b_x = np.sin(Bangle)
	b_y = np.cos(Bangle)

	# length scale; typically 25% of image size but can be adjusted
	L_z = np.shape(Bangle)
	L   = int(length*L_z[0])

	# run licpy for texture
	texture = runlic(b_x,b_y,L)

	return texture

def fplotLIC(I_dir,texture,wcs=True):
	'''
	Plots the LIC texture overlaid on the Stokes I image and saves the image in the working directory with "_LIC.pdf" appended to the file name.

	Input:
	I_file  : path to Stokes I file                                       [str]
	texture : LIC texture                                                 [float]
	wcs     : if True, plots using the WCS defined by the Stokes I header [default=True]
	'''

	# extract Stokes I data
	I_data = fits.getdata(I_dir)
	
	if wcs==True:
		# use WCS defined by Stokes I FITS header
		fig = plt.figure(figsize=(5,5))
		f1  = aplpy.FITSFigure(I_dir,figure=fig,subplot=(1,1,1))
		f1.show_colorscale(cmap="plasma")
		# texture
		plt.imshow(texture, origin="lower",alpha=0.4,cmap="binary",clim=[np.mean(texture)-np.std(texture),np.mean(texture)+np.std(texture)])
		# axis labels
		f1.axis_labels.set_font(size=20)
		# tick labels
		f1.tick_labels.show()
		f1.tick_labels.set_font(size=20)
		# colour bar
		f1.add_colorbar()
		f1.colorbar.set_axis_label_text(r"$I_{353}\,(\mathrm{MJy/sr})$")
		f1.colorbar.set_axis_label_font(size=20)
		# remove whitespace
		plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
		# save + show
		fig.canvas.draw()
		f1.save(I_dir.split(".fits")[0]+"_LIC.pdf")
	else:
		# use pixel coordinates
		fig = plt.figure(figsize=(6,4))
		ax  = fig.add_subplot(111)
		im  = ax.imshow(I_data,cmap="plasma",origin="lower")
		plt.imshow(texture, origin="lower",alpha=0.4,cmap="binary",clim=[np.mean(texture)-np.std(texture),np.mean(texture)+np.std(texture)])
		# axis labels
		plt.xlabel("x (pixels)",fontsize=20)
		plt.ylabel("y (pixels)",fontsize=20)
		# tick labels
		ax.tick_params(axis="both",which="major",labelsize=15)
		# colour bar
		cbar = plt.colorbar(im,ax=ax,orientation="vertical")
		cbar.set_label(r"$I_{353}\,(\mathrm{MJy/sr})$",fontsize=20)
		# remove whitespace
		plt.subplots_adjust(top=1,bottom=0,right=1,left=0.1,hspace=0,wspace=0)
		# save + show
		plt.savefig(I_dir.split(".fits")[0]+"_LIC.pdf")
		plt.show()

