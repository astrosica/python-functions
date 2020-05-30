import aplpy
import numpy as np
import pylab as plt
import astropy.wcs as wcs
from astropy.io import fits
from licpy.lic import runlic

def fBangle(Q_file,U_file,toIAU=False):
	'''
	Computes the plane-of-sky magnetic field angle. Stokes maps must be in IAU convention!
	
	COSMO polarization angle convention to IAU B-field convention = flip sign of Stokes U
	
	Input
	Q_file : path to Stokes Q file
	U_file : path to Stokes U file
	toIAU  : if True, converts from COSMO to IAU convention (default=False)

	Output
	Bangle : magnetic field angles [radians]
	'''

	# flip both Q and U to convert from electric
	# field to magnetic field (equivalent to a
	# 90 degree rotation)
	Q = fits.getdata(Q_file)
	U = fits.getdata(U_file)

	Q *= -1.
	U *= -1.

	if toIAU==True:
		# if converting from COSMOS to IAU convention
		# flip sign of Stokes U
		U *= -1.

	# magnetic field angle
	Bangle = np.mod(0.5*np.arctan2(U,Q), np.pi)

	return Bangle

def fLICtexture(Bangle):
	'''
	Creates the LIC texture to be overplotted on an image.

	Input
	Bangle : magnetic field angles in IAU convention [radians]

	Output
	texture: LIC texture
	'''

	# the IAU convention meaasures angles from north while
	# LIC measures angles from west; rotate angles for LIC
	Bangle = (np.pi/2.)-Bangle

	# compute the x- and y-components of magnetic field;
	# this is why polarization data must be in the IAU convention
	# otherwise the following will be the wrong components of the field
	b_x =  np.sin(Bangle)
	b_y = -np.cos(Bangle)

	# length scale; typically 25% of image size
	L_z = np.shape(Bangle)
	L   = int(0.25*L_z[0])

	# run LIC to compute texture
	texture = runlic(b_x,b_y,L)

	return texture

def fplotLIC(image_file,texture,wcs=True):
	'''
	Overplots the LIC texture on an image.

	Input
	image_file : image to overplot LIC texture
	texture    : LIC texture
	wcs        : if True, uses the header to plot in WCS
	'''

	image_data,image_header = fits.getdata(image_file,header=True)
	
	if wcs==True:
		fig = plt.figure(figsize=(5,5))
		f1  = aplpy.FITSFigure(image_file,figure=fig,subplot=(1,1,1))
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
		f1.colorbar.set_axis_label_text(r"label")
		f1.colorbar.set_axis_label_font(size=20)
		
		# remove whitespace
		plt.subplots_adjust(top = 1,bottom=0,right=1,left=0,hspace=0,wspace=0)
		
		fig.canvas.draw()
		f1.save(image_file.split(".fits")[0]+"_C_LIC.pdf")
	else:
		fig = plt.figure(figsize=(5,5))
		ax = fig.add_subplot(111)
		im = ax.imshow(image,cmap="plasma",origin="lower")
		plt.xlabel("pixels")
		plt.ylabel("pixels")
		plt.imshow(texture, origin="lower",alpha=0.4,cmap="binary",clim=[np.mean(texture)-np.std(texture),np.mean(texture)+np.std(texture)])
		plt.gca().add_collection(lc)
		plt.colorbar(im, ax=ax, orientation="vertical")
		plt.show()
		plt.savefig(imagefile.split(".fits")[0]+"_angles.pdf")

