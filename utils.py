# a bunch of useful routines

import numpy as np
import scipy.interpolate as sciint
import scipy.fftpack as fftpack
import scipy.ndimage.filters as scifilt
import scipy.signal as scisig
import tsinsar.mints as mints

def _diagonalConvolution(image, xm, ym, dx, dy, 
                         covariance, inverse=False):
    '''
    Convolve an image of the size of the interferogram with a diagonal covariance.
    Returns the convoluted function.

    Args:
        * image         : 1d array containing the data
        * xm            : 1d array the size of image containing the x coordinates
        * ym            : 1d array the size of the 
        * covariance    : Float diagonal value
        * inverse       : convolution by the inverse function (default is False)
    '''
    if inverse:
        return image/covariance 
    else:
        return image*covariance

def _detrend(intImage):
    '''
    Remove a ramp from the image.
    '''
    
    # Coordinates
    x = np.arange(intImage.shape[1])
    y = np.arange(intImage.shape[0])
    x,y = np.meshgrid(x,y)

    # Invert
    G = np.vstack((x.flatten(), y.flatten(), np.ones((np.prod(x.shape),)))).T
    m, n, p, res = np.linalg.lstsq(G, intImage.flatten())

    # All done
    flatImage = (intImage.flatten()-np.dot(G,m)).reshape(intImage.shape)
    trend = intImage - flatImage
    return flatImage, trend

def _expConvolution(image, xm, ym, dx, dy, 
                    covariance, inverse=False, donotinterpolate=False):
    '''
    Convolve an image of the size of the interferogram with an exponential (or inverse exponential)
    function.
    The function is of the form:
        f(x1,x2) = Sigma^2 exp(-||x1,x2||/Lambda)
        where ||x1,x2|| is the distance between pixels x1 and x2.
    Returns the convoluted function.
    
    Args:
        * image         : 1d array containing the data
        * xm            : 1d array the size of image containing the x coordinates
        * ym            : 1d array the size of the 
        * covariance    : (Lambda,Sigma)
                            - Lambda    : Float, Correlation length
                            - Sigma     : Float, Amplitude of the correlation function  
        * inverse       : convolution by the inverse function (default is False)
        * donotinterpolate: Bypasses the interpolation (image has to be a 2d array)
    '''

    # One simple case
    if (image==0.).all():
        return image

    # Get Lambda and Sigma
    Lambda, Sigma = covariance

    # Get number of data
    nPoints = float(len(ym))

    # Interpolate
    if donotinterpolate:
        intImage = image
        ymin = 0
        xmin = 0
    else:
        intImage, xmin, ymin = _mintsInterp(image, xm, ym)
    nTotal = float(np.prod(intImage.shape))

    # Padding size
    padlength = intImage.shape[0]/2, intImage.shape[1]/2

    # Different case for inverse or direct
    if not inverse:
        
        # Zero Padding
        intImage = np.pad(intImage, ((padlength[0], padlength[0]),
                                     (padlength[1], padlength[1])),
                                     mode='constant')
        pad = np.zeros(intImage.shape)
        pad[padlength[0]+ym-ymin, padlength[1]+xm-xmin] = 1.
        intImage *= pad

    else:
        
        # Linear Padding
        intImage = np.pad(intImage, ((padlength[0], padlength[0]),
                                     (padlength[1], padlength[1])),
                                     mode='reflect')
        l,c = intImage.shape
        hl = scisig.gaussian(l, padlength[0])
        hc = scisig.gaussian(c, padlength[1])
        intImage = hl[:,np.newaxis]*intImage*hc[np.newaxis,:]

    # For debugging
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(intImage, interpolation='nearest')
    #plt.colorbar()
    #plt.show()
    # For debugging

    # Do the FFT
    fm = fftpack.fft2(intImage)
    u = fftpack.fftfreq(intImage.shape[1], d=dx)
    v = fftpack.fftfreq(intImage.shape[0], d=dy)
    u,v = np.meshgrid(u,v)

    # Select the convolution function
    if inverse:
        H = _expInvF
    else:
        H = _expF

    # Convolve with the function
    dfm = H(u,v,Lambda,Sigma)*fm
    dm = np.real(fftpack.ifft2(dfm))
    
    # un-padding
    dm = dm[padlength[0]:-padlength[0], padlength[1]:-padlength[1]]

    # all done
    if donotinterpolate:
        return dm
    else:
        return dm[ym-ymin, xm-xmin]

def _nearestInterp(image, xm, ym):
    '''
    Computes the interpolation on the smallest rectangle inside the data
    Args:
        * image         : 1d array of Data
        * xm            : 1d array of x-coordinates
        * ym            : 1d array of y-coordinates
    '''
    
    # Get the minimum area
    xmin, xmax = np.min(xm), np.max(xm)+1
    ymin, ymax = np.min(ym), np.max(ym)+1
    x = np.arange(xmax-xmin)
    y = np.arange(ymax-ymin)
    x,y = np.meshgrid(x,y)

    # Save what's not to be interpolated
    intImage = np.zeros((ymax-ymin, xmax-xmin)); intImage[:,:] = np.nan
    intImage[ym-ymin, xm-xmin] = image
    ii = np.flatnonzero(np.isnan(intImage))

    # Get x and y
    x = x.flatten()[ii]
    y = y.flatten()[ii]

    # Create the interpolator
    interpolator = sciint.NearestNDInterpolator(np.vstack((xm-xmin, ym-ymin)).T, image)

    # Interpolate
    intImage[y,x] = interpolator(x,y)
    del interpolator, x, y

    # Is there still holes?
    if np.isnan(intImage).any():
        _fillHoles(intImage)
    
    # All done
    return intImage, xmin, ymin

def _linearInterp(image, xm, ym):
    '''
    Computes the interpolation on the smallest rectangle inside the data
    Args:
        * image         : 1d array of Data
        * xm            : 1d array of x-coordinates
        * ym            : 1d array of y-coordinates
    '''
    
    # Get the minimum area
    xmin, xmax = np.min(xm), np.max(xm)+1
    ymin, ymax = np.min(ym), np.max(ym)+1
    x = np.arange(xmax-xmin)
    y = np.arange(ymax-ymin)
    x,y = np.meshgrid(x,y)

    # Save what's not to be interpolated
    intImage = np.zeros((ymax-ymin, xmax-xmin)); intImage[:,:] = np.nan
    intImage[ym-ymin, xm-xmin] = image
    ii = np.flatnonzero(np.isnan(intImage))

    # Get x and y
    x = x.flatten()[ii]
    y = y.flatten()[ii]

    # Create the interpolator
    interpolator = sciint.LinearNDInterpolator(np.vstack((xm-xmin, ym-ymin)).T, image)

    # Interpolate
    intImage[y,x] = interpolator(x,y)
    del interpolator, x, y

    # Is there still holes?
    if np.isnan(intImage).any():
        _fillHoles(intImage)
    
    # All done
    return intImage, xmin, ymin

def _mintsInterp(image, xm, ym):
    '''
    Computes the interpolation on the smallest rectangle inside the data
    using the interpolation from MInTS
    Args:
        * image         : 1d array of Data
        * xm            : 1d array of x-coordinates
        * ym            : 1d array of y-coordinates
    '''

    # Get the minimum area
    xmin, xmax = np.min(xm), np.max(xm)+1
    ymin, ymax = np.min(ym), np.max(ym)+1
    x = np.arange(xmax-xmin)
    y = np.arange(ymax-ymin)
    x,y = np.meshgrid(x,y)

    # Save what's not to be interpolated
    intImage = np.zeros((ymax-ymin, xmax-xmin)); intImage[:,:] = np.nan
    intImage[ym-ymin, xm-xmin] = image

    # Interpolate using inpaints
    intImage = mints.inpaint(intImage)

    # All done
    return intImage, xmin, ymin

def _fillHoles(ifg):
    '''
    Fill the holes that are outside the Qhull area
    Assumption: if the hole has not been filled through linear interpolation, then
                it is a corner...
    '''
 
    # While loop
    while np.isnan(ifg).any():
    
        # Take the nans
        lst, cst = np.where(np.isnan(ifg))

        # Isolate the area
        subimage, x0, x1, y0, y1 = _isolateNaNs(ifg, lst[0], cst[0])

        # Do a small interpolation
        x = range(x0, x1); y = range(y0, y1)
        x,y = np.meshgrid(x,y)
        
        # Get the number of non-NaN 
        count = np.flatnonzero(np.isfinite(subimage)).shape[0]

        # if low number of points
        if count<4000:
            inter = sciint.Rbf(x[np.isfinite(subimage)], 
                               y[np.isfinite(subimage)], 
                               subimage[np.isfinite(subimage)])
        else:
            inter = sciint.NearestNDInterpolator(np.vstack((x[np.isfinite(subimage)], 
                                                           y[np.isfinite(subimage)])).T,
                                                subimage[np.isfinite(subimage)])
        subimage[np.isnan(subimage)] = inter(x[np.isnan(subimage)], 
                                             y[np.isnan(subimage)])

        # Put back subimage in image
        ifg[y0:y1,x0:x1] = subimage

        # Clean up 
        del inter, x, y, subimage

    return

def _isolateNaNs(image, lst, cst):
    '''
    Returns a subimage that encompasses a region with NaNs starting from a pixel
    '''

    def update(image, xmin, xmax, ymin, ymax):
        if xmin>0: xmin -= 1
        if xmax<image.shape[1]-1: xmax += 1
        if ymin>0: ymin -= 1
        if ymax<image.shape[0]-1: ymax += 1
        return xmin, xmax, ymin, ymax

    def check(image, xmin, xmax, ymin, ymax):
        # Xmin
        if (np.isnan(image[ymin:ymax+1,xmin]).any()):
            xLow = False
        else:
            xLow = True
        if xmin==0: xLow = True
        # Ymin
        if (np.isnan(image[ymin,xmin:xmax+1]).any()):
            yLow = False
        else:
            yLow = True
        if ymin==0: yLow = True
        # Ymax
        if (np.isnan(image[ymax,xmin:xmax+1]).any()):
            yUp = False
        else:
            yUp = True
        if ymax==image.shape[0]-1: yUp = True
        # Xmax
        if (np.isnan(image[ymin:ymax+1,xmax]).any()):
            xUp = False
        else:
            xUp = True
        if xmax==image.shape[1]-1: xUp = True
        return xLow*yLow*yUp*xUp

    # Start
    xmin, xmax, ymin, ymax = update(image, cst, cst, lst, lst)

    # While loop
    while not check(image, xmin, xmax, ymin, ymax):
        xmin, xmax, ymin, ymax = update(image, xmin, xmax, ymin, ymax)

    # All done
    return image[ymin:ymax+1, xmin:xmax+1], xmin, xmax+1, ymin, ymax+1

# Covariance functions
def _expF(u, v, lam, sig):
    return (sig*sig*lam*lam*2.0*np.pi)/((1 + (lam*u*2.0*np.pi)**2 + \
            (lam*v*2.0*np.pi)**2)**(1.5))

def _expInvF(u, v, lam, sig):
    return 1./_expF(u, v, lam, sig)

def _gaussF(u, v, lam, sig):
    return sig**2/(2*np.pi) *np.exp(-lam*lam*(u*u+v*v)/2.)

def _gaussInvF(u, v, lam, sig):
    return 1./_gaussF(u, v, lam, sig)

# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

def _matrixConvolution(m, x, y, dx, dy, Lambda, Sigma, inverse=False):
    '''
    Do the matrix form convolution. Testing mode only...

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    Make sure you have a tiny image (less than 100x100)
    Otherwise your computer will die in pain...
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    '''

    X,XX = np.meshgrid(x.flatten()*dx, x.flatten()*dx)
    Y,YY = np.meshgrid(y.flatten()*dy, y.flatten()*dy)
    Cov = Sigma*Sigma*np.exp(-np.sqrt( (X-XX)**2 + (Y-YY)**2 )/Lambda)
    if inverse:
        Cov = np.linalg.inv(Cov)
    mCm = np.dot(Cov, m.flatten()).reshape(m.shape)
    if inverse:
        mCm /= dx*dy
    else:
        mCm *= dx*dy
    return mCm


