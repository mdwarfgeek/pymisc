# Alternative pure Python sky background estimation routines using
# median of list (for floating point data) rather than histogramming.

import math
import numpy

from medsig import *

def skylevel(a, mask=None, clip_low=None, clip_high=3.0):
  if mask is not None:
    atmp = a[mask].flatten()
  else:
    atmp = a.flatten()

  npt = atmp.size
  sky, noise = medsig(atmp)

  for i in range(5):
    ww = numpy.ones_like(atmp, dtype=numpy.bool)
    if clip_low is not None:
      ww[(atmp-sky) < -clip_low*noise] = 0
    if clip_high is not None:
      ww[(atmp-sky) > clip_high*noise] = 0
    
    atmp = atmp[ww]
    
    nnew = atmp.size
    nclip = npt - nnew
    if nclip <= 0:
      break

    npt = nnew
    sky, noise = medsig(atmp)

  return sky, noise

def skyann(map, xcent, ycent, rinn, rout,
           mask=None, clip_low=None, clip_high=3.0):

  ny, nx = map.shape
  
  nobj = xcent.size
  skylev = numpy.empty_like(xcent)
  skyrms = numpy.empty_like(xcent)

  for iobj in range(nobj):
    # Extract coords for this object and convert to zero-based.
    if numpy.ndim(xcent) == 0:
      thisxcent = xcent - 1
    else:
      thisxcent = xcent[iobj] - 1
    if numpy.ndim(ycent) == 0:
      thisycent = ycent - 1
    else:
      thisycent = ycent[iobj] - 1

    # Bounds.
    rb = rout + 0.5
    
    xmin = int(math.floor(thisxcent - rb))
    if xmin < 0:
      xmin = 0

    xmax = int(math.ceil(thisxcent + rb))
    if xmax >= nx:
      xmax = nx-1

    ymin = int(math.floor(thisycent - rb))
    if ymin < 0:
      ymin = 0

    ymax = int(math.ceil(thisycent + rb))
    if ymax >= ny:
      ymax = ny-1

    # Extract what we need.
    thismap = map[ymin:ymax+1,xmin:xmax+1]
    if mask is not None:
      thismask = mask[ymin:ymax+1,xmin:xmax+1]
    else:
      thismask = None
      
    xtmp = numpy.arange(xmin, xmax+1)
    ytmp = numpy.arange(ymin, ymax+1)
    xx = numpy.tile(xtmp, (ymax-ymin+1, 1))
    yy = numpy.transpose(numpy.tile(ytmp, (xmax-xmin+1, 1)))

    # Make mask for desired annulus.
    rr = numpy.hypot(xx-thisxcent, yy-thisycent)
    ww = numpy.logical_and(rr >= rinn, rr <= rout)

    annmap = thismap[ww]
    if thismask is not None:
      annmask = thismask[ww]
    else:
      annmask = None
      
    # Compute sky level.
    thisskylev, thisskyrms = skylevel(annmap,
                                      mask=annmask,
                                      clip_low=clip_low,
                                      clip_high=clip_high)

    if numpy.ndim(xcent) == 0:
      skylev = thisskylev
      skyrms = thisskyrms
    else:
      skylev[iobj] = thisskylev
      skyrms[iobj] = thisskyrms

  return skylev, skyrms
