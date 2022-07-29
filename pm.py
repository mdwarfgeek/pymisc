import math
import numpy

# units rad, rad/yr, yr.  catalogue (sky projected).

def pmcorr(ra, de, pmra, pmde, ep1, ep2):
  sa = numpy.sin(ra)
  ca = numpy.cos(ra)
  sd = numpy.sin(de)
  cd = numpy.cos(de)
  
  x = ca * cd
  y = sa * cd
  z = sd

  dxdt = -pmra * sa - pmde * ca*sd
  dydt =  pmra * ca - pmde * sa*sd
  dzdt =  pmde * cd

  dep = ep2 - ep1

  x += dep * dxdt
  y += dep * dydt
  z += dep * dzdt
  
  a = numpy.arctan2(y, x)
  a = numpy.where(a < 0, 2*math.pi+a, a)

  d = numpy.arctan2(z, numpy.hypot(x, y))

  return a, d

# Utility to convert MJD to Julian epoch for above.

def mjd2ep(mjd):
  return(2000.0 + (mjd - 51544.5) / 365.25)

