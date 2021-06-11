import math
import numpy

# RA, Dec to standard coordinates.

def standc(a, d, tpa, tpd):
  sd = numpy.sin(d)
  cd = numpy.cos(d)
  stpd = numpy.sin(tpd)
  ctpd = numpy.cos(tpd)

  c = numpy.cos(a - tpa)

  denom = stpd * sd + ctpd * cd * c

  xi = numpy.sin(a - tpa) * cd / denom
  xn = (ctpd * sd - stpd * cd * c) / denom

  return xi, xn

# Standard coordinates to RA, Dec.

def xixn(xi, xn, tpa, tpd, wrap=True):
  stpd = numpy.sin(tpd)
  ctpd = numpy.cos(tpd)

  denom = ctpd - xn * stpd

  aa = numpy.arctan2(xi, denom)
  if wrap:
    a = numpy.fmod(aa + tpa, 2*math.pi)
    a = numpy.where(a < 0, 2*math.pi+a, a)
  else:
    a = aa + tpa
    
  d = numpy.arctan2(stpd + xn * ctpd, numpy.hypot(xi, denom))

  return a, d

