import numpy

from fpcoord import *

def radeclimits(ra, dec, hwidth):
  # Evaluate coordinates at the 4 corners and the 4 centres of each side.
  xi = numpy.array([ -hwidth, -hwidth,  hwidth, hwidth,
                         0.0,     0.0, -hwidth, hwidth ])
  xn = numpy.array([ -hwidth,  hwidth, -hwidth, hwidth,
                     -hwidth,  hwidth,     0.0,    0.0 ])

  ralim, declim = xixn(xi, xn, ra, dec, wrap=False)
  
  # Handle special cases.
  pio2 = numpy.pi/2
  twopi = numpy.pi*2
  
  if dec + hwidth > pio2 and dec - hwidth < -pio2:
    # Contains both poles, RA and Dec range of all.
    ra_low  = 0.0
    ra_high = twopi

    dec_low  = -pio2
    dec_high = pio2

  elif dec + hwidth > pio2:
    # Contains the North pole, RA range of all.
    ra_low  = 0.0
    ra_high = twopi

    dec_low  = numpy.min(declim)
    dec_high = pio2

  elif dec - hwidth < -pio2:
    # Contains the South pole, RA range of all.
    ra_low  = 0.0
    ra_high = twopi

    dec_low  = -pio2
    dec_high = numpy.max(declim)

  else:
    ra_low   = numpy.min(ralim)
    ra_high  = numpy.max(ralim)
    dec_low  = numpy.min(declim)
    dec_high = numpy.max(declim)

  # Handle RA wrap.
  if ra_low < 0:
    o_ra_low  = twopi + ra_low
    o_ra_high = twopi
    
    ra_low  = 0

    l_ra_low = numpy.array([ra_low, o_ra_low])
    l_ra_high = numpy.array([ra_high, o_ra_high])
    
  elif ra_high > twopi:
    o_ra_low  = ra_low
    o_ra_high = twopi
      
    ra_low  = 0
    ra_high = ra_high - twopi
      
    l_ra_low = numpy.array([ra_low, o_ra_low])
    l_ra_high = numpy.array([ra_high, o_ra_high])

  else:
    l_ra_low = numpy.array([ra_low])
    l_ra_high = numpy.array([ra_high])
    
  return l_ra_low, l_ra_high, dec_low, dec_high
