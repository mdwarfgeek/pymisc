import re
import string

import numpy
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval
from scipy.interpolate import splev

# UNFINISHED!  Need to test "LINEAR" as well as "MULTISPEC",
# units, and implement/test the rest of the non-linear dispersions.
# But it seems to work well enough to use TRES data.

def multispec_lambda(hdr, nord, nwave):
  # Output array.
  wave = numpy.empty([ nord, nwave ], dtype=numpy.double)
  wave.fill(numpy.nan)

  # Pixel coordinate array.
  x = 1 + numpy.arange(nwave, dtype=numpy.double)  # everything is 1-indexed

  # Read CTYPE1 to decide how to handle the file.
  if "CTYPE1" in hdr:
    ctype1 = hdr["CTYPE1"].strip().upper()
  else:
    ctype1 = "LINEAR"

  if ctype1 != "MULTISPE":
    l1 = 0.0
    p1 = 1.0
    dl = 1.0
    dc = 0

    if "CRVAL1" in hdr:
      l1 = float(hdr["CRVAL1"])

    if "CRPIX1" in hdr:
      p1 = float(hdr["CRPIX1"])

    if "CD1_1" in hdr:
      dl = float(hdr["CD1_1"])
    elif "CDELT1" in hdr:
      dl = float(hdr["CDELT1"])

    if "DC-FLAG" in hdr:
      dc = int(hdr["DC-FLAG"])

    tmpwave = (x - p1) * dl + l1
    if dc:
      tmpwave = numpy.power(10.0, tmpwave)

    for iord in range(nord):
      wave[iord,:] = tmpwave

  else:
    # Concatenate WAT2_* keywords into one string.
    iwat = 1
    watstr = ""
  
    while True:
      watkey = "WAT2_{0:03d}".format(iwat)
      if watkey in hdr:
        watstr += hdr[watkey]
      else:
        break
  
      iwat += 1
  
    # Logical to physical coordinate transformation.
    if "LTV1" in hdr:
      ltv1 = float(hdr["LTV1"])
    else:
      ltv1 = 0.0
  
    if "LTM1_1" in hdr:
      ltm11 = float(hdr["LTM1_1"])
    else:
      ltm11 = 1.0
  
    xphys = (x - ltv1) / ltm11
  
    # Extract key=value pairs from string.  This is quite simplistic
    # but should be sufficient, presuming the IRAF "" strings don't
    # need support for escaping.
    iter = re.finditer(r"\s*([^\s=]+)\s*=\s*(\"[^\"]+\"|\S+)", watstr)
  
    # We'll use this to locate the keywords we're interested in.
    specn = re.compile(r"^spec(\d+)$")
  
    for m in iter:
      key, value = m.groups()
  
      # Strip off any remaining quotes.
      value = value.strip("\"")
  
      # Is key a "specn"?
      nm = specn.match(key)
      if nm:
        # Order number.
        iord, = nm.groups()
        iord = int(iord) - 1  # we want to start at 0
  
        if iord >= 0 and iord < nord:
          # Split value into fields.
          ll = value.split()
  
          # Fields are:
          # ap beam dtype lambda_1 delta_lambda n_lambda z aplow aphigh [func]
          # "beam" is the actual grating order number for echelle spectra
          # dtype has the same meaning as DC-FLAG
          beam   = int(ll[1])
          dtype  = int(ll[2])
          l1     = float(ll[3])
          dl     = float(ll[4])
          nl     = int(ll[5])
          z      = float(ll[6])
  
          tmpwave = numpy.zeros([nl])

          if dtype == 0:  # linear
            tmpwave = l1 + dl * (xphys[0:nl]-1)
          elif dtype == 1:  # log
            tmpwave = numpy.power(10.0, l1 + dl * (xphys[0:nl]-1))
          elif dtype == 2:  # nonlinear
            # Read functions.
            ioff = 9
            
            while ioff < len(ll):
              wti   = float(ll[ioff])
              dli   = float(ll[ioff+1])
              ftype = int(ll[ioff+2])
              ioff += 3
              
              if ftype == 1 or ftype == 2:  # cheby, legendre
                ncoeff = int(ll[ioff])
                pmin = float(ll[ioff+1])
                pmax = float(ll[ioff+2])
                ioff += 3
                
                coeff = list(map(float, ll[ioff:ioff+ncoeff]))
                ioff += ncoeff
                
                n = (2*xphys - pmax - pmin) / (pmax - pmin)

                if ftype == 1:
                  tmpwave += wti*(dli+chebval(n, coeff))
                elif ftype == 2:
                  tmpwave += wti*(dli+legval(n, coeff))
              elif ftype == 3 or ftype == 4:  # spline
                if ftype == 3:  # spline1
                  degree = 1
                elif ftype == 4:  # spline3
                  degree = 3

                npc = int(ll[ioff])
                pmin = float(ll[ioff+1])
                pmax = float(ll[ioff+2])
                ioff += 3
  
                ncoeff = npc+degree

                knot = numpy.arange(ncoeff)
                coeff = list(map(float, ll[ioff:ioff+ncoeff]))
                ioff += ncoeff
  
                s = npc * (xphys[0:nl] - pmin) / (pmax - pmin)

                tck = knot, coeff, degree

                tmpwave += wti*(dli+splev(s, tck))
              elif ftype == 5:  # pixel array
                ncoords = int(ll[ioff])
                ioff += 1
  
                tmpwave += wti*(dli+list(map(float, ll[ioff:ioff+ncoords])))
                ioff += ncoords
              elif ftype == 6:  # sampled array
                npairs = int(ll[ioff])
                ioff += 2
  
                xknot = list(map(float, ll[ioff:ioff+2*npairs:2]))
                yknot = list(map(float, ll[ioff+1:ioff+2*npairs:2]))
                ioff += 2*npairs
  
                tmpwave += wti*(dli+numpy.interp(xphys[0:nl], xknot, yknot))
  
          wave[iord,:] = tmpwave / (1.0 + z)
#  else:
#    raise RuntimeError("multispec_lambda: unrecognised CTYPE1: " + ctype1)

  return wave
  
