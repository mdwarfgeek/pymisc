# Simplified routines for dealing with HEALPix pixelization.
# Only the nested variant (as used in GAIA) is currently supported.

import numpy

# The argument p should be an array of int64.
def heal2sky(p, nside):
  # Ensure numpy array of int64.
  if isinstance(p, numpy.ndarray):
    p = p.astype(numpy.int64)
  else:
    p = numpy.array(p, dtype=numpy.int64)

  # Gorski et al. (2005), ApJ, 622, 759, Sect. 4.2.
  # The description in the paper is a bit "lean", and some details
  # needed to implement it in practice are missing or unclear.  I've
  # tried to fill them in below.

  # Base resolution pixel number.
  f = p // (nside*nside)

  # Subpixel within base pixel.
  pprime = p % (nside*nside)

  # Location of southernmost corner of base resolution pixel,
  # Eq. (10)-(12).
  nphi = 4

  frow = f // nphi
  f1 = frow + 2
  f2 = 2*(f % nphi) - (frow % 2) + 1

  # Compress bits of subpixel to get x, y, Eq. (13) and (14).  This way
  # of doing it is very inefficient, try to think of a better solution.
  x = 0
  y = 0

  for i in range(32):
    x |= (pprime >> i) & (1 << i)
    y |= (pprime >> (i+1)) & (1 << i)

  # Vertical and horizontal subpixel coordinates, Eq. (15) and (16).
  v = x + y
  h = x - y

  # Ring index, Eq. (17).
  i = f1 * nside - v - 1

  # Selectors for the various cases.
  weq = numpy.logical_and(i >= nside, i <= 3*nside)
  wpp = numpy.logical_not(weq)
  wnp = i < nside
  wsp = i > 3*nside

  s = numpy.empty_like(i)
  s[weq] = (i[weq] - nside + 1) % 2  # equatorial belts
  s[wpp] = 1  # polar caps

  # Longitude index, Eq. (18).
  j = numpy.empty_like(i)
  j[weq] = (f2[weq] * nside + h[weq] + s[weq]) // 2  # eq belts
  j[wnp] = (f2[wnp] * i[wnp] + h[wnp] + s[wnp]) // 2  # N

  # The paper doesn't seem to mention that Eq. (18) is only for the
  # equatorial belts.  The expression for j in the polar caps is
  # different because the number of horizontal subpixels per base
  # pixel is i rather than nside here.
  ip = 4*nside-i
  j[wsp] = (f2[wsp] * ip[wsp] + h[wsp] + s[wsp]) // 2  # S

  z = numpy.empty_like(i, dtype=numpy.double)
  phi = numpy.empty_like(z)

  # Eq. (8)
  z[weq] = 4.0/3.0 - 2*i[weq] / (3.0*nside)  # eq

  # Eq. (9)
  phi[weq] = numpy.pi * (j[weq] - s[weq]/2) / (2.0*nside)  # eq

  # Eq. (4)
  z[wnp] = 1.0 - i[wnp]*i[wnp] / (3.0*nside*nside)  # N

  # Eq. (5)
  phi[wnp] = numpy.pi * (j[wnp] - s[wnp]/2) / (2.0*i[wnp])  # N

  # These equations aren't given in the paper so I had to figure it
  # out from the rather vague "mirror symmetry" remark made in
  # Sect. 4.1.
  z[wsp] = ip[wsp]*ip[wsp] / (3.0*nside*nside) - 1.0  # S
  phi[wsp] = numpy.pi * (j[wsp] - s[wsp]/2) / (2.0*ip[wsp])  # S

  # cos(90-dec) = z = sin(dec)
  ra = phi
  de = numpy.arcsin(z)

  return ra, de
