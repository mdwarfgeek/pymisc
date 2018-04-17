import math
import sys
import warnings

# Parameters for double-exponential model in Benedict et al. (2016).
_x0 = 0.076
_y0 = -11.41
_a1 = 1.64
_a2 = 19.81
_k1 = 1.0 / 0.05  # reciprocal of tau
_k2 = 1.0 / 3.10

# Parameters for iteration.
_maxiter = 100
_prec = 1.0e-14

def mk2mass_ben(mk):
  """mk2mass_ben - M_K to mass, Benedict et al. (2016) double exponential.

Usage:

  mass = mk2mass_ben(mk)

Where mk is absolute 2MASS K magnitude and mass is in solar masses.

This version inverts the double-exponential "forward model" (for going
from mass to absolute magnitude) which avoids some of the undesirable
behaviour of the polynomial given in the paper.  It is the same as the
current function in use inside the MEarth databases for estimating
masses.

NOTE: the range of the parameters is not checked to ensure the
relation is used within the domain of applicability, this is left to
the user.

References:

Benedict et al. (2016) AJ 152 141

"""

  # Redefine problem in terms of more convenient variables x, y.
  # x = m - x0 and y = mk - y0.
  y = mk - _y0

  # Initial guess, not really important because function is well behaved.
  # This uses the series expansion for the exponentials taken to terms
  # linear in x, which is equivalent to doing the first iteration of
  # Newton's method starting at x=0.
  x = (_a1 + _a2 - y) / (_k1*_a1 + _k2*_a2)
  
  # Newton's method to refine.
  i = 0

  while i < _maxiter:
    # Trap overflow of exponentials for very negative x.
    try:
      e1 = _a1 * math.exp(-_k1*x)
      e2 = _a2 * math.exp(-_k2*x)
    except OverflowError:
      # Overflow is equivalent to "negative mass" so return zero.
      return 0

    f  = e1 + e2 - y
    df = -(_k1*e1 + _k2*e2)
    
    if df == 0:
      # This happens for large m only (when we run well off the top
      # of the mass range so the exponentials underflow).  Return
      # undefined value to flag problem.
      return None

    delta = f / df
    
    x -= delta

    if abs(delta) < _prec:
      break

    i += 1
  
  if i >= _maxiter:
    warnings.warn("mk2mass_ben: iteration limit reached")

  m = x + _x0
  # Catch "negative mass" and clip to zero.
  if m <= 0:
    m = 0

  return m

def mk2mass_ben_poly(mk):
  """mk2mass_ben_poly - M_K to mass, Benedict et al. (2016) polynomial.

Usage:

  mass = mk2mass_ben_poly(mk)

Where mk is absolute 2MASS K magnitude and mass is in solar masses.

This version is the original polynomial "reverse model" (absolute
magnitude to mass) from the paper, for comparison purposes.

NOTE: the range of the parameters is not checked to ensure the
relation is used within the domain of applicability, this is left to
the user.

References:

Benedict et al. (2016) AJ 152 141

"""

  arg = mk - 7.5
  mass = (((-0.0032*arg + 0.0038) * arg + 0.0400) * arg - 0.1352) * arg + 0.2311
  return mass

def mass2mk_ben(m):
  """mass2mk_ben - mass to M_K, Benedict et al. (2016) double exponential.

Usage:

  mk = mass2mk_ben(mass)

Where mk is absolute 2MASS K magnitude and mass is in solar masses.

This version is the original double-exponential "forward model" (for
going from mass to absolute magnitude) from the paper.

NOTE: the range of the parameters is not checked to ensure the
relation is used within the domain of applicability, this is left to
the user.

References:

Benedict et al. (2016) AJ 152 141

"""

  x  = m - _x0
  e1 = _a1 * math.exp(-_k1*x)
  e2 = _a2 * math.exp(-_k2*x)

  mk = e1 + e2 + _y0

  return mk

def mk2mass_del(mk):
  """mk2mass_del - M_K to mass, Delfosse et al. (2000).

Usage:

  mass = mk2mass_del(mk)

Where mk is absolute CIT K magnitude and mass is in solar masses.

This version is the original polynomial from the paper.

NOTE: the range of the parameters is not checked to ensure the
relation is used within the domain of applicability, this is left to
the user.

References:

Delfosse et al. (2000) A&A 364 217

"""

  val = (((0.37529*mk - 6.2315) * mk + 13.205) * mk + 6.12) * mk + 1.8
  return 10.0**(1.0e-3 * val)

def mk2mass_del_mod(mk):
  """mk2mass_del_mod - M_K to mass, modified Delfosse et al. (2000).

Usage:

  mass = mk2mass_del_mod(mk)

Where mk is absolute CIT K magnitude and mass is in solar masses.

This version is the modified relation used in the MEarth databases,
data releases and publications prior to 2017 December 8.  The
modifications are to hold the slope fixed outside the nominal fitting
domain of mk in [4.5,9.5] in order to improve extrapolation behaviour
when the relation is used out of range.

NOTE: the range of the parameters is not checked to ensure the
relation is used within the domain of applicability, this is left to
the user.

References:

Delfosse et al. (2000) A&A 364 217

"""

  if mk < 4.5:
    arg = 4.5
    inrange = 0
  elif mk > 9.5:
    arg = 9.5
    inrange = 0
  else:
    arg = mk
    inrange = 1

  val = (((0.37529*arg - 6.2315) * arg + 13.205) * arg + 6.12) * arg + 1.8

  if not inrange:
    slope = ((0.37529*4*arg - 6.2315*3) * arg + 13.205*2) * arg + 6.12
    val += slope * (mk-arg)

  return 10.0**(1.0e-3 * val)

