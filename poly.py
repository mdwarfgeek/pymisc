import numpy
from medsig import *

def clippolyfit(x, y, deg, w=None, niter=10, nsig=5):
  if w is None:
    wtmp = numpy.ones_like(y)
  else:
    wtmp = numpy.copy(w)
#    wtmp = numpy.tile(w, y.shape)
    
  npt = wtmp.size

  for i in range(niter):
    coef = numpy.polynomial.polynomial.polyfit(x, y, deg, full=False, w=wtmp)
    yfit = numpy.polynomial.polynomial.polyval(x, coef)
    resid = y - yfit
    (medoff, sigoff) = medsig(resid)
    flag = numpy.absolute(resid-medoff) < nsig*sigoff

    # Did it change?  Stop if not.
    nptnew = numpy.sum(flag)
    if nptnew == npt:
      break

    wtmp *= flag
    npt = nptnew

  return coef

def cliplegfit(x, y, deg, w=None, niter=10, nsig=5):
  if w is None:
    wtmp = numpy.ones_like(y)
  else:
    wtmp = numpy.copy(w)
#    wtmp = numpy.tile(w, y.shape)

  npt = wtmp.size

  for i in range(niter):
    coef = numpy.polynomial.legendre.legfit(x, y, deg, full=False, w=wtmp)

    yfit = numpy.polynomial.legendre.legval(x, coef)
    resid = y - yfit
    (medoff, sigoff) = medsig(resid)
    flag = numpy.absolute(resid-medoff) < nsig*sigoff

    # Did it change?  Stop if not.
    nptnew = numpy.sum(flag)
    if nptnew == npt:
      break

    wtmp *= flag
    npt = nptnew

  return coef

def leg2poly_matrix(degree, xscl=1.0, xoff=0.0):
  ncoef = degree+1

  A = numpy.zeros([ ncoef, ncoef ])
  tmp = numpy.zeros([ ncoef ])
  
  A[0,0] = 1.0
  A[0,1] = xoff
  A[1,1] = xscl
  
  for n in range(1, degree):
    # x Pn vector: shift right is equivalent to multiplying by x.
    tmp[0] = 0
    tmp[1:ncoef] = xscl * A[0:ncoef-1,n]

    tmp += xoff * A[:,n]
    
    A[:,n+1] = ((2*n + 1) * tmp - n * A[:,n-1]) / (n + 1)

  return A
