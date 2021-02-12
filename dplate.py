import lfa
import math
import numpy

def clipdplate(comx, comy, refx, refy, w=None, niter=10, nsig=5):
  if w == None:
    wtmp = numpy.ones_like(refy)
  else:
    wtmp = numpy.tile(w, y.shape)

  npt = wtmp.size

  averr = 0

  for i in range(niter):
    coef = lfa.dplate(comx, comy, refx, refy, wt=wtmp)

    modx = coef[0]*comx + coef[1]*comy + coef[2]
    mody = coef[3]*comy + coef[4]*comx + coef[5]

    resx = modx-refx
    resy = mody-refy

    resid = numpy.sqrt(resx*resx + resy*resy)

    averr = 1.48 * numpy.median(resid)

    flag = resid < nsig*averr

    # Did it change?  Stop if not.
    nptnew = numpy.sum(flag)
    if nptnew == npt:
      break

    wtmp *= flag
    npt = nptnew

  return coef, averr
