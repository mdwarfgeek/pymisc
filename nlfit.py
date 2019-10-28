import numpy
import scipy.optimize

def nlfit(func, parminit, fixed, y, e_y, escale=True, flag=None):
  """nlfit - non-linear least-squares with tied parameters.

Basic usage:

  import nlfit

  def func(trial, y):
    y[:] = some f(trial)

  parm, e_parm, chisq, ndof = nlfit(func, parminit, fixed, y, e_y)

This routine is a simple wrapper for scipy.optimize.leastsq
implementing support for tied parameters by rewriting the parameter
vector to remove the tied parameters before passing into the standard
routine and patching them back in during the call to the objective
function.  The vector "fixed" is a boolean the same length as "parm"
and "parminit" specifying which parameters are fixed (true) or varied
(false).  "func" takes two arguments, the parameter vector as a numpy
array, and an output vector to take the function values at each data
point.

It's a bit of a kludge and lacks useful support for several of the
usual refinements such as propagating through additional function
arguments.

"""

  # Rewrite parameter vector.
  pmap = []

  for iparm in range(parminit.size):
    if fixed is None or not fixed[iparm]:
      pmap.append(iparm)

  pmap = numpy.array(pmap, dtype=numpy.int)

  # How many parameters are being varied?
  nvary = len(pmap)

  if nvary > y.size:
    raise RuntimeError("nlfit: more parameters than data points")

  # Repack into new vector for fit.
  pinit = parminit[pmap]

  # Define wrapper to convert calling conventions.
  rwt = 1.0 / e_y

  if flag is not None:
    rwt[numpy.logical_not(flag)] = 0.0
    ndp = numpy.sum(flag)
  else:
    ndp = y.size

  def wrap(p):
    trial = numpy.copy(parminit)
    trial[pmap] = p

    mod = numpy.empty_like(y)

    func(trial, mod)

    f = rwt * (y - mod)

    return f

  # Perform minimization.
  pfit, cov_pfit, infodict, errmsg, ier = scipy.optimize.leastsq(wrap, pinit, full_output=1)

  if ier < 1 or ier > 4:
    raise RuntimeError("leastsq: " + errmsg)

  # chi^2 and ndof.
  chisq = numpy.sum(infodict["fvec"]**2)
  ndof = ndp - pfit.size

  if escale and ndof > 0:
    varscl = chisq / ndof
  else:
    varscl = 1.0

  # Repack into output vectors.
  parm = numpy.copy(parminit)
  e_parm = numpy.zeros_like(parm)

  parm[pmap] = pfit

  if cov_pfit is not None:
    e_parm[pmap] = numpy.sqrt(numpy.diagonal(cov_pfit) * varscl)

  return parm, e_parm, chisq, ndof
