import math
import numpy
import scipy.stats

# For each source in "com" search for best source in "ref" within searchrad.
# The "ref" arrays must be sorted, on x if sorty=False or y if sorty=True.
# To process ra/dec lists, transform to standard coordinates xi, xn using
# "standc" prior to sorting and matching (including applying proper motion
# to epoch of "com" if necessary).
# Non-matches are indicated by -1 in the returned lists of indices.

def lrmatch(comx, comy, commag, comerr,
            refx, refy, refmag, referr,
            searchrad, order):
  # log of cumulative distribution of magnitude evaluted for each input,
  # which is the same as (log of) rank.  We use rankdata to ensure ties
  # are treated correctly.  Only want to use one of these.
  if commag is not None and refmag is not None:
    raise RuntimeError("only need one set of magnitudes")

  if commag is None:
    commaglogrank = numpy.zeros_like(comx)
  else:
    commaglogrank = scipy.stats.rankdata(commag)

  if refmag is None:
    refmaglogrank = numpy.zeros_like(refx)
  else:
    refmaglogrank = scipy.stats.rankdata(refmag)

  # Substitute for errors if not given.
  if referr is None:
    referr = numpy.zeros_like(refx)
    if comerr is None:
      comerr = numpy.ones_like(comx)
  else:
    if comerr is None:
      comerr = numpy.zeros_like(comx)

  # For each comparison object, search for ref objects.
  ncom = len(comx)

  best_ref_for_com = numpy.empty_like(comx, dtype=numpy.int)
  best_ref_for_com.fill(-1)

  best_com_for_ref = numpy.empty_like(refx, dtype=numpy.int)
  best_com_for_ref.fill(-1)

  best_lr_for_ref = numpy.empty_like(refx, dtype=numpy.double)

  for comrow in range(ncom):
    if order.lower() == "x":
      refrows, sep = bserchmult(comx[comrow], comy[comrow],
                                refx, refy,
                                searchrad)
    elif order.lower() == "y":
      refrows, sep = bserchmult(comy[comrow], comx[comrow],
                                refy, refx,
                                searchrad)
    else:
      raise RuntimeError("unknown order", order)

    if len(refrows) > 0:
      # Likelihood ratios.
      rnorm = sep / numpy.hypot(comerr[comrow], referr[refrows])
      lr = -0.5*rnorm**2 - commaglogrank[comrow] - refmaglogrank[refrows]
      
      ibest = numpy.argmax(lr)

      refrow = refrows[ibest]
      best_lr = lr[ibest]

      previdx = best_com_for_ref[refrow]
      if previdx >= 0 and best_lr_for_ref[refrow] > best_lr:
        # Keep.
        pass
      else:
        # Replace.
        best_com_for_ref[refrow] = comrow
        best_lr_for_ref[refrow] = best_lr

        if previdx >= 0:
          best_ref_for_com[previdx] = -1

        best_ref_for_com[comrow] = refrow

  return best_ref_for_com, best_com_for_ref

# Binary chop for array sorted on "a".

def bserchmult(coma, comb, refa, refb, errlim):
  isp, ifp = numpy.searchsorted(refa, [ coma - errlim, coma + errlim ])

  errsq = numpy.square(errlim)

  poserrsq = numpy.square(coma - refa[isp:ifp]) + numpy.square(comb - refb[isp:ifp])

  ww = poserrsq < errsq
  iref = isp + numpy.nonzero(ww)[0]
  sep = numpy.sqrt(poserrsq[ww])
  
  return iref, sep

def lrmatch1d(comx, commag, comerr,
              refx, refmag, referr,
              searchrad):
  # log of cumulative distribution of magnitude evaluted for each input,
  # which is the same as (log of) rank.  We use rankdata to ensure ties
  # are treated correctly.  Only want to use one of these.
  if commag is not None and refmag is not None:
    raise RuntimeError("only need one set of magnitudes")

  if commag is None:
    commaglogrank = numpy.zeros_like(comx)
  else:
    commaglogrank = scipy.stats.rankdata(commag)

  if refmag is None:
    refmaglogrank = numpy.zeros_like(refx)
  else:
    refmaglogrank = scipy.stats.rankdata(refmag)

  # Substitute for errors if not given.
  if referr is None:
    referr = numpy.zeros_like(refx)
    if comerr is None:
      comerr = numpy.ones_like(comx)
  else:
    if comerr is None:
      comerr = numpy.zeros_like(comx)

  # For each comparison object, search for ref objects.
  ncom = len(comx)

  best_ref_for_com = numpy.empty_like(comx, dtype=numpy.int)
  best_ref_for_com.fill(-1)

  best_com_for_ref = numpy.empty_like(refx, dtype=numpy.int)
  best_com_for_ref.fill(-1)

  best_lr_for_ref = numpy.empty_like(refx, dtype=numpy.double)

  for comrow in range(ncom):
    isp, ifp = numpy.searchsorted(refx, [ comx[comrow] - searchrad, comx[comrow] + searchrad ])

    refrows = numpy.arange(isp, ifp)
    sep = numpy.absolute(refx[isp:ifp] - comx[comrow])

    if len(refrows) > 0:
      # Likelihood ratios.
      rnorm = sep / numpy.hypot(comerr[comrow], referr[refrows])
      lr = -0.5*rnorm**2 - commaglogrank[comrow] - refmaglogrank[refrows]

      ibest = numpy.argmax(lr)

      refrow = refrows[ibest]
      best_lr = lr[ibest]

      previdx = best_com_for_ref[refrow]
      if previdx >= 0 and best_lr_for_ref[refrow] > best_lr:
        # Keep.
        pass
      else:
        # Replace.
        best_com_for_ref[refrow] = comrow
        best_lr_for_ref[refrow] = best_lr

        if previdx >= 0:
          best_ref_for_com[previdx] = -1

        best_ref_for_com[comrow] = refrow

  return best_ref_for_com, best_com_for_ref

