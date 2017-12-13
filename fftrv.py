import math
import numpy
import sys

from scipy.interpolate import InterpolatedUnivariateSpline

from medsig import *

# Possible future enhancements:
#
# Add continuum stuff, clipping here.
# Add input of Barycentric correction as zb and template redshift.
# Use logp1 to process these properly so numerics are stable.
# Need to make resolution of output more consistent

# For vsini accuracy of peak fitting for RV may be an issue

def fitpeak(x, y, pkfrac):
  # Locate peak.
  imax = numpy.argmax(y)
  xmax = x[imax]
  ymax = y[imax]
  
  nbin = len(y)

  # Fit parabola.
  ythr = pkfrac*ymax
  
  ia = imax-1
  while ia >= 0 and y[ia] > ythr:
    ia = ia-1
    
  ib = imax+1
  while ib < nbin and y[ib] > ythr:
    ib = ib+1

  xpar = x[ia+1:ib] - xmax
  ypar = y[ia+1:ib] - ymax
  
  coef = numpy.polynomial.polynomial.polyfit(xpar, ypar, 2, full=False)

  # Make sure it's a maximum.
  if coef[2] < 0:
    # Also check maximum is in range.
    dx = 0.5*coef[1]/coef[2]

    if dx >= numpy.min(xpar) and dx <= numpy.max(xpar):
      xbest = xmax - dx
      ybest = ymax + coef[0] - 0.5*coef[1]*dx
    else:
      xbest = None
      ybest = None
  else:
    xbest = None
    ybest = None

  if xbest is None:
    # Fall back to something simpler.
    print >>sys.stderr, "fitpeak: failed to find maximum in range"
    xbest, ybest = parint(x, y, imax)

  return xbest, ybest

def parint(x, y, ipk):
  xpk = x[ipk]
  ypk = y[ipk]

  dxa = x[ipk-1] - xpk
  dxb = x[ipk+1] - xpk
  dya = y[ipk-1] - ypk
  dyb = y[ipk+1] - ypk

  # Check there's a maximum.
  if dya < 0 and dyb < 0:
    dxasq = dxa*dxa
    dxbsq = dxb*dxb

    denom = dxb * dxasq - dxa * dxbsq

    b = (dyb * dxasq - dya * dxbsq) / denom
    c = (dya * dxb - dyb * dxa) / denom

    xbest = xpk - 0.5*b/c
    ybest = ypk - 0.25*b*b/c
  else:
    print >>sys.stderr, "parint: failed"
    xbest = xpk
    ybest = ypk

  return xbest, ybest

def rfftpower(vec):
  s = 2*numpy.sum(numpy.square(numpy.absolute(vec[1:])))
  s += numpy.square(numpy.absolute(vec[0]))

  return s

def resample(lw, wave, flux, k):

  log_wave = numpy.log(wave)

  spl = InterpolatedUnivariateSpline(log_wave, flux, k=k)

  return spl(lw)

def rotate(v, hbin, nbin, pbin):
  ret = numpy.zeros([pbin])

  ret[0:hbin] = v[hbin:nbin]
  ret[pbin-hbin:pbin] = v[0:hbin]

  return ret

def unrotate(v, hbin, nbin, pbin):
  ret = numpy.empty([nbin])

  ret[0:hbin] = v[pbin-hbin:pbin]
  ret[hbin:nbin] = v[0:hbin]

  return ret

def rotbroad(lw, lwsamp, hbin, nbin, pbin, zbroad, u1, u2):
  # Extent of kernel in log wavelength.
  lwmin = math.log1p(-zbroad)
  lwmax = math.log1p(zbroad)

  # Convert to samples.
  rimin = lwmin / lwsamp
  rimax = lwmax / lwsamp

  imin = int(rimin)
  imax = int(rimax)

  if imin < -hbin:
    imin = -hbin
  if imax >= hbin:
    imax = hbin-1

  if imax > imin:
    # Generate kernel.
    i = numpy.arange(imin, imax+1)
    ioff = -imin  # offset in i array
    
    z = numpy.expm1(i * lwsamp)
    
    y = 1.0 - (z / zbroad)**2
    
    norm = 1.0 / (math.pi * rimax * (1.0 - u1/3.0 - u2/6.0))
    a = 2.0 * (1.0 - u1 - u2) * norm
    b = 0.5*math.pi * (u1 + 2*u2) * norm
    c = 4.0 * u2 * norm / 3.0
    
    K = (a - c*y) * numpy.sqrt(y) + b*y

    # Format into full-size array with zero padding.
    kpad = numpy.zeros([pbin])
    kpad[0:imax+1] = K[ioff:ioff+imax+1]
    kpad[pbin+imin:pbin] = K[0:ioff]

    return kpad
  else:
    return None

def tukey(a, b, c, d, nbin):
  filt = numpy.ones(nbin, dtype=numpy.double)

  if b > 0:
    n = b - a
    x = n - numpy.arange(0, n)
    w = numpy.cos(0.5*math.pi*x/n)
    
    filt[0:a] = 0
    filt[a:b] = w

  if d > 0:
    n = d - c
    x = numpy.arange(0, n)
    w = numpy.cos(0.5*math.pi*x/n)
    
    filt[c:d] = w
    filt[d:nbin] = 0

  return filt

# How to set parameters:
# top_nrun / nrun upper filter boundary should be roughly at the
# spectral resolution: nrun = nbin * output resol / spec resol

class fftrv:
  """fftrv - FFT-based cross-correlation for radial velocities

Basic usage:

  import fftrv

  frv = fftrv.fftrv()

  z, h, zbest, hbest, sigt = frv.correlate(tmpl_wave, tmpl_flux,
                                           wave, flux)

This class implements a reasonably fast FFT-based cross correlation
for determining redshifts from spectra.  The parameters and many of
the methods used are very similar to the Kurtz & Mink (1998) package
and have been given the same names where possible.

The package has been used for reconnaissance grade radial velocities,
vsini estimation, and solution of SB2s using TODCOR.  It is not
intended for high precision applications.  For precise radial
velocities of single-lined objects I use it to supply the initial
guesses for a direct (in real space) correlation or least-squares
forward modeling (e.g. Anglada-Escude & Butler 2012) method.  These
aren't as polished or finished so aren't included, but may be released
some day.

Setup is stored inside an "fftrv" object which can be reused for an
arbitrary number of correlations with the same parameters.

The method "correlate" to compute correlations takes arrays of
wavelength and counts for the template and target spectra, and
optionally can also apply a rotational broadening kernel to template,
target or both.  To use rotational broadening, specify zbroad to
broaden the template and zbroadt to broaden the target spectrum (the
latter is intended for use inside TODCOR and is probably not needed
elsewhere).  Quadratic limb darkening coefficients for the rotational
broadening kernel are given in u1 and u2.  The quantity "zbroad" is
the rotational broadening specified as redshift (vsini divided by the
speed of light).

The spectra should already be continuum subtracted (or normalized) as
desired.  Barycentric corrections (if needed) should also be performed
externally.

The "correlate" method returns:

  z      Array of redshift values
  h      Array of normalized [0,1] correlation (same length as z)
  zbest  Redshift at peak correlation
  hbest  Peak correlation value
  sigt   Root mean square of prepared template spectrum.  This
         quantity is returned to the user because it is needed to
         implement TODCOR.

How I use it:

For TRES (resolving power approximately 44,000 fiber echelle) I
analyse the orders one at a time using the original, unnormalized
spectra in counts, and subtract a robust (clipped) Legendre polynomial
fit of degree 4 to remove the continuum.  Parameters to the "fftrv"
constructor are the defaults except I oversample the correlation by
setting nbin = 32 * npix where npix is the number of pixels per order
(for TRES, npix = 2304).  This seems to improve the velocity precision
somewhat.  For some purposes it can also help to set pkfit=0 to turn
off peak fitting.  Template spectrum is a high s/n observation of a
suitable velocity standard (usually Gl 699, Barnard's star; see
Nidever et al. 2002 for a set of suitable M-dwarf velocity standards).

"""

  def __init__(self,
               apodize=0.05,
               kinterp=3,
               nbin=8192,
               low_bin=5,
               top_low=20,
               top_nrun=250,
               nrun=500,
               zeropad=0,
               zmax=None,
               t_emchop=1,
               s_emchop=1,
               t_em_reject=5.0,
               s_em_reject=5.0,
               pkfit=1,
               pkfrac=0.5):

    # Store parameters.
    self.apodize = apodize
    self.kinterp = kinterp
    self.hbin = nbin//2
    self.nbin = self.hbin*2
    self.zeropad = zeropad
    self.zmax = zmax
    self.t_emchop = t_emchop
    self.s_emchop = s_emchop
    self.t_em_reject = t_em_reject
    self.s_em_reject = s_em_reject
    self.pkfit = pkfit
    self.pkfrac = pkfrac

    # Make bandpass filter.
    if zeropad:
      filt = tukey(2*low_bin-1, 2*top_low-1, 2*top_nrun-1, 2*nrun-1, nbin+1)
    else:
      filt = tukey(low_bin-1, top_low-1, top_nrun-1, nrun-1, self.hbin+1)

    self.rtfilt = numpy.sqrt(filt)

  def correlate(self,
                tmpl_wave, tmpl_flux,
                wave, flux,
                zbroad=0, u1=0, u2=0, zbroadt=0, # rotation
                lwmin=None, lwmax=None):  # user specified sampling

    # Decide wavelength sampling.
    wavemin = max(tmpl_wave[0], wave[0])
    wavemax = min(tmpl_wave[-1], wave[-1])

    lwmin = math.log(wavemin)
    lwmax = math.log(wavemax)

    lwsamp = (lwmax-lwmin) / (self.nbin-1)

    # Wavelength array.
    lw = lwmin + lwsamp * numpy.arange(self.nbin)

    # Chop emission lines.
    if self.t_emchop:
      medflux, sigflux = medsig(tmpl_flux)
      tmpl_msk = tmpl_flux < medflux + self.t_em_reject*sigflux
    else:
      tmpl_msk = numpy.ones_like(tmpl_flux, dtype=numpy.bool)

    if self.s_emchop:
      medflux, sigflux = medsig(flux)
      msk = flux < medflux + self.s_em_reject*sigflux
    else:
      msk = numpy.ones_like(flux, dtype=numpy.bool)

    # Resample.
    tmpl_bin = resample(lw,
                        tmpl_wave[tmpl_msk], tmpl_flux[tmpl_msk],
                        self.kinterp)
    targ_bin = resample(lw, wave[msk], flux[msk], self.kinterp)

    if self.zeropad and self.zmax is not None:
      # When zero-padding, restrict target spectrum so there's always
      # valid template spectrum to correlate with over the full desired
      # range of redshift.
      dlwclip = numpy.log1p(self.zmax)
      iclip = numpy.ceil(dlwclip / lwsamp)

      targ_bin[0:iclip] = 0
      targ_bin[self.nbin-iclip:self.nbin] = 0

    # Apodization window.
    kapod = int(self.apodize * self.nbin)
    apfilt = tukey(0, kapod, self.nbin-1-kapod, self.nbin-1, self.nbin)

    # Padded length for FFTs.
    if self.zeropad:
      pbin = 2*self.nbin
    else:
      pbin = self.nbin

    # Rotate so result comes out in the right place.
    tmpl_prep = rotate(tmpl_bin * apfilt, self.hbin, self.nbin, pbin)
    targ_prep = rotate(targ_bin * apfilt, self.hbin, self.nbin, pbin)

    # FFT.
    tmpl_ft = numpy.fft.rfft(tmpl_prep)
    targ_ft = numpy.fft.rfft(targ_prep)

    # Rotational broadening, if requested.
    if zbroad > 0:
      broad = rotbroad(lw, lwsamp, self.hbin, self.nbin, pbin, zbroad, u1, u2)

      if zbroad is not None:
        broad_ft = numpy.fft.rfft(broad)
        tmpl_ft *= broad_ft

    if zbroadt > 0:
      broad = rotbroad(lw, lwsamp, self.hbin, self.nbin, pbin, zbroadt, u1, u2)

      if zbroadt is not None:
        broad_ft = numpy.fft.rfft(broad)
        targ_ft *= broad_ft

    # Filter.
    tmpl_filt = tmpl_ft * self.rtfilt
    targ_filt = targ_ft * self.rtfilt

    # FFT of correlation function.
    corr_ft = numpy.conj(tmpl_filt) * targ_filt

    # Normalization.
    ssq_tmpl = rfftpower(tmpl_filt)
    ssq_targ = rfftpower(targ_filt)

    norm = pbin / numpy.sqrt(ssq_tmpl * ssq_targ)

    # Inverse FFT to get correlation function.
    corr_inv = numpy.fft.irfft(corr_ft * norm)

    # Rotate.
    corr = unrotate(corr_inv, self.hbin, self.nbin, pbin)

    # Redshift.
    dlw = lw - lw[self.hbin]
    z = numpy.expm1(dlw)

    # Locate peak.
    if self.pkfit:
      zbest, rbest = fitpeak(z, corr, self.pkfrac)
    else:
      zbest, rbest = parint(z, corr, numpy.argmax(corr))

    return z, corr, zbest, rbest, numpy.sqrt(ssq_tmpl/pbin)
