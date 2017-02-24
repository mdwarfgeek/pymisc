import lfa

def bary(obs, src, jdutc):
  """bary - Barycentric corrections.

Basic usage:

  import lfa
  import bary

  obs = lfa.observer(longitude, latitude, height)
  src = lfa.source(ra, de, pmra, pmde, plx, vrad, epoch)

  bjdtdb, zb = bary.bary(obs, src, jdutc)

This routine computes Barycentric Julian Date (BJD) in the TDB
time-system, and the corresponding redshift correction zb.  The latter
quantity can be applied directly to the observed wavelengths to adjust
them to what would be observed at the solar system barycenter by
multiplying by (1+zb), which is how I usually do it, or converted to
the more conventional form expressed as a velocity ("BCV") by
multiplying zb by the speed of light.

Inputs are "observer" and "source" structures describing the location
of the observer on the Earth and the source on the sky, and the time
of observation (which should be the midpoint) as a Julian Date in the
UTC time-system.  Units for longitude and latitude are radians, and
height (above the geoid) is in metres.  The "source" constructor
supports ICRS coordinates only, and takes right ascension and
declination in radians, proper motions are sky projected and are in
arcsec/yr, parallax in arcsec, and radial velocity in km/s.  Any of
the latter four quantities can be given as zero if not known to
disable these corrections.  The last argument is the Julian epoch at
which the coordinates (ra, de) were measured, usually 2000.0.

References:

Eastman et al. (2010) PASP 122 935
Lindegren & Dravins (2003) A&A 401 1185
Wright & Eastman (2014) PASP 126 838

"""

  # Convert to MJD as used internally.
  mjdutc = jdutc - lfa.ZMJD

  # Figure out TT-UTC from given UTC.
  iutc = int(mjdutc)
  ttmutc = obs.dtai(iutc, mjdutc-iutc) + lfa.DTT

  # Compute time-dependent quantities.
  obs.update(mjdutc, ttmutc, lfa.OBSERVER_UPDATE_ALL)

  # Figure out total clock correction TDB-UTC.
  dclock = ttmutc + obs.dtdb

  # Compute current BCRS position.
  (s, dsdt, pr) = obs.place(src, lfa.TR_MOTION)

  # Barycentric delay.
  delay = obs.bary_delay(s, pr)

  # BJD(TDB) summing Barycentric delay and clock correction.
  # While I advise against using it, I can see a possibility somebody
  # might want to calculate BJD(UTC) instead.  This can be done by
  # removing the "dclock" term.
  bjdtdb = jdutc + (delay+dclock)/lfa.DAY

  # Barycentric Doppler correction.
  zb = obs.bary_doppler(src, s, dsdt, pr)

  return bjdtdb, zb

