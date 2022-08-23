import os

import numpy

from catsearch import *
from fpcoord import *

dtype_full = numpy.dtype({ "names": ["source_id",
                                     "ra", "ra_error",
                                     "dec", "dec_error",
                                     "parallax", "parallax_error",
                                     "pmra", "pmra_error",
                                     "pmdec", "pmdec_error",
                                     "astrometric_excess_noise",
                                     "ruwe",
                                     "R_G", "G", "R_BP", "BP", "R_RP", "RP",
                                     "E_BP_BP"],
                           "formats": ["<i8",
                                       "<f8", "<f8",
                                       "<f8", "<f8",
                                       "<f8", "<f8",
                                       "<f8", "<f8",
                                       "<f8", "<f8",
                                       "<f8",
                                       "<f4",
                                       "<f4", "<f4", "<f4", "<f4", "<f4", "<f4",
                                       "<f4"] })

dtype_subset = numpy.dtype({ "names": ["source_id",
                                       "ra", "dec",
                                       "parallax", "pmra", "pmdec",
                                       "ruwe", "G", "BP", "RP"],
                             "formats": ["<i8",
                                         "<f8", "<f8",
                                         "<f8", "<f8", "<f8",
                                         "<f4", "<f4", "<f4", "<f4"] })

maxfile = 900
zonesperdeg = 5
basepath = "."

dtor = numpy.radians(1.0)

# Box cutout from GAIA catalogue.  Arguments in radians.
def boxgaia(cent_ra, cent_dec, width,
            basepath=None,
            subset=False):

  hwidth = width / 2

  if basepath is None:
    homedir = os.path.expanduser("~")
    basepath = os.path.join(homedir, "cats", "gaia", "edr3")
    if subset:
      basepath = os.path.join(basepath, "subset")
  
  if subset:
    dtype = dtype_subset
  else:
    dtype = dtype_full

  # Calculate coordinate limits corresponding to requested search box.
  ra_low, ra_high, dec_low, dec_high = radeclimits(cent_ra, cent_dec, hwidth)

  # GAIA uses decimal degrees.
  ra_low = numpy.degrees(ra_low)
  ra_high = numpy.degrees(ra_high)
  dec_low = numpy.degrees(dec_low)
  dec_high = numpy.degrees(dec_high)

  # Corresponding range of declination zones.
  ifile_low = int(math.floor((dec_low + 90) * zonesperdeg))
  ifile_high = int(math.floor((dec_high + 90) * zonesperdeg))

  # Clamp range.
  if ifile_low < 0:
    ifile_low = 0
  if ifile_high < 0:
    ifile_high = 0
  if ifile_low >= maxfile:
    ifile_low = maxfile-1
  if ifile_high >= maxfile:
    ifile_high = maxfile-1

  # Loop through zones we need.
  nfile = ifile_high - ifile_low + 1
  nr = len(ra_low)
  
  cutout = [None] * nfile * nr
  
  for ifile in range(nfile):
    zone = ifile_low + ifile + 1

    # File for zone.
    filename = os.path.join(basepath, "z{0:03d}".format(zone))

    # Memory map this file.
    data = numpy.memmap(filename,
                        dtype=dtype,
                        mode="r")

    for ir in range(nr):
      # File is sorted on RA, do binary chop to locate range of rows
      # corresponding to the requested RA range.
      ia, ib = numpy.searchsorted(data["ra"], [ ra_low[ir], ra_high[ir] ])
      ralimd = data[ia:ib]

      # Now select subset of rows also matching declination range.
      # This reduces computational burden of calculating standard
      # coordinates to only places where we need to.
      wd = numpy.logical_and(ralimd["dec"] >= dec_low,
                             ralimd["dec"] <= dec_high)
      radelimd = ralimd[wd]

      # Calculate standard coordinates for all remaining rows.
      xi, xn = standc(numpy.radians(radelimd["ra"]),
                      numpy.radians(radelimd["dec"]),
                      cent_ra, cent_dec)

      # Apply box.
      ws = numpy.logical_and(numpy.logical_and(xi >= -hwidth, xi <= hwidth),
                             numpy.logical_and(xn >= -hwidth, xn <= hwidth))
      thiscutout = radelimd[ws]

      # Convert result RA and DEC to radians for historical reasons.
      thiscutout["ra"] *= dtor
      thiscutout["dec"] *= dtor

      # Save result.
      cutout[ifile*nr+ir] = thiscutout

  # Return result, concatenating into single array.
  return numpy.concatenate(cutout)

