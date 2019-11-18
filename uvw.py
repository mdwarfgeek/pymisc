import lfa
import math
import numpy

# Solar U,V,W relative to LSR from Dehnen & Binney 1998.
sol_uvw = numpy.array([ 10.00, 5.25, 7.17 ])

# Rough peculiar velocity outer boundaries for thin and thick disc.
# Still trying to find a reference.
vpec_thin  =  85  # km/s
vpec_thick = 180  # km/s

def radec2uvw(ra, de,         # rad
              pmra, e_pmra,   # arcsec/yr
              pmde, e_pmde,   # arcsec/yr
              plx,  e_plx,    # arcsec
              vrad, e_vrad):  # km/s

  # Method of Johnson & Soderblom (1987) but in ICRS.
  kop = lfa.AU / (lfa.JYR*lfa.DAY*1000*plx)

  sa = math.sin(ra)
  ca = math.cos(ra)
  sd = math.sin(de)
  cd = math.cos(de)

  T = lfa.eq2gal
  A = numpy.array([[ ca*cd, -sa, -ca*sd ],
                   [ sa*cd,  ca, -sa*sd ],
                   [    sd,   0,     cd ]])

  B = numpy.dot(T, A)
  C = B*B

  p = numpy.array([ vrad,
                    kop*pmra,
                    kop*pmde ])

  v_vrad = e_vrad*e_vrad
  v_pmra = e_pmra*e_pmra
  v_pmde = e_pmde*e_pmde

  rv_plx = e_plx*e_plx / (plx*plx)

  q = numpy.array([ v_vrad,
                    kop*kop*(v_pmra + pmra*pmra*rv_plx),
                    kop*kop*(v_pmde + pmde*pmde*rv_plx) ])

  uvw = numpy.dot(B, p)
  e_uvw = numpy.sqrt(numpy.dot(C, q) +
                     2*pmra*pmde*kop*kop*rv_plx*(B[:,1]*B[:,2]).T)
  
  return uvw, e_uvw


