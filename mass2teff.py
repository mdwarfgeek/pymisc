import numpy

from scipy.interpolate import InterpolatedUnivariateSpline

# Stellar mass to Teff for stars with convective envelopes.
# From Mamajek table version 2017.10.19
# The table has discontinuous derivatives so we use linear interpolation.

#   Msun, Teff ],  # SpT 
_mass2teff_tbl = [
  [ 1.21, 6240 ],  # F7V  
  [ 1.18, 6170 ],  # F8V  
  [ 1.14, 6040 ],  # F9V  
  [ 1.08, 5920 ],  # G0V  
  [ 1.07, 5880 ],  # G1V  
  [ 1.02, 5770 ],  # G2V  
  [ 1.00, 5720 ],  # G3V  
  [ 0.99, 5680 ],  # G4V  
  [ 0.98, 5660 ],  # G5V  
  [ 0.97, 5590 ],  # G6V  
  [ 0.96, 5530 ],  # G7V  
  [ 0.94, 5490 ],  # G8V  
  [ 0.90, 5340 ],  # G9V  
  [ 0.87, 5280 ],  # K0V  
  [ 0.85, 5170 ],  # K1V  
  [ 0.82, 5040 ],  # K2V  
  [ 0.78, 4840 ],  # K3V  
  [ 0.73, 4620 ],  # K4V  
  [ 0.71, 4410 ],  # K5V  
  [ 0.68, 4230 ],  # K6V  
  [ 0.64, 4070 ],  # K7V  
  [ 0.63, 4000 ],  # K8V  
  [ 0.61, 3940 ],  # K9V  
  [ 0.60, 3870 ],  # M0V  
  [ 0.56, 3800 ],  # M0.5V
  [ 0.53, 3700 ],  # M1V  
  [ 0.50, 3650 ],  # M1.5V
  [ 0.48, 3550 ],  # M2V  
  [ 0.44, 3500 ],  # M2.5V
  [ 0.39, 3410 ],  # M3V  
  [ 0.28, 3250 ],  # M3.5V
  [ 0.22, 3200 ],  # M4V  
  [ 0.18, 3100 ],  # M4.5V
  [ 0.15, 3030 ],  # M5V  
  [ 0.12, 3000 ],  # M5.5V
  [ 0.11, 2850 ],  # M6V  
  [ 0.10, 2710 ],  # M6.5V
  [ 0.09, 2650 ],  # M7V  
  [ 0.08, 2600 ],  # M7.5V
  [ 0.077, 2500 ],  # M8V  
  [ 0.071, 2440 ],  # M8.5V
  [ 0.065, 2400 ]   # M9V  
]

class mass2teff:
  def __init__(self):
    ttbl = numpy.array(_mass2teff_tbl)

    x = ttbl[:,0]
    y = ttbl[:,1]
    
    ix = numpy.argsort(x)
    
    self.spl = InterpolatedUnivariateSpline(x[ix], y[ix], k=1)

  def __call__(self, mass, age):
    return(self.spl(mass))

