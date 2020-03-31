# The usual mass-radius relations for M-dwarfs.

# "Rss" (interferometry) from Boyajian et al. (2012).
def rss(m, age=None):
  return((0.3200 * m + 0.6063) * m + 0.0906)

# "Reb" (eclipsing binaries) from Boyajian et al. (2012).
def reb(m, age=None):
  return((-0.1297 * m + 1.0718) * m + 0.0135)

# Bayless & Orosz (2006) relation for eclipsing binaries.
def rbo(m, age=None):
  return((0.0374 * m + 0.9343) * m + 0.0324)

