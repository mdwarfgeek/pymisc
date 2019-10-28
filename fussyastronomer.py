import math

# Figure out how many digits to give based on error.
def ndp(err):
  ae = abs(err)

  if ae > 0:
    ndp = 1 - int(math.floor(math.log10(ae)))
    if ndp < 0:
      ndp = 0
  else:
    ndp = 0

  return ndp

def format(value, ndp):
  av = abs(value)
  if av > 0:
    ndig = int(math.ceil(math.log10(av)))
  else:
    ndig = 1

  w = ndig
  if value < 0:
    w += 1  # for - sign

  if ndp > 0:
    w += 1 + ndp

  fmt = "{0:" + str(w) + "." + str(ndp) + "f}"

  return fmt.format(value)

