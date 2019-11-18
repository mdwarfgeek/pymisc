import lfa

def convert_radec(radec):
  # Convert RA, DEC.  Try : first and then space.
  ra, rv = lfa.base60_to_10(radec, ':', lfa.UNIT_HR, lfa.UNIT_RAD)
  if rv < 0:
    ra, rv = lfa.base60_to_10(radec, ' ', lfa.UNIT_HR, lfa.UNIT_RAD)
    if rv < 0:
      raise RuntimeError("could not understand radec: " + radec)
    else:
      de, rv = lfa.base60_to_10(radec[rv:], ' ', lfa.UNIT_DEG, lfa.UNIT_RAD)
      if rv < 0:
        raise RuntimeError("could not understand radec: " + radec)
  else:
    de, rv = lfa.base60_to_10(radec[rv:], ':', lfa.UNIT_DEG, lfa.UNIT_RAD)
    if rv < 0:
      raise RuntimeError("could not understand radec: " + radec)

  return ra, de
