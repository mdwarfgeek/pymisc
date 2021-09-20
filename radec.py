import lfa

def convert_radec(radec, partial=False):
  # Convert RA, DEC.  Try : first and then space.
  ra, rva = lfa.base60_to_10(radec, ':', lfa.UNIT_HR, lfa.UNIT_RAD)
  if rva < 0:
    ra, rva = lfa.base60_to_10(radec, ' ', lfa.UNIT_HR, lfa.UNIT_RAD)
    if rva < 0:
      raise RuntimeError("could not understand radec: " + radec)
    else:
      de, rvd = lfa.base60_to_10(radec[rva:], ' ', lfa.UNIT_DEG, lfa.UNIT_RAD)
      if rvd < 0:
        raise RuntimeError("could not understand radec: " + radec)
  else:
    de, rvd = lfa.base60_to_10(radec[rva:], ':', lfa.UNIT_DEG, lfa.UNIT_RAD)
    if rvd < 0:
      raise RuntimeError("could not understand radec: " + radec)

  if partial:
    return ra, de, rva+rvd
  else:
    return ra, de
