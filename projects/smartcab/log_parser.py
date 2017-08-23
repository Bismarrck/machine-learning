from __future__ import print_function

import re
from collections import namedtuple

CabState = namedtuple(
  "CabState", 
  ("waypoint", "light", "left", "right", "oncoming")
)
state_patt = re.compile(r"^\((.*)\)")
q_patt = re.compile(r"--\s(.*)\s:\s([0-9.-]+)")


def parse_q_table(filename):

  table = {}
  parse_state = True
  state = None

  with open(filename) as f:
    for line in f:
      line = line.strip()
      if parse_state:
        m = state_patt.search(line)
        if m:
          state = CabState(*tuple(eval(s.strip()) 
            for s in m.group(1).split(",")))
          table[state] = {}
          parse_state = False
      else:
        m = q_patt.search(line)
        if m:
          action = m.group(1)
          if action == "None":
            action = None
          value = float(m.group(2))
          table[state][action] = value
        else:
          parse_state = True

  return table


if __name__ == "__main__":

  from collections import Counter

  table = parse_q_table("./logs/sim_improved-learning.txt")
  counter = Counter()

  for state, q_values in table.items():
    counter.update(q_values.keys())

  cs1 = CabState('right', 'red', None, 'forward', None)
  cs2 = CabState('right', 'green', 'left', 'left', 'forward')
  cs3 = CabState('left', 'red', None, 'left', 'forward') 
  cs4 = CabState('right', 'green', 'forward', 'forward', None)
  cs5 = CabState('forward', 'green', None, 'forward', None)

  print(table[cs1])
  print(table[cs2])
  print(table[cs3])
  print(table[cs4])
  print(table[cs5])

