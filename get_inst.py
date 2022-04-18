import dis
import types
from numpy import *
#Credit: Ashwini Chaudhary
def get_inst(f):
    ins = dis.get_instructions(f)
    for x in ins:
        try:
            if x.opcode == 100 and '<locals>' in next(ins).argval\
                                              and next(ins).opcode == 132:
                yield next(ins).argrepr
                yield from get_inst(x.argval)
        except Exception:
            pass
e = list(get_inst())
print(e)