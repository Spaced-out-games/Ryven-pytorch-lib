import re
from builtins import bool,bytearray,bytes,classmethod,dict,enumerate,float,function,filter,int,list,map,object,range,str,set,staticmethod,slice,super,tuple,type,zip
s = '''
args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`values`.
    device (:class:`torch.device`, optional)
'''
import re

def useRegex(input):
	pattern = re.compile("[a-zA-Z]+ \([^)]*\)")
	r = re.findall(pattern, input)
	ri = r[0]


	pattern = re.compile("[a-zA-Z]+")

	return re.findall(pattern, ri)#pattern.match(input, re.IGNORECASE)
r = useRegex(s)
print(r)
a