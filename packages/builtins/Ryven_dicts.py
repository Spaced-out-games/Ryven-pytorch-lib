from ryven.NENV import *
#clear, copy, fromkeys, get, items, keys, pop, popitem, setdefault, update,values

class DictFromKeysNode(Node):
	'''docstring'''
	title = 'title'
	tags = ['str']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		input = self.input
		result = input
		self.set_output_val(index,result)