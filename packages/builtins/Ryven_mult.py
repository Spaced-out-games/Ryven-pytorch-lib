from ryven.NENV import *
#nodes from this file are applicable to multiple data types
class ClearNode(Node):
	'''
	Clears all elements from a list or dict, eg:
	[0,4,6,1,5,...] --> []
	
	'''
	title = 'clear'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'list/dict')]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0)
		result.clear()
		self.set_output_val(0,result)
class CopyNode(Node):
	'''Return a copy of list. The new list can be edited without affecting old list'''
	title = 'copy'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1),label = 'list/dict')]
	init_outputs = [NodeOutputBP(label="new list")]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).copy()
		self.set_output_val(0,result)
