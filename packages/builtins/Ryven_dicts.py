from ryven.NENV import *
class DictFromKeysNode(Node):
	'''Create a new dictionary with keys from iterable and values set to value.'''
	title = 'dict_fromkeys'
	tags = ['dict']
	init_inputs = [
	NodeInputBP(dtype=dtypes.Data(default=1), label= 'iterable'),
	NodeInputBP(dtype=dtypes.Data(default=1), label= 'values')
	]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = dict.fromkeys(self.input(0), self.input(1))
		self.set_output_val(0,result)
class DictItemsNode(Node):
	'''a set-like object providing a view on D's items'''
	title = 'title'
	tags = ['dict']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'dict')]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).items()
		self.set_output_val(0,result)
class DictKeysNode(Node):
	'''D.keys() -> a set-like object providing a view on D's keys'''
	title = 'Get Keys'
	tags = ['dict']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'dict')]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).keys()
		self.set_output_val(0,result)
class DictPopItemNode(Node):
	'''Removes key from input. Returns a tuple'''
	title = 'Pop Item'
	tags = ['dict']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'input'),
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'key')
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).popitem()
		self.set_output_val(0,result)
class ToDictNode(Node):
	'''Convert input to dict, if input supports it'''
	title = 'title'
	tags = ['dict']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'input')]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = list(self.input(0))
		self.set_output_val(0,result)