from ryven.NENV import *
import numpy as np
np.ndarray(shape, dtype, buffer, offset, strides, order)
#append: arr, values, axis



class Make_ndarrayNode(Node):
	'''Create a numpy ndarray from an array-like'''
	title = 'list -> NumPy ndarray'
	tags = ['numpy', 'ndarrat']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'object'),
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'dtype'),
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'copy'),
		NodeInputBP(dtype=dtypes.Data(default=1),label = 'order'),
		#NodeInputBP(dtype=dtypes.Data(default=1),label = 'subok')
	]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		np_array = np.array(object = self.input(0), dtype = self.input(1), copy = self.input(2),order = self.input(3))
		self.set_output_val(np_array)
class np_ndarray_Node(Node):
	'''Create a numpy ndarray'''
	title = 'np.ndarray'
	tags = ['numpy', 'ndarray']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "shape"),
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "dtype"),
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "buffer"),
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "offset"),
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "strides"),
		NodeInputBP(dtype=dtypes.Data(default = 1), label = "order")
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		nd_array = np.ndarray(shape = self.inputs(0), dtype = self.inputs(1))
		self.set_output_val(0,result)