from ryven.NENV import *
from ast import literal_eval
import ast
class PopNode(Node):
	'''Pop the last element of a list or str'''
	title = 'pop'
	tags = ['list','str','wildcard']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'input')]
	init_outputs = [NodeOutputBP()]
	color ='#aaaaaa'
	style = "normal"
	def update_event(self, inp=-1):
		result = self.input(0).pop()
		self.set_output_val(0,result)
class ElemGetNode(Node):
	'''
	Gets the nth element of a list, str, or dict
	X needs to be a str if input is dict
	'''
	title = 'get'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'input'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'X')
		]
	init_outputs = [NodeOutputBP()]
	color ='#aaaaaa'
	style = "normal"
	def update_event(self, inp=-1):
		result = self.input(0)[self.input(1)]
		self.set_output_val(0,result)
class ElemSetNode(Node):
	'''
	Sets element n of list, str, or dict to X
	X needs to be a str if input is dict
	'''
	title = 'set'
	tags = ['list', 'str','wildcard','dict']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'input'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'n'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'X'),
		]
	init_outputs = [NodeOutputBP()]
	color = "#aaaaaa"
	def update_event(self, inp=-1):
		self.input(0)[self.input(1)] = self.input(2)
class RangeGetNode(Node):
	'''
	Gets elements A thru B of X
	Use -1 for shorthand of last element
	'''
	title = 'Get (in range)'
	tags = ['str','wildcard','list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = "X"),
		NodeInputBP(dtype=dtypes.Data(default=1), label = "A"),
		NodeInputBP(dtype=dtypes.Data(default=1), label = "B")

	]
	init_outputs = [NodeOutputBP()]
	color ='#aaaaaa'
	style = "normal"
	def update_event(self, inp=-1):
		input = self.input
		result = self.input(0)[self.input(1):self.input(2)]
		self.set_output_val(0,result)
class RangeSetNode(Node):
	'''Sets elements A thru B of X to Y'''
	title = 'title'
	tags = ['str']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = "X"),
		NodeInputBP(dtype=dtypes.Data(default=1), label = "A"),
		NodeInputBP(dtype=dtypes.Data(default=1), label = "B"),
		NodeInputBP(dtype=dtypes.Data(default=1), label = "Y")
		]
	init_outputs = [NodeOutputBP()]
	color ='#aaaaaa'
	style = "normal"
	def update_event(self, inp=-1):
		self.input(0)[self.input(1):self.input(2)] = self.input(3)
class LiteralNode(Node):
	'''
	Takes the literal contents of a string and evaluates it
	Can be used to define dictionarries and arrays in a sensible manner
	'''
	title = 'title'
	tags = ['str']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	style = "normal"
	def update_event(self, inp=-1):
		self.set_output_val(0, ast.literal_eval(self.input(0)))
wildcard_nodes = [
	PopNode,
	ElemGetNode,
	ElemSetNode,
	RangeGetNode,
	RangeSetNode,
	LiteralNode
]