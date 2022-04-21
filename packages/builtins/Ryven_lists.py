from ryven.NENV import *
#list > some shade of yellow
class AppendListNode(Node):
	'''
	Appends object to list.
	For example, if list = [0,0] and object = 5:
	list is now [0,0,5]
	'''
	title = 'append'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'object')
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		
		result = self.input(0)
		result.append(self.input(1))
		self.set_output_val(0,result)
class ListExtendNode(Node):
	'''Extend list by appending elements from the iterable.'''
	title = 'extend'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'iterable')
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).extend(self.input(1))
		self.set_output_val(0,result)
class ListIndexNode(Node):
	'''
	Return first index of value.
	Raises ValueError if the value is not present.

	Example:
	list = [0,3,6,1,9]
	list.index(3)
	>>>1


	'''
	title = 'index'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'value')
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).index(self.input(1))
		self.set_output_val(0,result)
class ListInsertNode(Node):
	'''
	Insert object before index

	Example:
	list = [0,2,7,1]
	list.insert(1,99)
	list
	>>>[0,99,2,7,1]
	'''
	title = 'insert'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'index'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'object')


		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).insert(self.input(1), self.input(2))
		self.set_output_val(0,result)
class ListRemoveNode(Node):
	'''
	Remove first occurrence of value.
	Raises ValueError if the value is not present.

	Example:
	list = [0,5,1,1]
	list.remove(1)
	list
	>>>[0,5,1]

	'''
	title = 'remove'
	tags = ['list']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
		NodeInputBP(dtype=dtypes.Data(default=1), label = 'value')
	]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).remove(self.input(1))
		self.set_output_val(0,result)
class ListReverseNode(Node):
	'''
	Reverses order of elements in list

	Example:
	list = [0,1,2,3,4]
	list.reverse()
	list
	>>>[4,3,2,1,0]
	'''
	title = 'reverse'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).reverse()
		self.set_output_val(0,result)
class ListSortNode(Node):
	'''
	Sorts list and returns it
	Example:
	list = [5,8,1,2,7,5,7]
	list.sort()
	list
	>>>[1,2,5,5,7,8]
	'''
	title = 'title'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		input = self.input
		result = input
		self.set_output_val(0,result)