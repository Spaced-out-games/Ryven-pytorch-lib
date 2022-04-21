from ryven.NENV import *
#append,clear,copy, count, extend, index, insert, mro, pop, remove, reverse,sort
#list > some shade of yelllow
class AppendListNode(Node):
	'''
	Appends object to list.
	For example, if list = [0,0] and object = 5:
	list is now [0,0,5]
	'''
	title = 'Append'
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
class ClearListode(Node):
	'''
	Clears all elements from a list, eg:
	[0,4,6,1,5,...] --> []
	'''
	title = 'clear'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'list')]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0)
		result.clear()
		self.set_output_val(0,result)
class ListCopyNode(Node):
	'''Return a copy of list. The new list can be edited without affecting old list'''
	title = 'copy'
	tags = ['list']
	init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1),label = 'list')]
	init_outputs = [NodeOutputBP(label="new list")]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		result = self.input(0).copy()
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

