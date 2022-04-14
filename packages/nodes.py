from ryven.NENV import *

import torch as t

class TensorNodeBase(Node):
	version = 'v0.0'
	init_inputs = [
		NodeInputBP(dtype = dtypes.Data(size = "l"))
	]
	init_outputs =[
		NodeOutputBP()
	]
	style = 'small'
	def __init__(self, params):
		super().__init__(params)

		self.num_inputs = 0
		self.actions['add input'] = {'method': self.add_input}

	def place_event(self):
		for i in range(len(self.inputs)):
			self.register_new_input(i)

	def add_input(self):
		self.create_input_dt(dtype=dtypes.Data(size='s'))
		self.register_new_input(self.num_inputs)
		self.update()

	def remove_input(self, index):
		self.delete_input(index)
		self.num_inputs -= 1
		# del self.actions[f'remove input {index}']
		self.rebuild_remove_actions()
		self.update()

	def register_new_input(self, index):
		self.actions[f'remove input {index}'] = {
			'method': self.remove_input,
			'data': index
		}
		self.num_inputs += 1

	def rebuild_remove_actions(self):

		remove_keys = []
		for k, v in self.actions.items():
			if k.startswith('remove input'):
				remove_keys.append(k)

		for k in remove_keys:
			del self.actions[k]

		for i in range(self.num_inputs):
			self.actions[f'remove input {i}'] = {'method': self.remove_input, 'data': i}

	def update_event(self, inp=-1):
		self.set_output_val(0, self.apply_op([self.input(i) for i in range(len(self.inputs))]))

	def apply_op(self, elements: list):
		return None
class TensorBase(TensorNodeBase):
	color = "#f58142"
class FromNumpy_Node(TensorBase):
	def apply_op(self, elements: list):
		return all(t.from_numpy(e) for e in elements)
tensor_nodes = [FromNumpy_Node]

nodes = [*tensor_nodes]