from ryven.NENV import *
import numpy as np
#list: #eaf10e
#tensor: #edb012
#numpy: #3693c9


class NodeBase(Node):
	
    version = 'v0.0'

    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(size='s')),
        NodeInputBP(dtype=dtypes.Data(size='s')),
    ]

    init_outputs = [
        NodeOutputBP(),
    ]

    style = 'small'

    def __init__(self, params):
        super().__init__(params)

        self.num_inputs = 0
        self.actions['add input'] = {'method': self.add_operand_input}

    def place_event(self):
        for i in range(len(self.inputs)):
            self.register_new_operand_input(i)

    def add_operand_input(self):
        self.create_input_dt(dtype=dtypes.Data(size='s'))
        self.register_new_operand_input(self.num_inputs)
        self.update()

    def remove_operand_input(self, index):
        self.delete_input(index)
        self.num_inputs -= 1
        # del self.actions[f'remove input {index}']
        self.rebuild_remove_actions()
        self.update()

    def register_new_operand_input(self, index):
        self.actions[f'remove input {index}'] = {
            'method': self.remove_operand_input,
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
            self.actions[f'remove input {i}'] = {'method': self.remove_operand_input, 'data': i}

    def update_event(self, inp=-1):
        self.set_output_val(0, self.apply_op([self.input(i) for i in range(len(self.inputs))]))

    def apply_op(self, elements: list):
        return None
class StringNodeBase(NodeBase):
    color = "#a98a16"
class TypeNodeBase(NodeBase):
    color = "#0816f7"

class TypeNode(TypeNodeBase):
    title = 'type'
    def apply_op(self,elements: list)->str:
        return all([type(elements[e]) for e in elements])
type_nodes = [TypeNode]












nodes = [
    *type_nodes
]