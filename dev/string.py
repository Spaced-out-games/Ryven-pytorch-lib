from ryven.NENV import *


class capitalizeNode(Node):
    '''Converts the first character to upper case'''
    title = 'Capitalize'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
    init_outputs = [NodeOutputBP()]
    color ='#e09e1f'
    def update_event(self, inp=-1):
        input = self.input[0]
        result = input.capitalize()
        self.set_output_val(0,result)
class casefoldNode(Node):
    '''Converts a string into lowercase'''
    title = 'casefold'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
    init_outputs = [NodeOutputBP()]
    color ='#e09e1f'
    def update_event(self, inp=-1):
        input = self.input
        result = input.casefold()
        self.set_output_val(0,result)
class centerNode(Node):
    '''Returns a centered string'''
    title = 'center'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
    init_outputs = [NodeOutputBP()]
    color ='#HEXCOL'
    def update_event(self, inp=-1):
        input = self.input
        result = input.center()
        self.set_output_val(0,result)
class countNode(Node):
    '''Returns the number of times a specified value occurs in a string'''
    title = 'count'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
    init_outputs = [NodeOutputBP()]
    color ='#HEXCOL'
    def update_event(self, inp=-1):
        input = self.input
        result = input.count()
        self.set_output_val(0,result)
class encodeNode(Node):
    '''docstring'''
    title = 'encode'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1))]
    init_outputs = [NodeOutputBP()]
    color ='#HEXCOL'
    def update_event(self, inp=-1):
        input = self.input
        result = input.encode
        self.set_output_val(0,result)