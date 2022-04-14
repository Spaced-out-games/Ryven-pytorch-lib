from ryven.NENV import *
import numpy as np
#list: #eaf10e
#tensor: #edb012
#numpy: #3693c9
pallete = {
    "list": "#eaf10e",
    "Tensor": "#edb012",
    "numpy": "#3693c9"
}
'''
append
count
pop
clear
remove
'''
class ListCountNode(Node):
    '''Number of occurences of element in list'''
    title = 'count'
    tags = ['list']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1),label = "list"),
        NodeInputBP(dtype=dtypes.Data(default=1), label = "element")
        ]
    init_outputs = [NodeOutputBP()]
    color = pallete['list']
    def update_event(self, inp=-1):
        result =  self.input(0).count(self.input(1))
        self.set_output_val(0,result)
class ListAppendNode(Node):
    '''Append element to the end of list'''
    title = 'append'
    tags = ['list']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'element')
    
    ]
    init_outputs = [NodeOutputBP()]
    color =pallete['list']#placeholder
    def update_event(self, inp=-1):
        self.input(0).append(self.input(1))
class ListPopNode(Node):
    '''Pop the last element of a list'''
    title = 'pop'
    tags = ['list']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'list')]
    init_outputs = [NodeOutputBP()]
    color ='#HEXCOL'
    def update_event(self, inp=-1):
        result = self.input(0).pop()
        self.set_output_val(0,result)
class ListClearNode(Node):
    '''Clears list'''
    title = 'clear'
    tags = ['list']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'list')]
    init_outputs = [NodeOutputBP()]
    color = pallete['list']
    def update_event(self, inp=-1):
        self.input(0).clear()
class ListGetNode(Node):
    '''Get nth element of list'''
    title = 'get'
    tags = ['list']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'n')
        ]
    init_outputs = [NodeOutputBP()]
    color = pallete['list']
    def update_event(self, inp=-1):
        result = self.input(0)[self.input(1)]
        self.set_output_val(0,result)
class ListSetNode(Node):
    '''Sets nth element of list to X'''
    title = 'set'
    tags = ['list']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'list'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'n'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'X'),
        ]
    init_outputs = [NodeOutputBP()]
    color = pallete['list']
    def update_event(self, inp=-1):
        self.input(0)[self.input(1)] = self.input(2)
class ListNode(Node):
    '''Converts Wildcard into a list'''
    title = 'list of'
    tags = ['list']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = "Wildcard")]
    init_outputs = [NodeOutputBP()]
    color =pallete['list']
    def update_event(self, inp=-1):
        self.set_output_val(index,list(self.input(0)))
list_nodes = [
    ListCountNode,
    ListAppendNode,
    ListPopNode,
    ListClearNode,
    ListGetNode,
    ListSetNode
]