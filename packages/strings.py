from ryven.NENV import *
#join, replace, upper, lower, get, set
HEXCOL = "#a57417"
class StringJoinNode(Node):
    '''Join two strings together'''
    title = 'join'
    tags = ['str']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1)),
        NodeInputBP(dtype=dtypes.Data(default=1))
    ]
    init_outputs = [NodeOutputBP()]
    color = HEXCOL
    def update_event(self, inp=-1):
        result = self.input(0).join(self.input(1))
        self.set_output_val(0,result)
class StringReplaceNode(Node):
    '''Replaces every occurence of sub in str with new'''
    title = 'replace'
    tags = ['str']
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'str'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'sub'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'new')
        ]
    init_outputs = [NodeOutputBP()]
    color = HEXCOL
    def update_event(self, inp=-1):
        input = self.input
        result = input
        self.set_output_val(0,result)
class StringUpperNode(Node):
    '''Converts string to uppercase'''
    title = 'upper'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'string')]
    init_outputs = [NodeOutputBP()]
    color = HEXCOL
    def update_event(self, inp=-1):
        result = self.input(0).upper()
        self.set_output_val(0,result)
class StringLowerNode(Node):
    '''Converts string to lowercase'''
    title = 'lower'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'string')]
    init_outputs = [NodeOutputBP()]
    color = HEXCOL
    def update_event(self, inp=-1):
        result = self.input(0).lower()
        self.set_output_val(0,result)
class StringNode(Node):
    '''Converts input to string'''
    title = '-> str'
    tags = ['str']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'input')]
    init_outputs = [NodeOutputBP()]
    color = HEXCOL
    def update_event(self, inp=-1):
        self.set_output_val(0,str(self.input(0)))
str_nodes = [
    StringJoinNode,
    StringReplaceNode,
    StringUpperNode,
    StringLowerNode,
    StringNode
]