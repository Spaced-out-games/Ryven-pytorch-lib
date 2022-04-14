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