from ryven.NENV import *


from random import random

class RandNode(Node):
    """Generates scaled random float values"""
    # the docstring will be shown as tooltip in the editor

    title = 'Rand'  # the display_title is title by default
    tags = ['random', 'numbers']  # for better search
    
    init_inputs = [  # one input
        NodeInputBP(dtype=dtypes.Data(default=1))
        # the dtype will automatically provide a suitable widget
    ]
    init_outputs = [  # and one output
        NodeOutputBP()
    ]
    color = '#fcba03'

    def update_event(self, inp=-1):
        # update first output
        self.set_output_val(0, 
            random() * self.input(0)  # random float between 0 and value at input
        )
class PrintNode(Node):
    title = 'Print'
    init_inputs = [
        NodeInputBP(),
    ]
    color = '#A9D5EF'

    def update_event(self, inp=-1):
        print(self.input(0))

export_nodes(
    RandNode,
	PrintNode
)