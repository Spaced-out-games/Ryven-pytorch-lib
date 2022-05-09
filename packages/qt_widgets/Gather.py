from ryven.NENV import *
print(__file__)
widgets = import_widgets("*/widgets.py")

class MakeTupleNode(Node):
    def __init__(self):
        title = 'MakeTuple'
        version = 'v0.1'
        main_widget_class = widgets.GatherWidget
        main_widget_class.IncreaseArgsButton.clicked.connect(self.update_event)
        main_widget_pos = 'between ports'
        init_inputs = [

        ]
        init_outputs = [
            NodeOutputBP()
        ]
        color = '#99dd55'

        def update_event(self, inp=-1):
            self.create_input(label = "input_"+str(len(self.inputs)))
        def test(self):
            print('test')
'''
class MakeKwargsNode(Node):
    def __init__(self):
        title = 'MakeTuple'
        version = 'v0.1'
        main_widget_class = widgets.GatherWidget
        main_widget_class.IncreaseArgsButton.clicked.connect(self.update_event)
        main_widget_pos = 'between ports'
        init_inputs = [

        ]
        init_outputs = [
            NodeOutputBP(type_='exec')
        ]
        color = '#99dd55'

        def update_event(self, inp=-1):
            self.create_input(label = "input_"+str(len(self.inputs)))
        def test(self):
            print('test')
'''
nodes = [MakeTupleNode]