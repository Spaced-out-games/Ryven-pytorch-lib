from inspect import getfullargspec,getdoc,getsource
'''
class SimpleNode(Node):
	docstring
	title = 'title'
	tags = ['str']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=1))
		]
	init_outputs = [NodeOutputBP()]
	color ='#HEXCOL'
	def update_event(self, inp=-1):
		input = self.input
		result = input
		self.set_output_val(index,result)
'''
color = "#DEADBE"
def create_simplenode_from_function(f):
	args = getfullargspec(f)
	docstring = getdoc(f)
	name = f.__name__
	array = []
	for a in args:
		array.append("NodeInputBP(dtype=dtypes.Data(default=1), label = '" + a + "')" + (",\n\t\t" if a < len(args) else "\n]"))
	return[
		"class " + name + "Node(Node):",
		'\t"""' + docstring + '"""',
		"\ttags =['" + name + "','" + f.__module__ + "']",
		"\tinit_inputs = [" + str(array),
		"\tinit_outputs = NodeOutputBP()",
		"\tcolor = '"+ color + "'",
	]
a = create_simplenode_from_function(abs)
st = ""
for s in a:
	st +=a
print(a)