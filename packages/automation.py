import inspect

def node_from_function(name: str, call: str, obj: object, color: str):
    """Create a node class definition from a given function.

    Parameters
    ----------
    name: str
        Becomes the title of the node.
    call: str
        Represents the full function call without parentheses, e.g. `math.sqrt`
    obj: object
        The function object. Its souce will be parsed using the inspect module,
        which might not work for some functions, expecially those of C bindings.
    color: str
        The color hex code the node will get.

    Returns
    -------
    node_def: str
        The code defining the node class.
    """
    
    try:
        sig = inspect.getfullargspec(obj)
    except Exception as e:
        raise Exception(f"Could not parse source of {name}.")
    
    inputs = '\n'.join([f"NodeInputBP('{param_name}')," for param_name in sig.args])
    node_name = f'{name}_Node'

    node_def = f"""
class {node_name}(Node):
    \"\"\"{obj.__doc__}\"\"\"

    title = '{name}'
    init_inputs = [
        {inputs}
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '{color}'

    def update_event(self, inp=-1):
        self.set_output_val(0, {call}({
            ', '.join([f'self.input({i})' for i in range(len(sig.args))]) 
                                        }))
"""

    return node_def
