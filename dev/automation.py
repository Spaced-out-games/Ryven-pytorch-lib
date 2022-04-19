from imp import is_builtin
from string import ascii_uppercase
import sys
from inspect import isfunction, getargspec,getfullargspec
def num_args(f):
  if isfunction(f):
    return len(getargspec(f).args)
  else:
    spec = f.__doc__.split('\n')[0]
    args = spec[spec.find('(')+1:spec.find(')')]
    return args.count(',')+1 if args else 0
def node_from_function(name: str, call: str, obj: object, color: str):
    """Create a node class definition from a given function.

    Parameters
    ----------
    name: str
        Becomes the title of the node.
    call: str
        Represents the full function call without parentheses, e.g. `math.sqrt`
    obj: object
        The function object. Its source will be parsed using the inspect module,
        which might not work for some functions, expecially those of C bindings.
    color: str
        The color hex code the node will get.

    Returns
    -------
    node_def: str
        The code defining the node class.
    """
    
    try:
        sig = getfullargspec(obj)
        args = sig.args
        warning = ""
    except Exception as e:
        #raise Exception(f"Could not parse source of {name}.")
        #"""
        
        warning = f'''\n
"""
WARNING: Module {name} was generated using fallback option. May contain bugs
"""
'''
        
        #"""
        argcnt = num_args(obj)
        args = []
        for i in range(argcnt):
            args.append(chr(i + 97))
    
    inputs = '\n'.join([f"NodeInputBP('{param_name}')," for param_name in args])
    node_name = f'{name}_Node'

    node_def = f"""
{warning}
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
            ', '.join([f'self.input({i})' for i in range(len(args))]) 
                                        }))
"""

    return node_def
def attributes_without_builtins(o):
    """_summary_

    Rarameters:
    ----------
        o (object): object to filter

    Returns:
    ----------
        list: dir(o) with builtin attributes like __eq__ removed
    """
    d = dir(o)
    for i in range(len(d)):
        if d[i].find("__") == 0:
            d[i] = ""
    while d.count("") > 0:
        d.pop(d.index(""))
    return d
import math

module = math
d = attributes_without_builtins(module)
for i in d:
    f = getattr(module,i)
    name = module.__name__.capitalize() + i.capitalize() + "Node"
    code = node_from_function(name,i,f,"#aaaaaa")
    if isinstance(code, str):
        print(code)
        

