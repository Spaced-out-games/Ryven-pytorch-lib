from ryven.NENV import *

#TODO: implement automatic node discovery and export to Ryven
#Once finished, file should be ran to export all libraries under packages into Ryven folder
import sys
import os
sys.path.append(os.path.dirname(__file__))

package_names = [name for name in os.listdir("./packages") if name.find(".") == -1]
print(package_names)

