
from ryven.NENV import *
import torch



"""
WARNING: Module AvgNode was generated using fallback option. May contain bugs
"""

class AvgNode(Node):
    """Members:

  SUM

  AVG"""

    title = 'AvgNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.AVG(self.input(0)))



"""
WARNING: Module AggregationtypeNode was generated using fallback option. May contain bugs
"""

class AggregationtypeNode(Node):
    """Members:

  SUM

  AVG"""

    title = 'AggregationtypeNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.AggregationType(self.input(0)))



"""
WARNING: Module AliasdbNode was generated using fallback option. May contain bugs
"""

class AliasdbNode(Node):
    """None"""

    title = 'AliasdbNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.AliasDb())



"""
WARNING: Module AnytypeNode was generated using fallback option. May contain bugs
"""

class AnytypeNode(Node):
    """None"""

    title = 'AnytypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.AnyType())



"""
WARNING: Module ArgumentNode was generated using fallback option. May contain bugs
"""

class ArgumentNode(Node):
    """None"""

    title = 'ArgumentNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Argument())



"""
WARNING: Module ArgumentspecNode was generated using fallback option. May contain bugs
"""

class ArgumentspecNode(Node):
    """None"""

    title = 'ArgumentspecNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ArgumentSpec())


class Bfloat16storageNode(Node):
    """None"""

    title = 'Bfloat16storageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BFloat16Storage(self.input(0)))



"""
WARNING: Module Bfloat16tensorNode was generated using fallback option. May contain bugs
"""

class Bfloat16tensorNode(Node):
    """None"""

    title = 'Bfloat16tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BFloat16Tensor())



"""
WARNING: Module BenchmarkconfigNode was generated using fallback option. May contain bugs
"""

class BenchmarkconfigNode(Node):
    """None"""

    title = 'BenchmarkconfigNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BenchmarkConfig())



"""
WARNING: Module BenchmarkexecutionstatsNode was generated using fallback option. May contain bugs
"""

class BenchmarkexecutionstatsNode(Node):
    """None"""

    title = 'BenchmarkexecutionstatsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BenchmarkExecutionStats())



"""
WARNING: Module BlockNode was generated using fallback option. May contain bugs
"""

class BlockNode(Node):
    """None"""

    title = 'BlockNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Block())


class BoolstorageNode(Node):
    """None"""

    title = 'BoolstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BoolStorage(self.input(0)))



"""
WARNING: Module BooltensorNode was generated using fallback option. May contain bugs
"""

class BooltensorNode(Node):
    """None"""

    title = 'BooltensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BoolTensor())



"""
WARNING: Module BooltypeNode was generated using fallback option. May contain bugs
"""

class BooltypeNode(Node):
    """None"""

    title = 'BooltypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BoolType())



"""
WARNING: Module BufferdictNode was generated using fallback option. May contain bugs
"""

class BufferdictNode(Node):
    """None"""

    title = 'BufferdictNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.BufferDict())


class BytestorageNode(Node):
    """None"""

    title = 'BytestorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ByteStorage(self.input(0)))



"""
WARNING: Module BytetensorNode was generated using fallback option. May contain bugs
"""

class BytetensorNode(Node):
    """None"""

    title = 'BytetensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ByteTensor())



"""
WARNING: Module Conv_bn_fusionNode was generated using fallback option. May contain bugs
"""

class Conv_bn_fusionNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'Conv_bn_fusionNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CONV_BN_FUSION(self.input(0)))



"""
WARNING: Module CallstackNode was generated using fallback option. May contain bugs
"""

class CallstackNode(Node):
    """None"""

    title = 'CallstackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CallStack())



"""
WARNING: Module CapsuleNode was generated using fallback option. May contain bugs
"""

class CapsuleNode(Node):
    """None"""

    title = 'CapsuleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Capsule())


class CharstorageNode(Node):
    """None"""

    title = 'CharstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CharStorage(self.input(0)))



"""
WARNING: Module ChartensorNode was generated using fallback option. May contain bugs
"""

class ChartensorNode(Node):
    """None"""

    title = 'ChartensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CharTensor())



"""
WARNING: Module ClasstypeNode was generated using fallback option. May contain bugs
"""

class ClasstypeNode(Node):
    """None"""

    title = 'ClasstypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ClassType())



"""
WARNING: Module CodeNode was generated using fallback option. May contain bugs
"""

class CodeNode(Node):
    """None"""

    title = 'CodeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Code())



"""
WARNING: Module CompilationunitNode was generated using fallback option. May contain bugs
"""

class CompilationunitNode(Node):
    """None"""

    title = 'CompilationunitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CompilationUnit())



"""
WARNING: Module CompleteargumentspecNode was generated using fallback option. May contain bugs
"""

class CompleteargumentspecNode(Node):
    """None"""

    title = 'CompleteargumentspecNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.CompleteArgumentSpec())


class ComplexdoublestorageNode(Node):
    """None"""

    title = 'ComplexdoublestorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ComplexDoubleStorage(self.input(0)))


class ComplexfloatstorageNode(Node):
    """None"""

    title = 'ComplexfloatstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ComplexFloatStorage(self.input(0)))



"""
WARNING: Module ComplextypeNode was generated using fallback option. May contain bugs
"""

class ComplextypeNode(Node):
    """None"""

    title = 'ComplextypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ComplexType())



"""
WARNING: Module ConcretemoduletypeNode was generated using fallback option. May contain bugs
"""

class ConcretemoduletypeNode(Node):
    """None"""

    title = 'ConcretemoduletypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ConcreteModuleType())



"""
WARNING: Module ConcretemoduletypebuilderNode was generated using fallback option. May contain bugs
"""

class ConcretemoduletypebuilderNode(Node):
    """None"""

    title = 'ConcretemoduletypebuilderNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ConcreteModuleTypeBuilder())



"""
WARNING: Module DeepcopymemotableNode was generated using fallback option. May contain bugs
"""

class DeepcopymemotableNode(Node):
    """None"""

    title = 'DeepcopymemotableNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DeepCopyMemoTable())



"""
WARNING: Module DeviceobjtypeNode was generated using fallback option. May contain bugs
"""

class DeviceobjtypeNode(Node):
    """None"""

    title = 'DeviceobjtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DeviceObjType())



"""
WARNING: Module DicttypeNode was generated using fallback option. May contain bugs
"""

class DicttypeNode(Node):
    """None"""

    title = 'DicttypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DictType())



"""
WARNING: Module DisabletorchfunctionNode was generated using fallback option. May contain bugs
"""

class DisabletorchfunctionNode(Node):
    """None"""

    title = 'DisabletorchfunctionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DisableTorchFunction())


class DoublestorageNode(Node):
    """None"""

    title = 'DoublestorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DoubleStorage(self.input(0)))



"""
WARNING: Module DoubletensorNode was generated using fallback option. May contain bugs
"""

class DoubletensorNode(Node):
    """None"""

    title = 'DoubletensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.DoubleTensor())



"""
WARNING: Module EnumtypeNode was generated using fallback option. May contain bugs
"""

class EnumtypeNode(Node):
    """None"""

    title = 'EnumtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.EnumType())



"""
WARNING: Module ErrorreportNode was generated using fallback option. May contain bugs
"""

class ErrorreportNode(Node):
    """None"""

    title = 'ErrorreportNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ErrorReport())



"""
WARNING: Module ExecutionplanNode was generated using fallback option. May contain bugs
"""

class ExecutionplanNode(Node):
    """None"""

    title = 'ExecutionplanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ExecutionPlan())



"""
WARNING: Module Fuse_add_reluNode was generated using fallback option. May contain bugs
"""

class Fuse_add_reluNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'Fuse_add_reluNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FUSE_ADD_RELU(self.input(0)))



"""
WARNING: Module FatalerrorNode was generated using fallback option. May contain bugs
"""

class FatalerrorNode(Node):
    """None"""

    title = 'FatalerrorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FatalError())



"""
WARNING: Module FilecheckNode was generated using fallback option. May contain bugs
"""

class FilecheckNode(Node):
    """None"""

    title = 'FilecheckNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FileCheck())


class FloatstorageNode(Node):
    """None"""

    title = 'FloatstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FloatStorage(self.input(0)))



"""
WARNING: Module FloattensorNode was generated using fallback option. May contain bugs
"""

class FloattensorNode(Node):
    """None"""

    title = 'FloattensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FloatTensor())



"""
WARNING: Module FloattypeNode was generated using fallback option. May contain bugs
"""

class FloattypeNode(Node):
    """None"""

    title = 'FloattypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FloatType())



"""
WARNING: Module FunctionschemaNode was generated using fallback option. May contain bugs
"""

class FunctionschemaNode(Node):
    """None"""

    title = 'FunctionschemaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FunctionSchema())



"""
WARNING: Module FutureNode was generated using fallback option. May contain bugs
"""

class FutureNode(Node):
    """None"""

    title = 'FutureNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Future())



"""
WARNING: Module FuturetypeNode was generated using fallback option. May contain bugs
"""

class FuturetypeNode(Node):
    """None"""

    title = 'FuturetypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.FutureType())



"""
WARNING: Module GeneratorNode was generated using fallback option. May contain bugs
"""

class GeneratorNode(Node):
    """
Generator(device='cpu') -> Generator

Creates and returns a generator object that manages the state of the algorithm which
produces pseudo random numbers. Used as a keyword argument in many :ref:`inplace-random-sampling`
functions.

Arguments:
    device (:class:`torch.device`, optional): the desired device for the generator.

Returns:
    Generator: An torch.Generator object.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cuda = torch.Generator(device='cuda')
"""

    title = 'GeneratorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Generator())



"""
WARNING: Module GradientNode was generated using fallback option. May contain bugs
"""

class GradientNode(Node):
    """None"""

    title = 'GradientNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Gradient())



"""
WARNING: Module GraphNode was generated using fallback option. May contain bugs
"""

class GraphNode(Node):
    """None"""

    title = 'GraphNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Graph())



"""
WARNING: Module GraphexecutorstateNode was generated using fallback option. May contain bugs
"""

class GraphexecutorstateNode(Node):
    """None"""

    title = 'GraphexecutorstateNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.GraphExecutorState())



"""
WARNING: Module Hoist_conv_packed_paramsNode was generated using fallback option. May contain bugs
"""

class Hoist_conv_packed_paramsNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'Hoist_conv_packed_paramsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.HOIST_CONV_PACKED_PARAMS(self.input(0)))


class HalfstorageNode(Node):
    """None"""

    title = 'HalfstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.HalfStorage(self.input(0)))



"""
WARNING: Module HalfstoragebaseNode was generated using fallback option. May contain bugs
"""

class HalfstoragebaseNode(Node):
    """None"""

    title = 'HalfstoragebaseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.HalfStorageBase())



"""
WARNING: Module HalftensorNode was generated using fallback option. May contain bugs
"""

class HalftensorNode(Node):
    """None"""

    title = 'HalftensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.HalfTensor())



"""
WARNING: Module Insert_fold_prepack_opsNode was generated using fallback option. May contain bugs
"""

class Insert_fold_prepack_opsNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'Insert_fold_prepack_opsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.INSERT_FOLD_PREPACK_OPS(self.input(0)))



"""
WARNING: Module IodescriptorNode was generated using fallback option. May contain bugs
"""

class IodescriptorNode(Node):
    """None"""

    title = 'IodescriptorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.IODescriptor())



"""
WARNING: Module InferredtypeNode was generated using fallback option. May contain bugs
"""

class InferredtypeNode(Node):
    """None"""

    title = 'InferredtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.InferredType())


class IntstorageNode(Node):
    """None"""

    title = 'IntstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.IntStorage(self.input(0)))



"""
WARNING: Module InttensorNode was generated using fallback option. May contain bugs
"""

class InttensorNode(Node):
    """None"""

    title = 'InttensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.IntTensor())



"""
WARNING: Module InttypeNode was generated using fallback option. May contain bugs
"""

class InttypeNode(Node):
    """None"""

    title = 'InttypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.IntType())



"""
WARNING: Module InterfacetypeNode was generated using fallback option. May contain bugs
"""

class InterfacetypeNode(Node):
    """None"""

    title = 'InterfacetypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.InterfaceType())



"""
WARNING: Module JitexceptionNode was generated using fallback option. May contain bugs
"""

class JitexceptionNode(Node):
    """None"""

    title = 'JitexceptionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.JITException())



"""
WARNING: Module ListtypeNode was generated using fallback option. May contain bugs
"""

class ListtypeNode(Node):
    """None"""

    title = 'ListtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ListType())



"""
WARNING: Module LitescriptmoduleNode was generated using fallback option. May contain bugs
"""

class LitescriptmoduleNode(Node):
    """None"""

    title = 'LitescriptmoduleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.LiteScriptModule())



"""
WARNING: Module LockingloggerNode was generated using fallback option. May contain bugs
"""

class LockingloggerNode(Node):
    """None"""

    title = 'LockingloggerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.LockingLogger())



"""
WARNING: Module LoggerbaseNode was generated using fallback option. May contain bugs
"""

class LoggerbaseNode(Node):
    """None"""

    title = 'LoggerbaseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.LoggerBase())


class LongstorageNode(Node):
    """None"""

    title = 'LongstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.LongStorage(self.input(0)))



"""
WARNING: Module LongtensorNode was generated using fallback option. May contain bugs
"""

class LongtensorNode(Node):
    """None"""

    title = 'LongtensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.LongTensor())



"""
WARNING: Module MobileoptimizertypeNode was generated using fallback option. May contain bugs
"""

class MobileoptimizertypeNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'MobileoptimizertypeNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.MobileOptimizerType(self.input(0)))



"""
WARNING: Module ModuledictNode was generated using fallback option. May contain bugs
"""

class ModuledictNode(Node):
    """None"""

    title = 'ModuledictNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ModuleDict())



"""
WARNING: Module NodeNode was generated using fallback option. May contain bugs
"""

class NodeNode(Node):
    """None"""

    title = 'NodeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Node())



"""
WARNING: Module NonetypeNode was generated using fallback option. May contain bugs
"""

class NonetypeNode(Node):
    """None"""

    title = 'NonetypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.NoneType())



"""
WARNING: Module NooploggerNode was generated using fallback option. May contain bugs
"""

class NooploggerNode(Node):
    """None"""

    title = 'NooploggerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.NoopLogger())



"""
WARNING: Module NumbertypeNode was generated using fallback option. May contain bugs
"""

class NumbertypeNode(Node):
    """None"""

    title = 'NumbertypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.NumberType())



"""
WARNING: Module OptionaltypeNode was generated using fallback option. May contain bugs
"""

class OptionaltypeNode(Node):
    """None"""

    title = 'OptionaltypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.OptionalType())



"""
WARNING: Module ParameterdictNode was generated using fallback option. May contain bugs
"""

class ParameterdictNode(Node):
    """None"""

    title = 'ParameterdictNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ParameterDict())



"""
WARNING: Module PyobjecttypeNode was generated using fallback option. May contain bugs
"""

class PyobjecttypeNode(Node):
    """None"""

    title = 'PyobjecttypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.PyObjectType())



"""
WARNING: Module PytorchfilereaderNode was generated using fallback option. May contain bugs
"""

class PytorchfilereaderNode(Node):
    """None"""

    title = 'PytorchfilereaderNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.PyTorchFileReader())



"""
WARNING: Module PytorchfilewriterNode was generated using fallback option. May contain bugs
"""

class PytorchfilewriterNode(Node):
    """None"""

    title = 'PytorchfilewriterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.PyTorchFileWriter())


class Qint32storageNode(Node):
    """None"""

    title = 'Qint32storageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QInt32Storage(self.input(0)))



"""
WARNING: Module Qint32storagebaseNode was generated using fallback option. May contain bugs
"""

class Qint32storagebaseNode(Node):
    """None"""

    title = 'Qint32storagebaseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QInt32StorageBase())


class Qint8storageNode(Node):
    """None"""

    title = 'Qint8storageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QInt8Storage(self.input(0)))



"""
WARNING: Module Qint8storagebaseNode was generated using fallback option. May contain bugs
"""

class Qint8storagebaseNode(Node):
    """None"""

    title = 'Qint8storagebaseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QInt8StorageBase())


class Quint4x2storageNode(Node):
    """None"""

    title = 'Quint4x2storageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QUInt4x2Storage(self.input(0)))


class Quint8storageNode(Node):
    """None"""

    title = 'Quint8storageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.QUInt8Storage(self.input(0)))



"""
WARNING: Module Remove_dropoutNode was generated using fallback option. May contain bugs
"""

class Remove_dropoutNode(Node):
    """Members:

  CONV_BN_FUSION

  INSERT_FOLD_PREPACK_OPS

  REMOVE_DROPOUT

  FUSE_ADD_RELU

  HOIST_CONV_PACKED_PARAMS"""

    title = 'Remove_dropoutNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.REMOVE_DROPOUT(self.input(0)))



"""
WARNING: Module RreftypeNode was generated using fallback option. May contain bugs
"""

class RreftypeNode(Node):
    """None"""

    title = 'RreftypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.RRefType())



"""
WARNING: Module SumNode was generated using fallback option. May contain bugs
"""

class SumNode(Node):
    """Members:

  SUM

  AVG"""

    title = 'SumNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.SUM(self.input(0)))



"""
WARNING: Module ScriptclassNode was generated using fallback option. May contain bugs
"""

class ScriptclassNode(Node):
    """<property object at 0x7fdfe1f239a0>"""

    title = 'ScriptclassNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptClass())



"""
WARNING: Module ScriptclassfunctionNode was generated using fallback option. May contain bugs
"""

class ScriptclassfunctionNode(Node):
    """None"""

    title = 'ScriptclassfunctionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptClassFunction())



"""
WARNING: Module ScriptfunctionNode was generated using fallback option. May contain bugs
"""

class ScriptfunctionNode(Node):
    """
Functionally equivalent to a :class:`ScriptModule`, but represents a single
function and does not have any attributes or Parameters.
"""

    title = 'ScriptfunctionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptFunction())



"""
WARNING: Module ScriptmethodNode was generated using fallback option. May contain bugs
"""

class ScriptmethodNode(Node):
    """None"""

    title = 'ScriptmethodNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptMethod())



"""
WARNING: Module ScriptmoduleNode was generated using fallback option. May contain bugs
"""

class ScriptmoduleNode(Node):
    """None"""

    title = 'ScriptmoduleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptModule())



"""
WARNING: Module ScriptmoduleserializerNode was generated using fallback option. May contain bugs
"""

class ScriptmoduleserializerNode(Node):
    """None"""

    title = 'ScriptmoduleserializerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptModuleSerializer())



"""
WARNING: Module ScriptobjectNode was generated using fallback option. May contain bugs
"""

class ScriptobjectNode(Node):
    """None"""

    title = 'ScriptobjectNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ScriptObject())


class SetNode(Node):
    """A generic version of set."""

    title = 'SetNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Set(self.input(0)))


class ShortstorageNode(Node):
    """None"""

    title = 'ShortstorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ShortStorage(self.input(0)))



"""
WARNING: Module ShorttensorNode was generated using fallback option. May contain bugs
"""

class ShorttensorNode(Node):
    """None"""

    title = 'ShorttensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ShortTensor())


class SizeNode(Node):
    """None"""

    title = 'SizeNode'
    init_inputs = [
        NodeInputBP('iterable'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Size(self.input(0)))



"""
WARNING: Module StaticmoduleNode was generated using fallback option. May contain bugs
"""

class StaticmoduleNode(Node):
    """None"""

    title = 'StaticmoduleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.StaticModule())


class StorageNode(Node):
    """None"""

    title = 'StorageNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Storage(self.input(0)))



"""
WARNING: Module StoragecontextNode was generated using fallback option. May contain bugs
"""

class StoragecontextNode(Node):
    """None"""

    title = 'StoragecontextNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.StorageContext())



"""
WARNING: Module StreamNode was generated using fallback option. May contain bugs
"""

class StreamNode(Node):
    """None"""

    title = 'StreamNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Stream())



"""
WARNING: Module StreamobjtypeNode was generated using fallback option. May contain bugs
"""

class StreamobjtypeNode(Node):
    """None"""

    title = 'StreamobjtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.StreamObjType())



"""
WARNING: Module StringtypeNode was generated using fallback option. May contain bugs
"""

class StringtypeNode(Node):
    """None"""

    title = 'StringtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.StringType())



"""
WARNING: Module Type_checkingNode was generated using fallback option. May contain bugs
"""

class Type_checkingNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Type_checkingNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.TYPE_CHECKING(self.input(0)))



"""
WARNING: Module TensorNode was generated using fallback option. May contain bugs
"""

class TensorNode(Node):
    """None"""

    title = 'TensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Tensor())



"""
WARNING: Module TensortypeNode was generated using fallback option. May contain bugs
"""

class TensortypeNode(Node):
    """None"""

    title = 'TensortypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.TensorType())



"""
WARNING: Module ThroughputbenchmarkNode was generated using fallback option. May contain bugs
"""

class ThroughputbenchmarkNode(Node):
    """None"""

    title = 'ThroughputbenchmarkNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ThroughputBenchmark())



"""
WARNING: Module TracingstateNode was generated using fallback option. May contain bugs
"""

class TracingstateNode(Node):
    """None"""

    title = 'TracingstateNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.TracingState())



"""
WARNING: Module TupletypeNode was generated using fallback option. May contain bugs
"""

class TupletypeNode(Node):
    """None"""

    title = 'TupletypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.TupleType())



"""
WARNING: Module TypeNode was generated using fallback option. May contain bugs
"""

class TypeNode(Node):
    """None"""

    title = 'TypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Type())



"""
WARNING: Module Use_global_depsNode was generated using fallback option. May contain bugs
"""

class Use_global_depsNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Use_global_depsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.USE_GLOBAL_DEPS(self.input(0)))



"""
WARNING: Module Use_rtld_global_with_libtorchNode was generated using fallback option. May contain bugs
"""

class Use_rtld_global_with_libtorchNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Use_rtld_global_with_libtorchNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.USE_RTLD_GLOBAL_WITH_LIBTORCH(self.input(0)))



"""
WARNING: Module UseNode was generated using fallback option. May contain bugs
"""

class UseNode(Node):
    """None"""

    title = 'UseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Use())



"""
WARNING: Module ValueNode was generated using fallback option. May contain bugs
"""

class ValueNode(Node):
    """None"""

    title = 'ValueNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.Value())



"""
WARNING: Module _cNode was generated using fallback option. May contain bugs
"""

class _cNode(Node):
    """None"""

    title = '_cNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._C())


class _storagebaseNode(Node):
    """None"""

    title = '_storagebaseNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._StorageBase(self.input(0)))



"""
WARNING: Module _vfNode was generated using fallback option. May contain bugs
"""

class _vfNode(Node):
    """None"""

    title = '_vfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._VF())



"""
WARNING: Module _adaptive_avg_pool2dNode was generated using fallback option. May contain bugs
"""

class _adaptive_avg_pool2dNode(Node):
    """None"""

    title = '_adaptive_avg_pool2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._adaptive_avg_pool2d())



"""
WARNING: Module _adaptive_avg_pool3dNode was generated using fallback option. May contain bugs
"""

class _adaptive_avg_pool3dNode(Node):
    """None"""

    title = '_adaptive_avg_pool3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._adaptive_avg_pool3d())



"""
WARNING: Module _add_batch_dimNode was generated using fallback option. May contain bugs
"""

class _add_batch_dimNode(Node):
    """None"""

    title = '_add_batch_dimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._add_batch_dim())



"""
WARNING: Module _add_reluNode was generated using fallback option. May contain bugs
"""

class _add_reluNode(Node):
    """None"""

    title = '_add_reluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._add_relu())



"""
WARNING: Module _add_relu_Node was generated using fallback option. May contain bugs
"""

class _add_relu_Node(Node):
    """None"""

    title = '_add_relu_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._add_relu_())



"""
WARNING: Module _aminmaxNode was generated using fallback option. May contain bugs
"""

class _aminmaxNode(Node):
    """None"""

    title = '_aminmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._aminmax())



"""
WARNING: Module _amp_foreach_non_finite_check_and_unscale_Node was generated using fallback option. May contain bugs
"""

class _amp_foreach_non_finite_check_and_unscale_Node(Node):
    """None"""

    title = '_amp_foreach_non_finite_check_and_unscale_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._amp_foreach_non_finite_check_and_unscale_())



"""
WARNING: Module _amp_update_scale_Node was generated using fallback option. May contain bugs
"""

class _amp_update_scale_Node(Node):
    """None"""

    title = '_amp_update_scale_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._amp_update_scale_())


class _assertNode(Node):
    """A wrapper around Python's assert which is symbolically traceable.
    """

    title = '_assertNode'
    init_inputs = [
        NodeInputBP('condition'),
NodeInputBP('message'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._assert(self.input(0), self.input(1)))



"""
WARNING: Module _assert_asyncNode was generated using fallback option. May contain bugs
"""

class _assert_asyncNode(Node):
    """
_assert_async(tensor) -> void

Asynchronously assert that the contents of tensor are nonzero.  For CPU tensors,
this is equivalent to ``assert tensor`` or ``assert tensor.is_nonzero()``; for
CUDA tensors, we DO NOT synchronize and you may only find out the assertion
failed at a later CUDA kernel launch.  Asynchronous assertion can be helpful for
testing invariants in CUDA tensors without giving up performance.  This function
is NOT intended to be used for regular error checking, as it will trash your CUDA
context if the assert fails (forcing you to restart your PyTorch process.)

Args:
    tensor (Tensor): a one element tensor to test to see if it is nonzero.  Zero
        elements (including False for boolean tensors) cause an assertion failure
        to be raised.
"""

    title = '_assert_asyncNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._assert_async())



"""
WARNING: Module _autograd_functionsNode was generated using fallback option. May contain bugs
"""

class _autograd_functionsNode(Node):
    """None"""

    title = '_autograd_functionsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._autograd_functions())



"""
WARNING: Module _baddbmm_mkl_Node was generated using fallback option. May contain bugs
"""

class _baddbmm_mkl_Node(Node):
    """None"""

    title = '_baddbmm_mkl_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._baddbmm_mkl_())



"""
WARNING: Module _batch_norm_impl_indexNode was generated using fallback option. May contain bugs
"""

class _batch_norm_impl_indexNode(Node):
    """None"""

    title = '_batch_norm_impl_indexNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._batch_norm_impl_index())



"""
WARNING: Module _bmmNode was generated using fallback option. May contain bugs
"""

class _bmmNode(Node):
    """None"""

    title = '_bmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._bmm())



"""
WARNING: Module _cast_byteNode was generated using fallback option. May contain bugs
"""

class _cast_byteNode(Node):
    """None"""

    title = '_cast_byteNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Byte())



"""
WARNING: Module _cast_charNode was generated using fallback option. May contain bugs
"""

class _cast_charNode(Node):
    """None"""

    title = '_cast_charNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Char())



"""
WARNING: Module _cast_doubleNode was generated using fallback option. May contain bugs
"""

class _cast_doubleNode(Node):
    """None"""

    title = '_cast_doubleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Double())



"""
WARNING: Module _cast_floatNode was generated using fallback option. May contain bugs
"""

class _cast_floatNode(Node):
    """None"""

    title = '_cast_floatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Float())



"""
WARNING: Module _cast_halfNode was generated using fallback option. May contain bugs
"""

class _cast_halfNode(Node):
    """None"""

    title = '_cast_halfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Half())



"""
WARNING: Module _cast_intNode was generated using fallback option. May contain bugs
"""

class _cast_intNode(Node):
    """None"""

    title = '_cast_intNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Int())



"""
WARNING: Module _cast_longNode was generated using fallback option. May contain bugs
"""

class _cast_longNode(Node):
    """None"""

    title = '_cast_longNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Long())



"""
WARNING: Module _cast_shortNode was generated using fallback option. May contain bugs
"""

class _cast_shortNode(Node):
    """None"""

    title = '_cast_shortNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cast_Short())



"""
WARNING: Module _catNode was generated using fallback option. May contain bugs
"""

class _catNode(Node):
    """None"""

    title = '_catNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cat())



"""
WARNING: Module _choose_qparams_per_tensorNode was generated using fallback option. May contain bugs
"""

class _choose_qparams_per_tensorNode(Node):
    """None"""

    title = '_choose_qparams_per_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._choose_qparams_per_tensor())



"""
WARNING: Module _classesNode was generated using fallback option. May contain bugs
"""

class _classesNode(Node):
    """None"""

    title = '_classesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._classes())



"""
WARNING: Module _coalesceNode was generated using fallback option. May contain bugs
"""

class _coalesceNode(Node):
    """None"""

    title = '_coalesceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._coalesce())



"""
WARNING: Module _compute_linear_combinationNode was generated using fallback option. May contain bugs
"""

class _compute_linear_combinationNode(Node):
    """None"""

    title = '_compute_linear_combinationNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._compute_linear_combination())



"""
WARNING: Module _conjNode was generated using fallback option. May contain bugs
"""

class _conjNode(Node):
    """None"""

    title = '_conjNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._conj())



"""
WARNING: Module _convolutionNode was generated using fallback option. May contain bugs
"""

class _convolutionNode(Node):
    """None"""

    title = '_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._convolution())



"""
WARNING: Module _convolution_modeNode was generated using fallback option. May contain bugs
"""

class _convolution_modeNode(Node):
    """None"""

    title = '_convolution_modeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._convolution_mode())



"""
WARNING: Module _convolution_nogroupNode was generated using fallback option. May contain bugs
"""

class _convolution_nogroupNode(Node):
    """None"""

    title = '_convolution_nogroupNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._convolution_nogroup())



"""
WARNING: Module _copy_fromNode was generated using fallback option. May contain bugs
"""

class _copy_fromNode(Node):
    """None"""

    title = '_copy_fromNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._copy_from())



"""
WARNING: Module _ctc_lossNode was generated using fallback option. May contain bugs
"""

class _ctc_lossNode(Node):
    """None"""

    title = '_ctc_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._ctc_loss())



"""
WARNING: Module _cudnn_ctc_lossNode was generated using fallback option. May contain bugs
"""

class _cudnn_ctc_lossNode(Node):
    """None"""

    title = '_cudnn_ctc_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cudnn_ctc_loss())



"""
WARNING: Module _cudnn_init_dropout_stateNode was generated using fallback option. May contain bugs
"""

class _cudnn_init_dropout_stateNode(Node):
    """None"""

    title = '_cudnn_init_dropout_stateNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cudnn_init_dropout_state())



"""
WARNING: Module _cudnn_rnnNode was generated using fallback option. May contain bugs
"""

class _cudnn_rnnNode(Node):
    """None"""

    title = '_cudnn_rnnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cudnn_rnn())



"""
WARNING: Module _cudnn_rnn_flatten_weightNode was generated using fallback option. May contain bugs
"""

class _cudnn_rnn_flatten_weightNode(Node):
    """None"""

    title = '_cudnn_rnn_flatten_weightNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cudnn_rnn_flatten_weight())



"""
WARNING: Module _cufft_clear_plan_cacheNode was generated using fallback option. May contain bugs
"""

class _cufft_clear_plan_cacheNode(Node):
    """None"""

    title = '_cufft_clear_plan_cacheNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cufft_clear_plan_cache())



"""
WARNING: Module _cufft_get_plan_cache_max_sizeNode was generated using fallback option. May contain bugs
"""

class _cufft_get_plan_cache_max_sizeNode(Node):
    """None"""

    title = '_cufft_get_plan_cache_max_sizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cufft_get_plan_cache_max_size())



"""
WARNING: Module _cufft_get_plan_cache_sizeNode was generated using fallback option. May contain bugs
"""

class _cufft_get_plan_cache_sizeNode(Node):
    """None"""

    title = '_cufft_get_plan_cache_sizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cufft_get_plan_cache_size())



"""
WARNING: Module _cufft_set_plan_cache_max_sizeNode was generated using fallback option. May contain bugs
"""

class _cufft_set_plan_cache_max_sizeNode(Node):
    """None"""

    title = '_cufft_set_plan_cache_max_sizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cufft_set_plan_cache_max_size())



"""
WARNING: Module _cummax_helperNode was generated using fallback option. May contain bugs
"""

class _cummax_helperNode(Node):
    """None"""

    title = '_cummax_helperNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cummax_helper())



"""
WARNING: Module _cummin_helperNode was generated using fallback option. May contain bugs
"""

class _cummin_helperNode(Node):
    """None"""

    title = '_cummin_helperNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._cummin_helper())



"""
WARNING: Module _debug_has_internal_overlapNode was generated using fallback option. May contain bugs
"""

class _debug_has_internal_overlapNode(Node):
    """None"""

    title = '_debug_has_internal_overlapNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._debug_has_internal_overlap())



"""
WARNING: Module _dim_arangeNode was generated using fallback option. May contain bugs
"""

class _dim_arangeNode(Node):
    """None"""

    title = '_dim_arangeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._dim_arange())



"""
WARNING: Module _dirichlet_gradNode was generated using fallback option. May contain bugs
"""

class _dirichlet_gradNode(Node):
    """None"""

    title = '_dirichlet_gradNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._dirichlet_grad())



"""
WARNING: Module _embedding_bagNode was generated using fallback option. May contain bugs
"""

class _embedding_bagNode(Node):
    """None"""

    title = '_embedding_bagNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._embedding_bag())



"""
WARNING: Module _embedding_bag_forward_onlyNode was generated using fallback option. May contain bugs
"""

class _embedding_bag_forward_onlyNode(Node):
    """None"""

    title = '_embedding_bag_forward_onlyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._embedding_bag_forward_only())



"""
WARNING: Module _empty_affine_quantizedNode was generated using fallback option. May contain bugs
"""

class _empty_affine_quantizedNode(Node):
    """None"""

    title = '_empty_affine_quantizedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._empty_affine_quantized())



"""
WARNING: Module _empty_per_channel_affine_quantizedNode was generated using fallback option. May contain bugs
"""

class _empty_per_channel_affine_quantizedNode(Node):
    """None"""

    title = '_empty_per_channel_affine_quantizedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._empty_per_channel_affine_quantized())



"""
WARNING: Module _euclidean_distNode was generated using fallback option. May contain bugs
"""

class _euclidean_distNode(Node):
    """None"""

    title = '_euclidean_distNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._euclidean_dist())



"""
WARNING: Module _fake_quantize_learnable_per_channel_affineNode was generated using fallback option. May contain bugs
"""

class _fake_quantize_learnable_per_channel_affineNode(Node):
    """None"""

    title = '_fake_quantize_learnable_per_channel_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fake_quantize_learnable_per_channel_affine())



"""
WARNING: Module _fake_quantize_learnable_per_tensor_affineNode was generated using fallback option. May contain bugs
"""

class _fake_quantize_learnable_per_tensor_affineNode(Node):
    """None"""

    title = '_fake_quantize_learnable_per_tensor_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fake_quantize_learnable_per_tensor_affine())



"""
WARNING: Module _fft_c2cNode was generated using fallback option. May contain bugs
"""

class _fft_c2cNode(Node):
    """None"""

    title = '_fft_c2cNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fft_c2c())



"""
WARNING: Module _fft_c2rNode was generated using fallback option. May contain bugs
"""

class _fft_c2rNode(Node):
    """None"""

    title = '_fft_c2rNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fft_c2r())



"""
WARNING: Module _fft_r2cNode was generated using fallback option. May contain bugs
"""

class _fft_r2cNode(Node):
    """None"""

    title = '_fft_r2cNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fft_r2c())



"""
WARNING: Module _foreach_absNode was generated using fallback option. May contain bugs
"""

class _foreach_absNode(Node):
    """None"""

    title = '_foreach_absNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_abs())



"""
WARNING: Module _foreach_abs_Node was generated using fallback option. May contain bugs
"""

class _foreach_abs_Node(Node):
    """None"""

    title = '_foreach_abs_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_abs_())



"""
WARNING: Module _foreach_acosNode was generated using fallback option. May contain bugs
"""

class _foreach_acosNode(Node):
    """None"""

    title = '_foreach_acosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_acos())



"""
WARNING: Module _foreach_acos_Node was generated using fallback option. May contain bugs
"""

class _foreach_acos_Node(Node):
    """None"""

    title = '_foreach_acos_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_acos_())



"""
WARNING: Module _foreach_addNode was generated using fallback option. May contain bugs
"""

class _foreach_addNode(Node):
    """None"""

    title = '_foreach_addNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_add())



"""
WARNING: Module _foreach_add_Node was generated using fallback option. May contain bugs
"""

class _foreach_add_Node(Node):
    """None"""

    title = '_foreach_add_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_add_())



"""
WARNING: Module _foreach_addcdivNode was generated using fallback option. May contain bugs
"""

class _foreach_addcdivNode(Node):
    """None"""

    title = '_foreach_addcdivNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_addcdiv())



"""
WARNING: Module _foreach_addcdiv_Node was generated using fallback option. May contain bugs
"""

class _foreach_addcdiv_Node(Node):
    """None"""

    title = '_foreach_addcdiv_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_addcdiv_())



"""
WARNING: Module _foreach_addcmulNode was generated using fallback option. May contain bugs
"""

class _foreach_addcmulNode(Node):
    """None"""

    title = '_foreach_addcmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_addcmul())



"""
WARNING: Module _foreach_addcmul_Node was generated using fallback option. May contain bugs
"""

class _foreach_addcmul_Node(Node):
    """None"""

    title = '_foreach_addcmul_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_addcmul_())



"""
WARNING: Module _foreach_asinNode was generated using fallback option. May contain bugs
"""

class _foreach_asinNode(Node):
    """None"""

    title = '_foreach_asinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_asin())



"""
WARNING: Module _foreach_asin_Node was generated using fallback option. May contain bugs
"""

class _foreach_asin_Node(Node):
    """None"""

    title = '_foreach_asin_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_asin_())



"""
WARNING: Module _foreach_atanNode was generated using fallback option. May contain bugs
"""

class _foreach_atanNode(Node):
    """None"""

    title = '_foreach_atanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_atan())



"""
WARNING: Module _foreach_atan_Node was generated using fallback option. May contain bugs
"""

class _foreach_atan_Node(Node):
    """None"""

    title = '_foreach_atan_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_atan_())



"""
WARNING: Module _foreach_ceilNode was generated using fallback option. May contain bugs
"""

class _foreach_ceilNode(Node):
    """None"""

    title = '_foreach_ceilNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_ceil())



"""
WARNING: Module _foreach_ceil_Node was generated using fallback option. May contain bugs
"""

class _foreach_ceil_Node(Node):
    """None"""

    title = '_foreach_ceil_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_ceil_())



"""
WARNING: Module _foreach_cosNode was generated using fallback option. May contain bugs
"""

class _foreach_cosNode(Node):
    """None"""

    title = '_foreach_cosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_cos())



"""
WARNING: Module _foreach_cos_Node was generated using fallback option. May contain bugs
"""

class _foreach_cos_Node(Node):
    """None"""

    title = '_foreach_cos_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_cos_())



"""
WARNING: Module _foreach_coshNode was generated using fallback option. May contain bugs
"""

class _foreach_coshNode(Node):
    """None"""

    title = '_foreach_coshNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_cosh())



"""
WARNING: Module _foreach_cosh_Node was generated using fallback option. May contain bugs
"""

class _foreach_cosh_Node(Node):
    """None"""

    title = '_foreach_cosh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_cosh_())



"""
WARNING: Module _foreach_divNode was generated using fallback option. May contain bugs
"""

class _foreach_divNode(Node):
    """None"""

    title = '_foreach_divNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_div())



"""
WARNING: Module _foreach_div_Node was generated using fallback option. May contain bugs
"""

class _foreach_div_Node(Node):
    """None"""

    title = '_foreach_div_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_div_())



"""
WARNING: Module _foreach_erfNode was generated using fallback option. May contain bugs
"""

class _foreach_erfNode(Node):
    """None"""

    title = '_foreach_erfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_erf())



"""
WARNING: Module _foreach_erf_Node was generated using fallback option. May contain bugs
"""

class _foreach_erf_Node(Node):
    """None"""

    title = '_foreach_erf_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_erf_())



"""
WARNING: Module _foreach_erfcNode was generated using fallback option. May contain bugs
"""

class _foreach_erfcNode(Node):
    """None"""

    title = '_foreach_erfcNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_erfc())



"""
WARNING: Module _foreach_erfc_Node was generated using fallback option. May contain bugs
"""

class _foreach_erfc_Node(Node):
    """None"""

    title = '_foreach_erfc_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_erfc_())



"""
WARNING: Module _foreach_expNode was generated using fallback option. May contain bugs
"""

class _foreach_expNode(Node):
    """None"""

    title = '_foreach_expNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_exp())



"""
WARNING: Module _foreach_exp_Node was generated using fallback option. May contain bugs
"""

class _foreach_exp_Node(Node):
    """None"""

    title = '_foreach_exp_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_exp_())



"""
WARNING: Module _foreach_expm1Node was generated using fallback option. May contain bugs
"""

class _foreach_expm1Node(Node):
    """None"""

    title = '_foreach_expm1Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_expm1())



"""
WARNING: Module _foreach_expm1_Node was generated using fallback option. May contain bugs
"""

class _foreach_expm1_Node(Node):
    """None"""

    title = '_foreach_expm1_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_expm1_())



"""
WARNING: Module _foreach_floorNode was generated using fallback option. May contain bugs
"""

class _foreach_floorNode(Node):
    """None"""

    title = '_foreach_floorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_floor())



"""
WARNING: Module _foreach_floor_Node was generated using fallback option. May contain bugs
"""

class _foreach_floor_Node(Node):
    """None"""

    title = '_foreach_floor_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_floor_())



"""
WARNING: Module _foreach_fracNode was generated using fallback option. May contain bugs
"""

class _foreach_fracNode(Node):
    """None"""

    title = '_foreach_fracNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_frac())



"""
WARNING: Module _foreach_frac_Node was generated using fallback option. May contain bugs
"""

class _foreach_frac_Node(Node):
    """None"""

    title = '_foreach_frac_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_frac_())



"""
WARNING: Module _foreach_lgammaNode was generated using fallback option. May contain bugs
"""

class _foreach_lgammaNode(Node):
    """None"""

    title = '_foreach_lgammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_lgamma())



"""
WARNING: Module _foreach_lgamma_Node was generated using fallback option. May contain bugs
"""

class _foreach_lgamma_Node(Node):
    """None"""

    title = '_foreach_lgamma_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_lgamma_())



"""
WARNING: Module _foreach_logNode was generated using fallback option. May contain bugs
"""

class _foreach_logNode(Node):
    """None"""

    title = '_foreach_logNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log())



"""
WARNING: Module _foreach_log10Node was generated using fallback option. May contain bugs
"""

class _foreach_log10Node(Node):
    """None"""

    title = '_foreach_log10Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log10())



"""
WARNING: Module _foreach_log10_Node was generated using fallback option. May contain bugs
"""

class _foreach_log10_Node(Node):
    """None"""

    title = '_foreach_log10_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log10_())



"""
WARNING: Module _foreach_log1pNode was generated using fallback option. May contain bugs
"""

class _foreach_log1pNode(Node):
    """None"""

    title = '_foreach_log1pNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log1p())



"""
WARNING: Module _foreach_log1p_Node was generated using fallback option. May contain bugs
"""

class _foreach_log1p_Node(Node):
    """None"""

    title = '_foreach_log1p_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log1p_())



"""
WARNING: Module _foreach_log2Node was generated using fallback option. May contain bugs
"""

class _foreach_log2Node(Node):
    """None"""

    title = '_foreach_log2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log2())



"""
WARNING: Module _foreach_log2_Node was generated using fallback option. May contain bugs
"""

class _foreach_log2_Node(Node):
    """None"""

    title = '_foreach_log2_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log2_())



"""
WARNING: Module _foreach_log_Node was generated using fallback option. May contain bugs
"""

class _foreach_log_Node(Node):
    """None"""

    title = '_foreach_log_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_log_())



"""
WARNING: Module _foreach_maximumNode was generated using fallback option. May contain bugs
"""

class _foreach_maximumNode(Node):
    """None"""

    title = '_foreach_maximumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_maximum())



"""
WARNING: Module _foreach_minimumNode was generated using fallback option. May contain bugs
"""

class _foreach_minimumNode(Node):
    """None"""

    title = '_foreach_minimumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_minimum())



"""
WARNING: Module _foreach_mulNode was generated using fallback option. May contain bugs
"""

class _foreach_mulNode(Node):
    """None"""

    title = '_foreach_mulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_mul())



"""
WARNING: Module _foreach_mul_Node was generated using fallback option. May contain bugs
"""

class _foreach_mul_Node(Node):
    """None"""

    title = '_foreach_mul_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_mul_())



"""
WARNING: Module _foreach_negNode was generated using fallback option. May contain bugs
"""

class _foreach_negNode(Node):
    """None"""

    title = '_foreach_negNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_neg())



"""
WARNING: Module _foreach_neg_Node was generated using fallback option. May contain bugs
"""

class _foreach_neg_Node(Node):
    """None"""

    title = '_foreach_neg_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_neg_())



"""
WARNING: Module _foreach_reciprocalNode was generated using fallback option. May contain bugs
"""

class _foreach_reciprocalNode(Node):
    """None"""

    title = '_foreach_reciprocalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_reciprocal())



"""
WARNING: Module _foreach_reciprocal_Node was generated using fallback option. May contain bugs
"""

class _foreach_reciprocal_Node(Node):
    """None"""

    title = '_foreach_reciprocal_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_reciprocal_())



"""
WARNING: Module _foreach_roundNode was generated using fallback option. May contain bugs
"""

class _foreach_roundNode(Node):
    """None"""

    title = '_foreach_roundNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_round())



"""
WARNING: Module _foreach_round_Node was generated using fallback option. May contain bugs
"""

class _foreach_round_Node(Node):
    """None"""

    title = '_foreach_round_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_round_())



"""
WARNING: Module _foreach_sigmoidNode was generated using fallback option. May contain bugs
"""

class _foreach_sigmoidNode(Node):
    """None"""

    title = '_foreach_sigmoidNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sigmoid())



"""
WARNING: Module _foreach_sigmoid_Node was generated using fallback option. May contain bugs
"""

class _foreach_sigmoid_Node(Node):
    """None"""

    title = '_foreach_sigmoid_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sigmoid_())



"""
WARNING: Module _foreach_sinNode was generated using fallback option. May contain bugs
"""

class _foreach_sinNode(Node):
    """None"""

    title = '_foreach_sinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sin())



"""
WARNING: Module _foreach_sin_Node was generated using fallback option. May contain bugs
"""

class _foreach_sin_Node(Node):
    """None"""

    title = '_foreach_sin_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sin_())



"""
WARNING: Module _foreach_sinhNode was generated using fallback option. May contain bugs
"""

class _foreach_sinhNode(Node):
    """None"""

    title = '_foreach_sinhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sinh())



"""
WARNING: Module _foreach_sinh_Node was generated using fallback option. May contain bugs
"""

class _foreach_sinh_Node(Node):
    """None"""

    title = '_foreach_sinh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sinh_())



"""
WARNING: Module _foreach_sqrtNode was generated using fallback option. May contain bugs
"""

class _foreach_sqrtNode(Node):
    """None"""

    title = '_foreach_sqrtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sqrt())



"""
WARNING: Module _foreach_sqrt_Node was generated using fallback option. May contain bugs
"""

class _foreach_sqrt_Node(Node):
    """None"""

    title = '_foreach_sqrt_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sqrt_())



"""
WARNING: Module _foreach_subNode was generated using fallback option. May contain bugs
"""

class _foreach_subNode(Node):
    """None"""

    title = '_foreach_subNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sub())



"""
WARNING: Module _foreach_sub_Node was generated using fallback option. May contain bugs
"""

class _foreach_sub_Node(Node):
    """None"""

    title = '_foreach_sub_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_sub_())



"""
WARNING: Module _foreach_tanNode was generated using fallback option. May contain bugs
"""

class _foreach_tanNode(Node):
    """None"""

    title = '_foreach_tanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_tan())



"""
WARNING: Module _foreach_tan_Node was generated using fallback option. May contain bugs
"""

class _foreach_tan_Node(Node):
    """None"""

    title = '_foreach_tan_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_tan_())



"""
WARNING: Module _foreach_tanhNode was generated using fallback option. May contain bugs
"""

class _foreach_tanhNode(Node):
    """None"""

    title = '_foreach_tanhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_tanh())



"""
WARNING: Module _foreach_tanh_Node was generated using fallback option. May contain bugs
"""

class _foreach_tanh_Node(Node):
    """None"""

    title = '_foreach_tanh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_tanh_())



"""
WARNING: Module _foreach_truncNode was generated using fallback option. May contain bugs
"""

class _foreach_truncNode(Node):
    """None"""

    title = '_foreach_truncNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_trunc())



"""
WARNING: Module _foreach_trunc_Node was generated using fallback option. May contain bugs
"""

class _foreach_trunc_Node(Node):
    """None"""

    title = '_foreach_trunc_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_trunc_())



"""
WARNING: Module _foreach_zero_Node was generated using fallback option. May contain bugs
"""

class _foreach_zero_Node(Node):
    """None"""

    title = '_foreach_zero_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._foreach_zero_())



"""
WARNING: Module _fused_dropoutNode was generated using fallback option. May contain bugs
"""

class _fused_dropoutNode(Node):
    """None"""

    title = '_fused_dropoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._fused_dropout())



"""
WARNING: Module _grid_sampler_2d_cpu_fallbackNode was generated using fallback option. May contain bugs
"""

class _grid_sampler_2d_cpu_fallbackNode(Node):
    """None"""

    title = '_grid_sampler_2d_cpu_fallbackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._grid_sampler_2d_cpu_fallback())



"""
WARNING: Module _has_compatible_shallow_copy_typeNode was generated using fallback option. May contain bugs
"""

class _has_compatible_shallow_copy_typeNode(Node):
    """None"""

    title = '_has_compatible_shallow_copy_typeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._has_compatible_shallow_copy_type())


class _import_dotted_nameNode(Node):
    """None"""

    title = '_import_dotted_nameNode'
    init_inputs = [
        NodeInputBP('name'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._import_dotted_name(self.input(0)))



"""
WARNING: Module _index_copy_Node was generated using fallback option. May contain bugs
"""

class _index_copy_Node(Node):
    """None"""

    title = '_index_copy_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._index_copy_())



"""
WARNING: Module _index_put_impl_Node was generated using fallback option. May contain bugs
"""

class _index_put_impl_Node(Node):
    """None"""

    title = '_index_put_impl_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._index_put_impl_())



"""
WARNING: Module _initextensionNode was generated using fallback option. May contain bugs
"""

class _initextensionNode(Node):
    """None"""

    title = '_initextensionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._initExtension())



"""
WARNING: Module _jit_internalNode was generated using fallback option. May contain bugs
"""

class _jit_internalNode(Node):
    """
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

    title = '_jit_internalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._jit_internal())



"""
WARNING: Module _linalg_inv_out_helper_Node was generated using fallback option. May contain bugs
"""

class _linalg_inv_out_helper_Node(Node):
    """None"""

    title = '_linalg_inv_out_helper_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._linalg_inv_out_helper_())



"""
WARNING: Module _linalg_qr_helperNode was generated using fallback option. May contain bugs
"""

class _linalg_qr_helperNode(Node):
    """None"""

    title = '_linalg_qr_helperNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._linalg_qr_helper())



"""
WARNING: Module _linalg_solve_out_helper_Node was generated using fallback option. May contain bugs
"""

class _linalg_solve_out_helper_Node(Node):
    """None"""

    title = '_linalg_solve_out_helper_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._linalg_solve_out_helper_())



"""
WARNING: Module _linalg_utilsNode was generated using fallback option. May contain bugs
"""

class _linalg_utilsNode(Node):
    """Various linear algebra utility methods for internal use.

"""

    title = '_linalg_utilsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._linalg_utils(self.input(0)))


class _load_global_depsNode(Node):
    """None"""

    title = '_load_global_depsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._load_global_deps())



"""
WARNING: Module _lobpcgNode was generated using fallback option. May contain bugs
"""

class _lobpcgNode(Node):
    """Locally Optimal Block Preconditioned Conjugate Gradient methods.
"""

    title = '_lobpcgNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._lobpcg(self.input(0)))



"""
WARNING: Module _log_softmaxNode was generated using fallback option. May contain bugs
"""

class _log_softmaxNode(Node):
    """None"""

    title = '_log_softmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._log_softmax())



"""
WARNING: Module _log_softmax_backward_dataNode was generated using fallback option. May contain bugs
"""

class _log_softmax_backward_dataNode(Node):
    """None"""

    title = '_log_softmax_backward_dataNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._log_softmax_backward_data())



"""
WARNING: Module _logcumsumexpNode was generated using fallback option. May contain bugs
"""

class _logcumsumexpNode(Node):
    """None"""

    title = '_logcumsumexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._logcumsumexp())



"""
WARNING: Module _lowrankNode was generated using fallback option. May contain bugs
"""

class _lowrankNode(Node):
    """Implement various linear algebra algorithms for low rank matrices.
"""

    title = '_lowrankNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._lowrank(self.input(0)))



"""
WARNING: Module _lu_with_infoNode was generated using fallback option. May contain bugs
"""

class _lu_with_infoNode(Node):
    """None"""

    title = '_lu_with_infoNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._lu_with_info())



"""
WARNING: Module _make_dualNode was generated using fallback option. May contain bugs
"""

class _make_dualNode(Node):
    """None"""

    title = '_make_dualNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._make_dual())



"""
WARNING: Module _make_per_channel_quantized_tensorNode was generated using fallback option. May contain bugs
"""

class _make_per_channel_quantized_tensorNode(Node):
    """None"""

    title = '_make_per_channel_quantized_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._make_per_channel_quantized_tensor())



"""
WARNING: Module _make_per_tensor_quantized_tensorNode was generated using fallback option. May contain bugs
"""

class _make_per_tensor_quantized_tensorNode(Node):
    """None"""

    title = '_make_per_tensor_quantized_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._make_per_tensor_quantized_tensor())



"""
WARNING: Module _masked_scaleNode was generated using fallback option. May contain bugs
"""

class _masked_scaleNode(Node):
    """None"""

    title = '_masked_scaleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._masked_scale())



"""
WARNING: Module _mkldnnNode was generated using fallback option. May contain bugs
"""

class _mkldnnNode(Node):
    """None"""

    title = '_mkldnnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._mkldnn())



"""
WARNING: Module _mkldnn_reshapeNode was generated using fallback option. May contain bugs
"""

class _mkldnn_reshapeNode(Node):
    """None"""

    title = '_mkldnn_reshapeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._mkldnn_reshape())



"""
WARNING: Module _mkldnn_transposeNode was generated using fallback option. May contain bugs
"""

class _mkldnn_transposeNode(Node):
    """None"""

    title = '_mkldnn_transposeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._mkldnn_transpose())



"""
WARNING: Module _mkldnn_transpose_Node was generated using fallback option. May contain bugs
"""

class _mkldnn_transpose_Node(Node):
    """None"""

    title = '_mkldnn_transpose_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._mkldnn_transpose_())



"""
WARNING: Module _namedtensor_internalsNode was generated using fallback option. May contain bugs
"""

class _namedtensor_internalsNode(Node):
    """None"""

    title = '_namedtensor_internalsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._namedtensor_internals())



"""
WARNING: Module _nnpack_availableNode was generated using fallback option. May contain bugs
"""

class _nnpack_availableNode(Node):
    """None"""

    title = '_nnpack_availableNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._nnpack_available())



"""
WARNING: Module _nnpack_spatial_convolutionNode was generated using fallback option. May contain bugs
"""

class _nnpack_spatial_convolutionNode(Node):
    """None"""

    title = '_nnpack_spatial_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._nnpack_spatial_convolution())



"""
WARNING: Module _opsNode was generated using fallback option. May contain bugs
"""

class _opsNode(Node):
    """None"""

    title = '_opsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._ops())



"""
WARNING: Module _pack_padded_sequenceNode was generated using fallback option. May contain bugs
"""

class _pack_padded_sequenceNode(Node):
    """None"""

    title = '_pack_padded_sequenceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._pack_padded_sequence())



"""
WARNING: Module _pad_packed_sequenceNode was generated using fallback option. May contain bugs
"""

class _pad_packed_sequenceNode(Node):
    """None"""

    title = '_pad_packed_sequenceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._pad_packed_sequence())



"""
WARNING: Module _remove_batch_dimNode was generated using fallback option. May contain bugs
"""

class _remove_batch_dimNode(Node):
    """None"""

    title = '_remove_batch_dimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._remove_batch_dim())



"""
WARNING: Module _reshape_from_tensorNode was generated using fallback option. May contain bugs
"""

class _reshape_from_tensorNode(Node):
    """None"""

    title = '_reshape_from_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._reshape_from_tensor())



"""
WARNING: Module _rowwise_pruneNode was generated using fallback option. May contain bugs
"""

class _rowwise_pruneNode(Node):
    """None"""

    title = '_rowwise_pruneNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._rowwise_prune())



"""
WARNING: Module _s_whereNode was generated using fallback option. May contain bugs
"""

class _s_whereNode(Node):
    """None"""

    title = '_s_whereNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._s_where())



"""
WARNING: Module _sample_dirichletNode was generated using fallback option. May contain bugs
"""

class _sample_dirichletNode(Node):
    """None"""

    title = '_sample_dirichletNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sample_dirichlet())



"""
WARNING: Module _saturate_weight_to_fp16Node was generated using fallback option. May contain bugs
"""

class _saturate_weight_to_fp16Node(Node):
    """None"""

    title = '_saturate_weight_to_fp16Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._saturate_weight_to_fp16())



"""
WARNING: Module _shape_as_tensorNode was generated using fallback option. May contain bugs
"""

class _shape_as_tensorNode(Node):
    """None"""

    title = '_shape_as_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._shape_as_tensor())



"""
WARNING: Module _sixNode was generated using fallback option. May contain bugs
"""

class _sixNode(Node):
    """None"""

    title = '_sixNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._six())



"""
WARNING: Module _sobol_engine_drawNode was generated using fallback option. May contain bugs
"""

class _sobol_engine_drawNode(Node):
    """None"""

    title = '_sobol_engine_drawNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sobol_engine_draw())



"""
WARNING: Module _sobol_engine_ff_Node was generated using fallback option. May contain bugs
"""

class _sobol_engine_ff_Node(Node):
    """None"""

    title = '_sobol_engine_ff_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sobol_engine_ff_())



"""
WARNING: Module _sobol_engine_initialize_state_Node was generated using fallback option. May contain bugs
"""

class _sobol_engine_initialize_state_Node(Node):
    """None"""

    title = '_sobol_engine_initialize_state_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sobol_engine_initialize_state_())



"""
WARNING: Module _sobol_engine_scramble_Node was generated using fallback option. May contain bugs
"""

class _sobol_engine_scramble_Node(Node):
    """None"""

    title = '_sobol_engine_scramble_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sobol_engine_scramble_())



"""
WARNING: Module _softmaxNode was generated using fallback option. May contain bugs
"""

class _softmaxNode(Node):
    """None"""

    title = '_softmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._softmax())



"""
WARNING: Module _softmax_backward_dataNode was generated using fallback option. May contain bugs
"""

class _softmax_backward_dataNode(Node):
    """None"""

    title = '_softmax_backward_dataNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._softmax_backward_data())



"""
WARNING: Module _sparse_addmmNode was generated using fallback option. May contain bugs
"""

class _sparse_addmmNode(Node):
    """None"""

    title = '_sparse_addmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_addmm())



"""
WARNING: Module _sparse_coo_tensor_unsafeNode was generated using fallback option. May contain bugs
"""

class _sparse_coo_tensor_unsafeNode(Node):
    """None"""

    title = '_sparse_coo_tensor_unsafeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_coo_tensor_unsafe())



"""
WARNING: Module _sparse_csr_tensorNode was generated using fallback option. May contain bugs
"""

class _sparse_csr_tensorNode(Node):
    """
_sparse_csr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, requires_grad=False) -> Tensor

Constructs a :ref:`sparse tensor in CSR (Compressed Sparse Row) <sparse-csr-docs>` with specified
values at the given :attr:`crow_indices` and :attr:`col_indices`. Sparse matrix multiplication operations
in CSR format are typically faster than that for sparse tensors in COO format. Make you have a look
at :ref:`the note on the data type of the indices <sparse-csr-docs>`.

Args:
    crow_indices (array_like): One-dimensional array of size size[0] + 1. The last element
        is the number of non-zeros. This tensor encodes the index in values and col_indices
        depending on where the given row starts. Each successive number in the tensor
        subtracted by the number before it denotes the number of elements in a given row.
    col_indices (array_like): Column co-ordinates of each element in values. Strictly one
        dimensional tensor with the same length as values.
    values (array_list): Initial values for the tensor. Can be a list, tuple, NumPy ``ndarray``, scalar,
        and other types.
    size (list, tuple, :class:`torch.Size`, optional): Size of the sparse tensor. If not provided, the
        size will be inferred as the minimum size big enough to hold all non-zero elements.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`values`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example ::
    >>> crow_indices = [0, 2, 4]
    >>> col_indices = [0, 1, 0, 1]
    >>> values = [1, 2, 3, 4]
    >>> torch._sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
    ...                         torch.tensor(col_indices, dtype=torch.int64),
    ...                         torch.tensor(values), dtype=torch.double)
    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([0, 1, 0, 1]),
           values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
           dtype=torch.float64, layout=torch.sparse_csr)
"""

    title = '_sparse_csr_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_csr_tensor())



"""
WARNING: Module _sparse_log_softmaxNode was generated using fallback option. May contain bugs
"""

class _sparse_log_softmaxNode(Node):
    """None"""

    title = '_sparse_log_softmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_log_softmax())



"""
WARNING: Module _sparse_log_softmax_backward_dataNode was generated using fallback option. May contain bugs
"""

class _sparse_log_softmax_backward_dataNode(Node):
    """None"""

    title = '_sparse_log_softmax_backward_dataNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_log_softmax_backward_data())



"""
WARNING: Module _sparse_mask_helperNode was generated using fallback option. May contain bugs
"""

class _sparse_mask_helperNode(Node):
    """None"""

    title = '_sparse_mask_helperNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_mask_helper())



"""
WARNING: Module _sparse_mmNode was generated using fallback option. May contain bugs
"""

class _sparse_mmNode(Node):
    """None"""

    title = '_sparse_mmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_mm())



"""
WARNING: Module _sparse_softmaxNode was generated using fallback option. May contain bugs
"""

class _sparse_softmaxNode(Node):
    """None"""

    title = '_sparse_softmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_softmax())



"""
WARNING: Module _sparse_softmax_backward_dataNode was generated using fallback option. May contain bugs
"""

class _sparse_softmax_backward_dataNode(Node):
    """None"""

    title = '_sparse_softmax_backward_dataNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_softmax_backward_data())



"""
WARNING: Module _sparse_sparse_matmulNode was generated using fallback option. May contain bugs
"""

class _sparse_sparse_matmulNode(Node):
    """None"""

    title = '_sparse_sparse_matmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_sparse_matmul())



"""
WARNING: Module _sparse_sumNode was generated using fallback option. May contain bugs
"""

class _sparse_sumNode(Node):
    """None"""

    title = '_sparse_sumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._sparse_sum())



"""
WARNING: Module _stackNode was generated using fallback option. May contain bugs
"""

class _stackNode(Node):
    """None"""

    title = '_stackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._stack())



"""
WARNING: Module _standard_gammaNode was generated using fallback option. May contain bugs
"""

class _standard_gammaNode(Node):
    """None"""

    title = '_standard_gammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._standard_gamma())



"""
WARNING: Module _standard_gamma_gradNode was generated using fallback option. May contain bugs
"""

class _standard_gamma_gradNode(Node):
    """None"""

    title = '_standard_gamma_gradNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._standard_gamma_grad())



"""
WARNING: Module _storage_classesNode was generated using fallback option. May contain bugs
"""

class _storage_classesNode(Node):
    """set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements."""

    title = '_storage_classesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._storage_classes())



"""
WARNING: Module _string_classesNode was generated using fallback option. May contain bugs
"""

class _string_classesNode(Node):
    """Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object."""

    title = '_string_classesNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._string_classes(self.input(0)))



"""
WARNING: Module _tensorNode was generated using fallback option. May contain bugs
"""

class _tensorNode(Node):
    """None"""

    title = '_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._tensor())



"""
WARNING: Module _tensor_classesNode was generated using fallback option. May contain bugs
"""

class _tensor_classesNode(Node):
    """set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements."""

    title = '_tensor_classesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._tensor_classes())



"""
WARNING: Module _tensor_strNode was generated using fallback option. May contain bugs
"""

class _tensor_strNode(Node):
    """None"""

    title = '_tensor_strNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._tensor_str())



"""
WARNING: Module _test_serialization_subcmulNode was generated using fallback option. May contain bugs
"""

class _test_serialization_subcmulNode(Node):
    """None"""

    title = '_test_serialization_subcmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._test_serialization_subcmul())



"""
WARNING: Module _trilinearNode was generated using fallback option. May contain bugs
"""

class _trilinearNode(Node):
    """None"""

    title = '_trilinearNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._trilinear())



"""
WARNING: Module _uniqueNode was generated using fallback option. May contain bugs
"""

class _uniqueNode(Node):
    """None"""

    title = '_uniqueNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._unique())



"""
WARNING: Module _unique2Node was generated using fallback option. May contain bugs
"""

class _unique2Node(Node):
    """None"""

    title = '_unique2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._unique2())



"""
WARNING: Module _unpack_dualNode was generated using fallback option. May contain bugs
"""

class _unpack_dualNode(Node):
    """None"""

    title = '_unpack_dualNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._unpack_dual())



"""
WARNING: Module _use_cudnn_ctc_lossNode was generated using fallback option. May contain bugs
"""

class _use_cudnn_ctc_lossNode(Node):
    """None"""

    title = '_use_cudnn_ctc_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._use_cudnn_ctc_loss())



"""
WARNING: Module _use_cudnn_rnn_flatten_weightNode was generated using fallback option. May contain bugs
"""

class _use_cudnn_rnn_flatten_weightNode(Node):
    """None"""

    title = '_use_cudnn_rnn_flatten_weightNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._use_cudnn_rnn_flatten_weight())



"""
WARNING: Module _utilsNode was generated using fallback option. May contain bugs
"""

class _utilsNode(Node):
    """None"""

    title = '_utilsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._utils())



"""
WARNING: Module _utils_internalNode was generated using fallback option. May contain bugs
"""

class _utils_internalNode(Node):
    """None"""

    title = '_utils_internalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._utils_internal())



"""
WARNING: Module _validate_sparse_coo_tensor_argsNode was generated using fallback option. May contain bugs
"""

class _validate_sparse_coo_tensor_argsNode(Node):
    """None"""

    title = '_validate_sparse_coo_tensor_argsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._validate_sparse_coo_tensor_args())



"""
WARNING: Module _vmap_internalsNode was generated using fallback option. May contain bugs
"""

class _vmap_internalsNode(Node):
    """None"""

    title = '_vmap_internalsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._vmap_internals())



"""
WARNING: Module _weight_normNode was generated using fallback option. May contain bugs
"""

class _weight_normNode(Node):
    """None"""

    title = '_weight_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._weight_norm())



"""
WARNING: Module _weight_norm_cuda_interfaceNode was generated using fallback option. May contain bugs
"""

class _weight_norm_cuda_interfaceNode(Node):
    """None"""

    title = '_weight_norm_cuda_interfaceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch._weight_norm_cuda_interface())



"""
WARNING: Module AbsNode was generated using fallback option. May contain bugs
"""

class AbsNode(Node):
    """
abs(input, *, out=None) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
"""

    title = 'AbsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.abs())



"""
WARNING: Module Abs_Node was generated using fallback option. May contain bugs
"""

class Abs_Node(Node):
    """None"""

    title = 'Abs_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.abs_())



"""
WARNING: Module AbsoluteNode was generated using fallback option. May contain bugs
"""

class AbsoluteNode(Node):
    """
absolute(input, *, out=None) -> Tensor

Alias for :func:`torch.abs`
"""

    title = 'AbsoluteNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.absolute())



"""
WARNING: Module AcosNode was generated using fallback option. May contain bugs
"""

class AcosNode(Node):
    """
acos(input, *, out=None) -> Tensor

Computes the inverse cosine of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
    >>> torch.acos(a)
    tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
"""

    title = 'AcosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.acos())



"""
WARNING: Module Acos_Node was generated using fallback option. May contain bugs
"""

class Acos_Node(Node):
    """None"""

    title = 'Acos_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.acos_())



"""
WARNING: Module AcoshNode was generated using fallback option. May contain bugs
"""

class AcoshNode(Node):
    """
acosh(input, *, out=None) -> Tensor

Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

Note:
    The domain of the inverse hyperbolic cosine is `[1, inf)` and values outside this range
    will be mapped to ``NaN``, except for `+ INF` for which the output is mapped to `+ INF`.

.. math::
    \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4).uniform_(1, 2)
    >>> a
    tensor([ 1.3192, 1.9915, 1.9674, 1.7151 ])
    >>> torch.acosh(a)
    tensor([ 0.7791, 1.3120, 1.2979, 1.1341 ])
"""

    title = 'AcoshNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.acosh())



"""
WARNING: Module Acosh_Node was generated using fallback option. May contain bugs
"""

class Acosh_Node(Node):
    """None"""

    title = 'Acosh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.acosh_())



"""
WARNING: Module Adaptive_avg_pool1dNode was generated using fallback option. May contain bugs
"""

class Adaptive_avg_pool1dNode(Node):
    """
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
"""

    title = 'Adaptive_avg_pool1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.adaptive_avg_pool1d())



"""
WARNING: Module Adaptive_max_pool1dNode was generated using fallback option. May contain bugs
"""

class Adaptive_max_pool1dNode(Node):
    """None"""

    title = 'Adaptive_max_pool1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.adaptive_max_pool1d())



"""
WARNING: Module AddNode was generated using fallback option. May contain bugs
"""

class AddNode(Node):
    """
add(input, other, *, out=None) -> Tensor

Adds the scalar :attr:`other` to each element of the input :attr:`input`
and returns a new resulting tensor.

.. math::
    \text{out} = \text{input} + \text{other}

If :attr:`input` is of type FloatTensor or DoubleTensor, :attr:`other` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the input tensor.
    other (Number): the number to be added to each element of :attr:`input`

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

.. function:: add(input, other, *, alpha=1, out=None) -> Tensor

Each element of the tensor :attr:`other` is multiplied by the scalar
:attr:`alpha` and added to each element of the tensor :attr:`input`.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out} = \text{input} + \text{alpha} \times \text{other}

If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`alpha` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    alpha (Number): the scalar multiplier for :attr:`other`
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(a, b, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
"""

    title = 'AddNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.add())



"""
WARNING: Module AddbmmNode was generated using fallback option. May contain bugs
"""

class AddbmmNode(Node):
    """
addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

.. math::
    out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
must be real numbers, otherwise they should be integers.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Args:
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): matrix to be added
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)
    tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
            [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
            [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])
"""

    title = 'AddbmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addbmm())



"""
WARNING: Module AddcdivNode was generated using fallback option. May contain bugs
"""

class AddcdivNode(Node):
    """
addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiply the result by the scalar :attr:`value` and add it to :attr:`input`.

.. warning::
    Integer division with addcdiv is no longer supported, and in a future
    release addcdiv will perform a true division of tensor1 and tensor2.
    The historic addcdiv behavior can be implemented as
    (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype)
    for integer inputs and as (input + value * tensor1 / tensor2) for float inputs.
    The future addcdiv behavior is just the latter implementation:
    (input + value * tensor1 / tensor2), for all dtypes.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}


The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor

Keyword args:
    value (Number, optional): multiplier for :math:`\text{tensor1} / \text{tensor2}`
    out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, t1, t2, value=0.1)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
"""

    title = 'AddcdivNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addcdiv())



"""
WARNING: Module AddcmulNode was generated using fallback option. May contain bugs
"""

class AddcmulNode(Node):
    """
addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
and add it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i

The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied

Keyword args:
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
    out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, t1, t2, value=0.1)
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
"""

    title = 'AddcmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addcmul())



"""
WARNING: Module AddmmNode was generated using fallback option. May contain bugs
"""

class AddmmNode(Node):
    """
addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Args:
    input (Tensor): matrix to be added
    mat1 (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])
"""

    title = 'AddmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addmm())



"""
WARNING: Module AddmvNode was generated using fallback option. May contain bugs
"""

class AddmvNode(Node):
    """
addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`input` is added to the final result.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
:attr:`out` will be 1-D tensor of size `n`.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers

Args:
    input (Tensor): vector to be added
    mat (Tensor): matrix to be matrix multiplied
    vec (Tensor): vector to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.3768, -5.5565])
"""

    title = 'AddmvNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addmv())



"""
WARNING: Module Addmv_Node was generated using fallback option. May contain bugs
"""

class Addmv_Node(Node):
    """None"""

    title = 'Addmv_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addmv_())



"""
WARNING: Module AddrNode was generated using fallback option. May contain bugs
"""

class AddrNode(Node):
    """
addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`input`.

Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
:attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
of size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a matrix of size
:math:`(n \times m)` and :attr:`out` will be a matrix of size
:math:`(n \times m)`.

Args:
    input (Tensor): matrix to be added
    vec1 (Tensor): the first vector of the outer product
    vec2 (Tensor): the second vector of the outer product

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`\text{vec1} \otimes \text{vec2}` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

    >>> vec1 = torch.arange(1., 4.)
    >>> vec2 = torch.arange(1., 3.)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
    tensor([[ 1.,  2.],
            [ 2.,  4.],
            [ 3.,  6.]])
"""

    title = 'AddrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.addr())



"""
WARNING: Module Affine_grid_generatorNode was generated using fallback option. May contain bugs
"""

class Affine_grid_generatorNode(Node):
    """None"""

    title = 'Affine_grid_generatorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.affine_grid_generator())


class Align_tensorsNode(Node):
    """None"""

    title = 'Align_tensorsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.align_tensors())



"""
WARNING: Module AllNode was generated using fallback option. May contain bugs
"""

class AllNode(Node):
    """
all(input) -> Tensor

Tests if all elements in :attr:`input` evaluate to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.all(a)
    tensor(False, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.all(a)
    tensor(False)

.. function:: all(input, dim, keepdim=False, *, out=None) -> Tensor

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if all elements in the row evaluate to `True` and `False` otherwise.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=torch.bool)
    >>> torch.all(a, dim=1)
    tensor([ True, False,  True,  True], dtype=torch.bool)
    >>> torch.all(a, dim=0)
    tensor([ True, False], dtype=torch.bool)
"""

    title = 'AllNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.all())



"""
WARNING: Module AllcloseNode was generated using fallback option. May contain bugs
"""

class AllcloseNode(Node):
    """
allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool

This function checks if all :attr:`input` and :attr:`other` satisfy the condition:

.. math::
    \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
`numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_

Args:
    input (Tensor): first tensor to compare
    other (Tensor): second tensor to compare
    atol (float, optional): absolute tolerance. Default: 1e-08
    rtol (float, optional): relative tolerance. Default: 1e-05
    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

Example::

    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
"""

    title = 'AllcloseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.allclose())



"""
WARNING: Module Alpha_dropoutNode was generated using fallback option. May contain bugs
"""

class Alpha_dropoutNode(Node):
    """None"""

    title = 'Alpha_dropoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.alpha_dropout())



"""
WARNING: Module Alpha_dropout_Node was generated using fallback option. May contain bugs
"""

class Alpha_dropout_Node(Node):
    """None"""

    title = 'Alpha_dropout_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.alpha_dropout_())



"""
WARNING: Module AmaxNode was generated using fallback option. May contain bugs
"""

class AmaxNode(Node):
    """
amax(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the maximum value of each slice of the :attr:`input` tensor in the given
dimension(s) :attr:`dim`.

.. note::
    The difference between ``max``/``min`` and ``amax``/``amin`` is:
        - ``amax``/``amin`` supports reducing on multiple dimensions,
        - ``amax``/``amin`` does not return indices,
        - ``amax``/``amin`` evenly distributes gradient between equal values,
          while ``max(dim)``/``min(dim)`` propagates gradient only to a single
          index in the source tensor.

If :attr:`keepdim is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim`s are squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
            [-0.7158,  1.1775,  2.0992,  0.4817],
            [-0.0053,  0.0164, -1.3738, -0.0507],
            [ 1.9700,  1.1106, -1.0318, -1.0816]])
    >>> torch.amax(a, 1)
    tensor([1.4878, 2.0992, 0.0164, 1.9700])
"""

    title = 'AmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.amax())



"""
WARNING: Module AminNode was generated using fallback option. May contain bugs
"""

class AminNode(Node):
    """
amin(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the minimum value of each slice of the :attr:`input` tensor in the given
dimension(s) :attr:`dim`.

.. note::
    The difference between ``max``/``min`` and ``amax``/``amin`` is:
        - ``amax``/``amin`` supports reducing on multiple dimensions,
        - ``amax``/``amin`` does not return indices,
        - ``amax``/``amin`` evenly distributes gradient between equal values,
          while ``max(dim)``/``min(dim)`` propagates gradient only to a single
          index in the source tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension(s) :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim`s are squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having fewer dimensions than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.6451, -0.4866,  0.2987, -1.3312],
            [-0.5744,  1.2980,  1.8397, -0.2713],
            [ 0.9128,  0.9214, -1.7268, -0.2995],
            [ 0.9023,  0.4853,  0.9075, -1.6165]])
    >>> torch.amin(a, 1)
    tensor([-1.3312, -0.5744, -1.7268, -1.6165])
"""

    title = 'AminNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.amin())



"""
WARNING: Module AngleNode was generated using fallback option. May contain bugs
"""

class AngleNode(Node):
    """
angle(input, *, out=None) -> Tensor

Computes the element-wise angle (in radians) of the given :attr:`input` tensor.

.. math::
    \text{out}_{i} = angle(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

.. note:: Starting in PyTorch 1.8, angle returns pi for negative real numbers,
          zero for non-negative real numbers, and propagates NaNs. Previously
          the function would return zero for all real numbers and not propagate
          floating-point NaNs.

Example::

    >>> torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159
    tensor([ 135.,  135,  -45])
"""

    title = 'AngleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.angle())



"""
WARNING: Module AnyNode was generated using fallback option. May contain bugs
"""

class AnyNode(Node):
    """
any(input) -> Tensor

Args:
    input (Tensor): the input tensor.

Tests if any element in :attr:`input` evaluates to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.any(a)
    tensor(True, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.any(a)
    tensor(True)

.. function:: any(input, dim, keepdim=False, *, out=None) -> Tensor

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if any element in the row evaluate to `True` and `False` otherwise.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> torch.any(a, 1)
    tensor([ True,  True,  True, False])
    >>> torch.any(a, 0)
    tensor([True, True])
"""

    title = 'AnyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.any())



"""
WARNING: Module ArangeNode was generated using fallback option. May contain bugs
"""

class ArangeNode(Node):
    """
arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
with values from the interval ``[start, end)`` taken with common difference
:attr:`step` beginning from `start`.

Note that non-integer :attr:`step` is subject to floating point rounding errors when
comparing against :attr:`end`; to avoid inconsistency, we advise adding a small epsilon to :attr:`end`
in such cases.

.. math::
    \text{out}_{{i+1}} = \text{out}_{i} + \text{step}

Args:
    start (Number): the starting value for the set of points. Default: ``0``.
    end (Number): the ending value for the set of points
    step (Number): the gap between each pair of adjacent points. Default: ``1``.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). If `dtype` is not given, infer the data type from the other input
        arguments. If any of `start`, `end`, or `stop` are floating-point, the
        `dtype` is inferred to be the default dtype, see
        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
        be `torch.int64`.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.arange(5)
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
"""

    title = 'ArangeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arange())



"""
WARNING: Module ArccosNode was generated using fallback option. May contain bugs
"""

class ArccosNode(Node):
    """
arccos(input, *, out=None) -> Tensor

Alias for :func:`torch.acos`.
"""

    title = 'ArccosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arccos())



"""
WARNING: Module Arccos_Node was generated using fallback option. May contain bugs
"""

class Arccos_Node(Node):
    """None"""

    title = 'Arccos_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arccos_())



"""
WARNING: Module ArccoshNode was generated using fallback option. May contain bugs
"""

class ArccoshNode(Node):
    """
arccosh(input, *, out=None) -> Tensor

Alias for :func:`torch.acosh`.
"""

    title = 'ArccoshNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arccosh())



"""
WARNING: Module Arccosh_Node was generated using fallback option. May contain bugs
"""

class Arccosh_Node(Node):
    """None"""

    title = 'Arccosh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arccosh_())



"""
WARNING: Module ArcsinNode was generated using fallback option. May contain bugs
"""

class ArcsinNode(Node):
    """
arcsin(input, *, out=None) -> Tensor

Alias for :func:`torch.asin`.
"""

    title = 'ArcsinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arcsin())



"""
WARNING: Module Arcsin_Node was generated using fallback option. May contain bugs
"""

class Arcsin_Node(Node):
    """None"""

    title = 'Arcsin_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arcsin_())



"""
WARNING: Module ArcsinhNode was generated using fallback option. May contain bugs
"""

class ArcsinhNode(Node):
    """
arcsinh(input, *, out=None) -> Tensor

Alias for :func:`torch.asinh`.
"""

    title = 'ArcsinhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arcsinh())



"""
WARNING: Module Arcsinh_Node was generated using fallback option. May contain bugs
"""

class Arcsinh_Node(Node):
    """None"""

    title = 'Arcsinh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arcsinh_())



"""
WARNING: Module ArctanNode was generated using fallback option. May contain bugs
"""

class ArctanNode(Node):
    """
arctan(input, *, out=None) -> Tensor

Alias for :func:`torch.atan`.
"""

    title = 'ArctanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arctan())



"""
WARNING: Module Arctan_Node was generated using fallback option. May contain bugs
"""

class Arctan_Node(Node):
    """None"""

    title = 'Arctan_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arctan_())



"""
WARNING: Module ArctanhNode was generated using fallback option. May contain bugs
"""

class ArctanhNode(Node):
    """
arctanh(input, *, out=None) -> Tensor

Alias for :func:`torch.atanh`.
"""

    title = 'ArctanhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arctanh())



"""
WARNING: Module Arctanh_Node was generated using fallback option. May contain bugs
"""

class Arctanh_Node(Node):
    """None"""

    title = 'Arctanh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.arctanh_())


class Are_deterministic_algorithms_enabledNode(Node):
    """Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """

    title = 'Are_deterministic_algorithms_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.are_deterministic_algorithms_enabled())



"""
WARNING: Module ArgmaxNode was generated using fallback option. May contain bugs
"""

class ArgmaxNode(Node):
    """
argmax(input) -> LongTensor

Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

.. note:: If there are multiple maximal values then the indices of the first maximal value are returned.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)

.. function:: argmax(input, dim, keepdim=False) -> LongTensor

Returns the indices of the maximum values of a tensor across a dimension.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce. If ``None``, the argmax of the flattened input is returned.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])
"""

    title = 'ArgmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.argmax())



"""
WARNING: Module ArgminNode was generated using fallback option. May contain bugs
"""

class ArgminNode(Node):
    """
argmin(input, dim=None, keepdim=False) -> LongTensor

Returns the indices of the minimum value(s) of the flattened tensor or along a dimension

This is the second value returned by :meth:`torch.min`. See its
documentation for the exact semantics of this method.

.. note:: If there are multiple minimal values then the indices of the first minimal value are returned.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce. If ``None``, the argmin of the flattened input is returned.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a)
    tensor(13)
    >>> torch.argmin(a, dim=1)
    tensor([ 2,  1,  3,  1])
    >>> torch.argmin(a, dim=1, keepdim=True)
    tensor([[2],
            [1],
            [3],
            [1]])
"""

    title = 'ArgminNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.argmin())



"""
WARNING: Module ArgsortNode was generated using fallback option. May contain bugs
"""

class ArgsortNode(Node):
    """
argsort(input, dim=-1, descending=False) -> LongTensor

Returns the indices that sort a tensor along a given dimension in ascending
order by value.

This is the second value returned by :meth:`torch.sort`.  See its documentation
for the exact semantics of this method.

Args:
    input (Tensor): the input tensor.
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
            [ 0.1598,  0.0788, -0.0745, -1.2700],
            [ 1.2208,  1.0722, -0.7064,  1.2564],
            [ 0.0669, -0.2318, -0.8229, -0.9280]])


    >>> torch.argsort(a, dim=1)
    tensor([[2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [3, 2, 1, 0]])
"""

    title = 'ArgsortNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.argsort())



"""
WARNING: Module As_stridedNode was generated using fallback option. May contain bugs
"""

class As_stridedNode(Node):
    """
as_strided(input, size, stride, storage_offset=0) -> Tensor

Create a view of an existing `torch.Tensor` :attr:`input` with specified
:attr:`size`, :attr:`stride` and :attr:`storage_offset`.

.. warning::
    More than one element of a created tensor may refer to a single memory
    location. As a result, in-place operations (especially ones that are
    vectorized) may result in incorrect behavior. If you need to write to
    the tensors, please clone them first.

    Many PyTorch functions, which return a view of a tensor, are internally
    implemented with this function. Those functions, like
    :meth:`torch.Tensor.expand`, are easier to read and are therefore more
    advisable to use.


Args:
    input (Tensor): the input tensor.
    size (tuple or ints): the shape of the output tensor
    stride (tuple or ints): the stride of the output tensor
    storage_offset (int, optional): the offset in the underlying storage of the output tensor

Example::

    >>> x = torch.randn(3, 3)
    >>> x
    tensor([[ 0.9039,  0.6291,  1.0795],
            [ 0.1586,  2.1939, -0.4900],
            [-0.1909, -0.7503,  1.9355]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2))
    >>> t
    tensor([[0.9039, 1.0795],
            [0.6291, 0.1586]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
    tensor([[0.6291, 0.1586],
            [1.0795, 2.1939]])
"""

    title = 'As_stridedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.as_strided())



"""
WARNING: Module As_strided_Node was generated using fallback option. May contain bugs
"""

class As_strided_Node(Node):
    """None"""

    title = 'As_strided_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.as_strided_())



"""
WARNING: Module As_tensorNode was generated using fallback option. May contain bugs
"""

class As_tensorNode(Node):
    """
as_tensor(data, dtype=None, device=None) -> Tensor

Convert the data into a `torch.Tensor`. If the data is already a `Tensor` with the same `dtype` and `device`,
no copy will be performed, otherwise a new `Tensor` will be returned with computational graph retained if data
`Tensor` has ``requires_grad=True``. Similarly, if the data is an ``ndarray`` of the corresponding `dtype` and
the `device` is the cpu, no copy will be performed.

Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
"""

    title = 'As_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.as_tensor())



"""
WARNING: Module AsinNode was generated using fallback option. May contain bugs
"""

class AsinNode(Node):
    """
asin(input, *, out=None) -> Tensor

Returns a new tensor with the arcsine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5962,  1.4985, -0.4396,  1.4525])
    >>> torch.asin(a)
    tensor([-0.6387,     nan, -0.4552,     nan])
"""

    title = 'AsinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.asin())



"""
WARNING: Module Asin_Node was generated using fallback option. May contain bugs
"""

class Asin_Node(Node):
    """None"""

    title = 'Asin_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.asin_())



"""
WARNING: Module AsinhNode was generated using fallback option. May contain bugs
"""

class AsinhNode(Node):
    """
asinh(input, *, out=None) -> Tensor

Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ])
    >>> torch.asinh(a)
    tensor([ 0.1599, -1.1534, -0.9435, -0.8990 ])
"""

    title = 'AsinhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.asinh())



"""
WARNING: Module Asinh_Node was generated using fallback option. May contain bugs
"""

class Asinh_Node(Node):
    """None"""

    title = 'Asinh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.asinh_())



"""
WARNING: Module AtanNode was generated using fallback option. May contain bugs
"""

class AtanNode(Node):
    """
atan(input, *, out=None) -> Tensor

Returns a new tensor with the arctangent  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    >>> torch.atan(a)
    tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
"""

    title = 'AtanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atan())



"""
WARNING: Module Atan2Node was generated using fallback option. May contain bugs
"""

class Atan2Node(Node):
    """
atan2(input, other, *, out=None) -> Tensor

Element-wise arctangent of :math:`\text{input}_{i} / \text{other}_{i}`
with consideration of the quadrant. Returns a new tensor with the signed angles
in radians between vector :math:`(\text{other}_{i}, \text{input}_{i})`
and vector :math:`(1, 0)`. (Note that :math:`\text{other}_{i}`, the second
parameter, is the x-coordinate, while :math:`\text{input}_{i}`, the first
parameter, is the y-coordinate.)

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
    >>> torch.atan2(a, torch.randn(4))
    tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
"""

    title = 'Atan2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atan2())



"""
WARNING: Module Atan_Node was generated using fallback option. May contain bugs
"""

class Atan_Node(Node):
    """None"""

    title = 'Atan_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atan_())



"""
WARNING: Module AtanhNode was generated using fallback option. May contain bugs
"""

class AtanhNode(Node):
    """
atanh(input, *, out=None) -> Tensor

Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

Note:
    The domain of the inverse hyperbolic tangent is `(-1, 1)` and values outside this range
    will be mapped to ``NaN``, except for the values `1` and `-1` for which the output is
    mapped to `+/-INF` respectively.

.. math::
    \text{out}_{i} = \tanh^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4).uniform_(-1, 1)
    >>> a
    tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ])
    >>> torch.atanh(a)
    tensor([ -1.7253, 0.3060, -1.2899, -0.1893 ])
"""

    title = 'AtanhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atanh())



"""
WARNING: Module Atanh_Node was generated using fallback option. May contain bugs
"""

class Atanh_Node(Node):
    """None"""

    title = 'Atanh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atanh_())


class Atleast_1dNode(Node):
    """
    Returns a 1-dimensional view of each input tensor with zero dimensions.
    Input tensors with one or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.randn(2)
        >>> x
        tensor([1.4584, 0.7583])
        >>> torch.atleast_1d(x)
        tensor([1.4584, 0.7583])
        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_1d(x)
        tensor([1.])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_1d((x,y))
        (tensor([0.5000]), tensor([1.]))
    """

    title = 'Atleast_1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atleast_1d())


class Atleast_2dNode(Node):
    """
    Returns a 2-dimensional view of each input tensor with zero dimensions.
    Input tensors with two or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_2d(x)
        tensor([[1.]])
        >>> x = torch.randn(2,2)
        >>> x
        tensor([[2.2086, 2.5165],
                [0.1757, 0.5194]])
        >>> torch.atleast_2d(x)
        tensor([[2.2086, 2.5165],
                [0.1757, 0.5194]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_2d((x,y))
        (tensor([[0.5000]]), tensor([[1.]]))
    """

    title = 'Atleast_2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atleast_2d())


class Atleast_3dNode(Node):
    """
    Returns a 3-dimensional view of each input tensor with zero dimensions.
    Input tensors with three or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example:

        >>> x = torch.tensor(0.5)
        >>> x
        tensor(0.5000)
        >>> torch.atleast_3d(x)
        tensor([[[0.5000]]])
        >>> y = torch.randn(2,2)
        >>> y
        tensor([[-0.8079,  0.7460],
                [-1.1647,  1.4734]])
        >>> torch.atleast_3d(y)
        tensor([[[-0.8079],
                [ 0.7460]],
                <BLANKLINE>
                [[-1.1647],
                [ 1.4734]]])
        >>> x = torch.randn(1,1,1)
        >>> x
        tensor([[[-1.5689]]])
        >>> torch.atleast_3d(x)
        tensor([[[-1.5689]]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_3d((x,y))
        (tensor([[[0.5000]]]), tensor([[[1.]]]))
    """

    title = 'Atleast_3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.atleast_3d())



"""
WARNING: Module AttrNode was generated using fallback option. May contain bugs
"""

class AttrNode(Node):
    """str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'."""

    title = 'AttrNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.attr(self.input(0)))



"""
WARNING: Module Autocast_decrement_nestingNode was generated using fallback option. May contain bugs
"""

class Autocast_decrement_nestingNode(Node):
    """None"""

    title = 'Autocast_decrement_nestingNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.autocast_decrement_nesting())



"""
WARNING: Module Autocast_increment_nestingNode was generated using fallback option. May contain bugs
"""

class Autocast_increment_nestingNode(Node):
    """None"""

    title = 'Autocast_increment_nestingNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.autocast_increment_nesting())



"""
WARNING: Module AutogradNode was generated using fallback option. May contain bugs
"""

class AutogradNode(Node):
    """
``torch.autograd`` provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

    title = 'AutogradNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.autograd())



"""
WARNING: Module Avg_pool1dNode was generated using fallback option. May contain bugs
"""

class Avg_pool1dNode(Node):
    """
avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See :class:`~torch.nn.AvgPool1d` for details and output shape.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    kernel_size: the size of the window. Can be a single number or a
      tuple `(kW,)`
    stride: the stride of the window. Can be a single number or a tuple
      `(sW,)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padW,)`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the
        output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``

Examples::

    >>> # pool of square window of size=3, stride=2
    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    tensor([[[ 2.,  4.,  6.]]])

"""

    title = 'Avg_pool1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.avg_pool1d())



"""
WARNING: Module BackendsNode was generated using fallback option. May contain bugs
"""

class BackendsNode(Node):
    """None"""

    title = 'BackendsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.backends())



"""
WARNING: Module BaddbmmNode was generated using fallback option. May contain bugs
"""

class BaddbmmNode(Node):
    """
baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a
:math:`(b \times n \times p)` tensor and :attr:`out` will be a
:math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
same as the scaling factors used in :meth:`torch.addbmm`.

.. math::
    \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Args:
    input (Tensor): the tensor to be added
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`\text{batch1} \mathbin{@} \text{batch2}` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
"""

    title = 'BaddbmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.baddbmm())



"""
WARNING: Module Bartlett_windowNode was generated using fallback option. May contain bugs
"""

class Bartlett_windowNode(Node):
    """
bartlett_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Bartlett window function.

.. math::
    w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
    \end{cases},

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.bartlett_window(L, periodic=True)`` equal to
``torch.bartlett_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{window\_length},)` containing the window

"""

    title = 'Bartlett_windowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bartlett_window())



"""
WARNING: Module Batch_normNode was generated using fallback option. May contain bugs
"""

class Batch_normNode(Node):
    """None"""

    title = 'Batch_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm())



"""
WARNING: Module Batch_norm_backward_elemtNode was generated using fallback option. May contain bugs
"""

class Batch_norm_backward_elemtNode(Node):
    """None"""

    title = 'Batch_norm_backward_elemtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_backward_elemt())



"""
WARNING: Module Batch_norm_backward_reduceNode was generated using fallback option. May contain bugs
"""

class Batch_norm_backward_reduceNode(Node):
    """None"""

    title = 'Batch_norm_backward_reduceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_backward_reduce())



"""
WARNING: Module Batch_norm_elemtNode was generated using fallback option. May contain bugs
"""

class Batch_norm_elemtNode(Node):
    """None"""

    title = 'Batch_norm_elemtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_elemt())



"""
WARNING: Module Batch_norm_gather_statsNode was generated using fallback option. May contain bugs
"""

class Batch_norm_gather_statsNode(Node):
    """None"""

    title = 'Batch_norm_gather_statsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_gather_stats())



"""
WARNING: Module Batch_norm_gather_stats_with_countsNode was generated using fallback option. May contain bugs
"""

class Batch_norm_gather_stats_with_countsNode(Node):
    """None"""

    title = 'Batch_norm_gather_stats_with_countsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_gather_stats_with_counts())



"""
WARNING: Module Batch_norm_statsNode was generated using fallback option. May contain bugs
"""

class Batch_norm_statsNode(Node):
    """None"""

    title = 'Batch_norm_statsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_stats())



"""
WARNING: Module Batch_norm_update_statsNode was generated using fallback option. May contain bugs
"""

class Batch_norm_update_statsNode(Node):
    """None"""

    title = 'Batch_norm_update_statsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.batch_norm_update_stats())



"""
WARNING: Module BernoulliNode was generated using fallback option. May contain bugs
"""

class BernoulliNode(Node):
    """
bernoulli(input, *, generator=None, out=None) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 \leq \text{input}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output tensor will draw a
value :math:`1` according to the :math:`\text{i}^{th}` probability value given
in :attr:`input`.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})

The returned :attr:`out` tensor only has values 0 or 1 and is of the same
shape as :attr:`input`.

:attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
point ``dtype``.

Args:
    input (Tensor): the input tensor of probability values for the Bernoulli distribution

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.1737,  0.0950,  0.3609],
            [ 0.7148,  0.0289,  0.2676],
            [ 0.9456,  0.8937,  0.7202]])
    >>> torch.bernoulli(a)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  1.,  1.]])

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
"""

    title = 'BernoulliNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bernoulli())



"""
WARNING: Module Bfloat16Node was generated using fallback option. May contain bugs
"""

class Bfloat16Node(Node):
    """None"""

    title = 'Bfloat16Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bfloat16())



"""
WARNING: Module BilinearNode was generated using fallback option. May contain bugs
"""

class BilinearNode(Node):
    """None"""

    title = 'BilinearNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bilinear())



"""
WARNING: Module Binary_cross_entropy_with_logitsNode was generated using fallback option. May contain bugs
"""

class Binary_cross_entropy_with_logitsNode(Node):
    """None"""

    title = 'Binary_cross_entropy_with_logitsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.binary_cross_entropy_with_logits())



"""
WARNING: Module BincountNode was generated using fallback option. May contain bugs
"""

class BincountNode(Node):
    """
bincount(input, weights=None, minlength=0) -> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
:attr:`input` unless :attr:`input` is empty, in which case the result is a
tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
:attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
:attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
``out[n] += weights[i]`` if :attr:`weights` is specified else
``out[n] += 1``.

Note:
    This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

Arguments:
    input (Tensor): 1-d int tensor
    weights (Tensor): optional, weight for each value in the input tensor.
        Should be of same size as input tensor.
    minlength (int): optional, minimum number of bins. Should be non-negative.

Returns:
    output (Tensor): a tensor of shape ``Size([max(input) + 1])`` if
    :attr:`input` is non-empty, else ``Size(0)``

Example::

    >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
    >>> weights = torch.linspace(0, 1, steps=5)
    >>> input, weights
    (tensor([4, 3, 6, 3, 4]),
     tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])

    >>> torch.bincount(input)
    tensor([0, 0, 0, 2, 2, 0, 1])

    >>> input.bincount(weights)
    tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
"""

    title = 'BincountNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bincount())



"""
WARNING: Module BinomialNode was generated using fallback option. May contain bugs
"""

class BinomialNode(Node):
    """None"""

    title = 'BinomialNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.binomial())



"""
WARNING: Module Bitwise_andNode was generated using fallback option. May contain bugs
"""

class Bitwise_andNode(Node):
    """
bitwise_and(input, other, *, out=None) -> Tensor

Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical AND.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example:

    >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([1, 0,  3], dtype=torch.int8)
    >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ False, True, False])
"""

    title = 'Bitwise_andNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bitwise_and())



"""
WARNING: Module Bitwise_notNode was generated using fallback option. May contain bugs
"""

class Bitwise_notNode(Node):
    """
bitwise_not(input, *, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example:

    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
"""

    title = 'Bitwise_notNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bitwise_not())



"""
WARNING: Module Bitwise_orNode was generated using fallback option. May contain bugs
"""

class Bitwise_orNode(Node):
    """
bitwise_or(input, other, *, out=None) -> Tensor

Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical OR.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example:

    >>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -2,  3], dtype=torch.int8)
    >>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, True, False])
"""

    title = 'Bitwise_orNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bitwise_or())



"""
WARNING: Module Bitwise_xorNode was generated using fallback option. May contain bugs
"""

class Bitwise_xorNode(Node):
    """
bitwise_xor(input, other, *, out=None) -> Tensor

Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical XOR.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example:

    >>> torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-2, -2,  0], dtype=torch.int8)
    >>> torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, False, False])
"""

    title = 'Bitwise_xorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bitwise_xor())



"""
WARNING: Module Blackman_windowNode was generated using fallback option. May contain bugs
"""

class Blackman_windowNode(Node):
    """
blackman_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Blackman window function.

.. math::
    w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.blackman_window(L, periodic=True)`` equal to
``torch.blackman_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{window\_length},)` containing the window

"""

    title = 'Blackman_windowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.blackman_window())


class Block_diagNode(Node):
    """Create a block diagonal matrix from provided tensors.

    Args:
        *tensors: One or more tensors with 0, 1, or 2 dimensions.

    Returns:
        Tensor: A 2 dimensional tensor with all the input tensors arranged in
        order such that their upper left and lower right corners are
        diagonally adjacent. All other elements are set to 0.

    Example::

        >>> import torch
        >>> A = torch.tensor([[0, 1], [1, 0]])
        >>> B = torch.tensor([[3, 4, 5], [6, 7, 8]])
        >>> C = torch.tensor(7)
        >>> D = torch.tensor([1, 2, 3])
        >>> E = torch.tensor([[4], [5], [6]])
        >>> torch.block_diag(A, B, C, D, E)
        tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 4, 5, 0, 0, 0, 0, 0],
                [0, 0, 6, 7, 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 6]])
    """

    title = 'Block_diagNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.block_diag())



"""
WARNING: Module BmmNode was generated using fallback option. May contain bugs
"""

class BmmNode(Node):
    """
bmm(input, mat2, *, deterministic=False, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied

Keyword Args:
    deterministic (bool, optional): flag to choose between a faster non-deterministic
                                    calculation, or a slower deterministic calculation.
                                    This argument is only available for sparse-dense CUDA bmm.
                                    Default: ``False``
    out (Tensor, optional): the output tensor.

Example::

    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
    torch.Size([10, 3, 5])
"""

    title = 'BmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bmm())



"""
WARNING: Module BoolNode was generated using fallback option. May contain bugs
"""

class BoolNode(Node):
    """None"""

    title = 'BoolNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bool())


class Broadcast_shapesNode(Node):
    """broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need create to intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """

    title = 'Broadcast_shapesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.broadcast_shapes())


class Broadcast_tensorsNode(Node):
    """broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """

    title = 'Broadcast_tensorsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.broadcast_tensors())



"""
WARNING: Module Broadcast_toNode was generated using fallback option. May contain bugs
"""

class Broadcast_toNode(Node):
    """
broadcast_to(input, shape) -> Tensor

Broadcasts :attr:`input` to the shape :attr:`\shape`.
Equivalent to calling ``input.expand(shape)``. See :meth:`~Tensor.expand` for details.

Args:
    input (Tensor): the input tensor.
    shape (list, tuple, or :class:`torch.Size`): the new shape.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> torch.broadcast_to(x, (3, 3))
    tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])
"""

    title = 'Broadcast_toNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.broadcast_to())



"""
WARNING: Module BucketizeNode was generated using fallback option. May contain bugs
"""

class BucketizeNode(Node):
    """
bucketize(input, boundaries, *, out_int32=False, right=False, out=None) -> Tensor

Returns the indices of the buckets to which each value in the :attr:`input` belongs, where the
boundaries of the buckets are set by :attr:`boundaries`. Return a new tensor with the same size
as :attr:`input`. If :attr:`right` is False (default), then the left boundary is closed. More
formally, the returned index satisfies the following rules:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - :attr:`right`
     - *returned index satisfies*
   * - False
     - ``boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]``
   * - True
     - ``boundaries[i-1] <= input[m][n]...[l][x] < boundaries[i]``

Args:
    input (Tensor or Scalar): N-D tensor or a Scalar containing the search value(s).
    boundaries (Tensor): 1-D tensor, must contain a monotonically increasing sequence.

Keyword args:
    out_int32 (bool, optional): indicate the output data type. torch.int32 if True, torch.int64 otherwise.
                                Default value is False, i.e. default output data type is torch.int64.
    right (bool, optional): if False, return the first suitable location that is found. If True, return the
                            last such index. If no suitable index found, return 0 for non-numerical value
                            (eg. nan, inf) or the size of :attr:`boundaries` (one pass the last index).
                            In other words, if False, gets the lower bound index for each value in :attr:`input`
                            from :attr:`boundaries`. If True, gets the upper bound index instead.
                            Default value is False.
    out (Tensor, optional): the output tensor, must be the same size as :attr:`input` if provided.


Example::

    >>> boundaries = torch.tensor([1, 3, 5, 7, 9])
    >>> boundaries
    tensor([1, 3, 5, 7, 9])
    >>> v = torch.tensor([[3, 6, 9], [3, 6, 9]])
    >>> v
    tensor([[3, 6, 9],
            [3, 6, 9]])
    >>> torch.bucketize(v, boundaries)
    tensor([[1, 3, 4],
            [1, 3, 4]])
    >>> torch.bucketize(v, boundaries, right=True)
    tensor([[2, 3, 5],
            [2, 3, 5]])
"""

    title = 'BucketizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.bucketize())



"""
WARNING: Module Can_castNode was generated using fallback option. May contain bugs
"""

class Can_castNode(Node):
    """
can_cast(from, to) -> bool

Determines if a type conversion is allowed under PyTorch casting rules
described in the type promotion :ref:`documentation <type-promotion-doc>`.

Args:
    from (dtype): The original :class:`torch.dtype`.
    to (dtype): The target :class:`torch.dtype`.

Example::

    >>> torch.can_cast(torch.double, torch.float)
    True
    >>> torch.can_cast(torch.float, torch.int)
    False
"""

    title = 'Can_castNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.can_cast())



"""
WARNING: Module CandidateNode was generated using fallback option. May contain bugs
"""

class CandidateNode(Node):
    """wait(arg0: torch._C.Future) -> object
"""

    title = 'CandidateNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.candidate(self.input(0)))


class Cartesian_prodNode(Node):
    """Do cartesian product of the given sequence of tensors. The behavior is similar to
    python's `itertools.product`.

    Args:
        *tensors: any number of 1 dimensional tensors.

    Returns:
        Tensor: A tensor equivalent to converting all the input tensors into lists,
        do `itertools.product` on these lists, and finally convert the resulting list
        into tensor.

    Example::

        >>> a = [1, 2, 3]
        >>> b = [4, 5]
        >>> list(itertools.product(a, b))
        [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
        >>> tensor_a = torch.tensor(a)
        >>> tensor_b = torch.tensor(b)
        >>> torch.cartesian_prod(tensor_a, tensor_b)
        tensor([[1, 4],
                [1, 5],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5]])
    """

    title = 'Cartesian_prodNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cartesian_prod())



"""
WARNING: Module CatNode was generated using fallback option. May contain bugs
"""

class CatNode(Node):
    """
cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be empty.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

Args:
    tensors (sequence of Tensors): any python sequence of tensors of the same type.
        Non-empty tensors provided must have the same shape, except in the
        cat dimension.
    dim (int, optional): the dimension over which the tensors are concatenated

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
"""

    title = 'CatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cat())


class CdistNode(Node):
    """Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
        p: p value for the p-norm distance to calculate between each vector pair
            :math:`\in [0, \infty]`.
        compute_mode:
            'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
            euclidean distance (p = 2) if P > 25 or R > 25
            'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            Default: use_mm_for_euclid_dist_if_necessary.

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
    `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
    scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Example:

        >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch.cdist(a, b, p=2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """

    title = 'CdistNode'
    init_inputs = [
        NodeInputBP('x1'),
NodeInputBP('x2'),
NodeInputBP('p'),
NodeInputBP('compute_mode'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cdist(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module CdoubleNode was generated using fallback option. May contain bugs
"""

class CdoubleNode(Node):
    """None"""

    title = 'CdoubleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cdouble())



"""
WARNING: Module CeilNode was generated using fallback option. May contain bugs
"""

class CeilNode(Node):
    """
ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
"""

    title = 'CeilNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ceil())



"""
WARNING: Module Ceil_Node was generated using fallback option. May contain bugs
"""

class Ceil_Node(Node):
    """None"""

    title = 'Ceil_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ceil_())



"""
WARNING: Module CeluNode was generated using fallback option. May contain bugs
"""

class CeluNode(Node):
    """None"""

    title = 'CeluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.celu())



"""
WARNING: Module Celu_Node was generated using fallback option. May contain bugs
"""

class Celu_Node(Node):
    """
celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`.
"""

    title = 'Celu_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.celu_())



"""
WARNING: Module CfloatNode was generated using fallback option. May contain bugs
"""

class CfloatNode(Node):
    """None"""

    title = 'CfloatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cfloat())


class Chain_matmulNode(Node):
    """Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
    using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
    of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
    needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
    If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.

    .. warning::

        :func:`torch.chain_matmul` is deprecated and will be removed in a future PyTorch release.
        Use :func:`torch.linalg.multi_dot` instead, which accepts a list of two or more tensors
        rather than multiple arguments.

    Args:
        matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.
        out (Tensor, optional): the output tensor. Ignored if :attr:`out` = ``None``.

    Returns:
        Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \times p_{i + 1}`, then the product
        would be of dimensions :math:`p_{1} \times p_{N + 1}`.

    Example::

        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = torch.randn(5, 6)
        >>> d = torch.randn(6, 7)
        >>> torch.chain_matmul(a, b, c, d)
        tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
                [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
                [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

    .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
    """

    title = 'Chain_matmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.chain_matmul())



"""
WARNING: Module Channel_shuffleNode was generated using fallback option. May contain bugs
"""

class Channel_shuffleNode(Node):
    """
channel_shuffle(input, groups) -> Tensor

Divide the channels in a tensor of shape :math:`(*, C , H, W)`
into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`,
while keeping the original tensor shape.

See :class:`~torch.nn.ChannelShuffle` for details.

Args:
    input (Tensor): the input tensor
    groups (int): number of groups to divide channels in and rearrange.

Examples::

    >>> input = torch.randn(1, 4, 2, 2)
    >>> print(input)
    [[[[1, 2],
       [3, 4]],
      [[5, 6],
       [7, 8]],
      [[9, 10],
       [11, 12]],
      [[13, 14],
       [15, 16]],
     ]]
    >>> output = torch.nn.functional.channel_shuffle(input, 2)
    >>> print(output)
    [[[[1, 2],
       [3, 4]],
      [[9, 10],
       [11, 12]],
      [[5, 6],
       [7, 8]],
      [[13, 14],
       [15, 16]],
     ]]
"""

    title = 'Channel_shuffleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.channel_shuffle())



"""
WARNING: Module Channels_lastNode was generated using fallback option. May contain bugs
"""

class Channels_lastNode(Node):
    """None"""

    title = 'Channels_lastNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.channels_last())



"""
WARNING: Module Channels_last_3dNode was generated using fallback option. May contain bugs
"""

class Channels_last_3dNode(Node):
    """None"""

    title = 'Channels_last_3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.channels_last_3d())



"""
WARNING: Module CholeskyNode was generated using fallback option. May contain bugs
"""

class CholeskyNode(Node):
    """
cholesky(input, upper=False, *, out=None) -> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix :math:`A` or for batches of symmetric positive-definite matrices.

If :attr:`upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
the decomposition has the form:

.. math::

  A = U^TU

If :attr:`upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
the decomposition has the form:

.. math::

    A = LL^T

If :attr:`upper` is ``True``, and :math:`A` is a batch of symmetric positive-definite
matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
of each of the individual matrices. Similarly, when :attr:`upper` is ``False``, the returned
tensor will be composed of lower-triangular Cholesky factors of each of the individual
matrices.

.. warning::

    :func:`torch.cholesky` is deprecated in favor of :func:`torch.linalg.cholesky`
    and will be removed in a future PyTorch release.

    ``L = torch.cholesky(A)`` should be replaced with

    .. code:: python

        L = torch.linalg.cholesky(A)

    ``U = torch.cholesky(A, upper=True)`` should be replaced with

    .. code:: python

        U = torch.linalg.cholesky(A.transpose(-2, -1).conj()).transpose(-2, -1).conj()

Args:
    input (Tensor): the input tensor :math:`A` of size :math:`(*, n, n)` where `*` is zero or more
                batch dimensions consisting of symmetric positive-definite matrices.
    upper (bool, optional): flag that indicates whether to return a
                            upper or lower triangular matrix. Default: ``False``

Keyword args:
    out (Tensor, optional): the output matrix

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> a
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> l
    tensor([[ 1.5528,  0.0000,  0.0000],
            [-0.4821,  1.0592,  0.0000],
            [ 0.9371,  0.5487,  0.7023]])
    >>> torch.mm(l, l.t())
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> a = torch.randn(3, 2, 2)
    >>> a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> z = torch.matmul(l, l.transpose(-1, -2))
    >>> torch.max(torch.abs(z - a)) # Max non-zero
    tensor(2.3842e-07)
"""

    title = 'CholeskyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cholesky())



"""
WARNING: Module Cholesky_inverseNode was generated using fallback option. May contain bugs
"""

class Cholesky_inverseNode(Node):
    """
cholesky_inverse(input, upper=False, *, out=None) -> Tensor

Computes the inverse of a symmetric positive-definite matrix :math:`A` using its
Cholesky factor :math:`u`: returns matrix ``inv``. The inverse is computed using
LAPACK routines ``dpotri`` and ``spotri`` (and the corresponding MAGMA routines).

If :attr:`upper` is ``False``, :math:`u` is lower triangular
such that the returned tensor is

.. math::
    inv = (uu^{{T}})^{{-1}}

If :attr:`upper` is ``True`` or not provided, :math:`u` is upper
triangular such that the returned tensor is

.. math::
    inv = (u^T u)^{{-1}}

Args:
    input (Tensor): the input 2-D tensor :math:`u`, a upper or lower triangular
           Cholesky factor
    upper (bool, optional): whether to return a lower (default) or upper triangular matrix

Keyword args:
    out (Tensor, optional): the output tensor for `inv`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[  0.9935,  -0.6353,   1.5806],
            [ -0.6353,   0.8769,  -1.7183],
            [  1.5806,  -1.7183,  10.6618]])
    >>> torch.cholesky_inverse(u)
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
    >>> a.inverse()
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
"""

    title = 'Cholesky_inverseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cholesky_inverse())



"""
WARNING: Module Cholesky_solveNode was generated using fallback option. May contain bugs
"""

class Cholesky_solveNode(Node):
    """
cholesky_solve(input, input2, upper=False, *, out=None) -> Tensor

Solves a linear system of equations with a positive semidefinite
matrix to be inverted given its Cholesky factor matrix :math:`u`.

If :attr:`upper` is ``False``, :math:`u` is and lower triangular and `c` is
returned such that:

.. math::
    c = (u u^T)^{{-1}} b

If :attr:`upper` is ``True`` or not provided, :math:`u` is upper triangular
and `c` is returned such that:

.. math::
    c = (u^T u)^{{-1}} b

`torch.cholesky_solve(b, u)` can take in 2D inputs `b, u` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `c`

Supports real-valued and complex-valued inputs.
For the complex-valued inputs the transpose operator above is the conjugate transpose.

Args:
    input (Tensor): input matrix :math:`b` of size :math:`(*, m, k)`,
                where :math:`*` is zero or more batch dimensions
    input2 (Tensor): input matrix :math:`u` of size :math:`(*, m, m)`,
                where :math:`*` is zero of more batch dimensions composed of
                upper or lower triangular Cholesky factor
    upper (bool, optional): whether to consider the Cholesky factor as a
                            lower or upper triangular matrix. Default: ``False``.

Keyword args:
    out (Tensor, optional): the output tensor for `c`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[ 0.7747, -1.9549,  1.3086],
            [-1.9549,  6.7546, -5.4114],
            [ 1.3086, -5.4114,  4.8733]])
    >>> b = torch.randn(3, 2)
    >>> b
    tensor([[-0.6355,  0.9891],
            [ 0.1974,  1.4706],
            [-0.4115, -0.6225]])
    >>> torch.cholesky_solve(b, u)
    tensor([[ -8.1625,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
    >>> torch.mm(a.inverse(), b)
    tensor([[ -8.1626,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
"""

    title = 'Cholesky_solveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cholesky_solve())



"""
WARNING: Module Choose_qparams_optimizedNode was generated using fallback option. May contain bugs
"""

class Choose_qparams_optimizedNode(Node):
    """None"""

    title = 'Choose_qparams_optimizedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.choose_qparams_optimized())



"""
WARNING: Module ChunkNode was generated using fallback option. May contain bugs
"""

class ChunkNode(Node):
    """
chunk(input, chunks, dim=0) -> List of Tensors

Splits a tensor into a specific number of chunks. Each chunk is a view of
the input tensor.

Last chunk will be smaller if the tensor size along the given dimension
:attr:`dim` is not divisible by :attr:`chunks`.

Arguments:
    input (Tensor): the tensor to split
    chunks (int): number of chunks to return
    dim (int): dimension along which to split the tensor
"""

    title = 'ChunkNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.chunk())



"""
WARNING: Module ClampNode was generated using fallback option. May contain bugs
"""

class ClampNode(Node):
    """
clamp(input, min=None, max=None, *, out=None) -> Tensor

Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
Letting min_value and max_value be :attr:`min` and :attr:`max`, respectively, this returns:

.. math::
    y_i = \min(\max(x_i, \text{min\_value}_i), \text{max\_value}_i)

If :attr:`min` is ``None``, there is no lower bound.
Or, if :attr:`max` is ``None`` there is no upper bound.


.. note::
    If :attr:`min` is greater than :attr:`max` :func:`torch.clamp(..., min, max) <torch.clamp>`
    sets all elements in :attr:`input` to the value of :attr:`max`.

Args:
    input (Tensor): the input tensor.
    min (Number or Tensor, optional): lower-bound of the range to be clamped to
    max (Number or Tensor, optional): upper-bound of the range to be clamped to

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])

    >>> min = torch.linspace(-1, 1, steps=4)
    >>> torch.clamp(a, min=min)
    tensor([-1.0000,  0.1734,  0.3333,  1.0000])

"""

    title = 'ClampNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp())



"""
WARNING: Module Clamp_Node was generated using fallback option. May contain bugs
"""

class Clamp_Node(Node):
    """None"""

    title = 'Clamp_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp_())



"""
WARNING: Module Clamp_maxNode was generated using fallback option. May contain bugs
"""

class Clamp_maxNode(Node):
    """None"""

    title = 'Clamp_maxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp_max())



"""
WARNING: Module Clamp_max_Node was generated using fallback option. May contain bugs
"""

class Clamp_max_Node(Node):
    """None"""

    title = 'Clamp_max_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp_max_())



"""
WARNING: Module Clamp_minNode was generated using fallback option. May contain bugs
"""

class Clamp_minNode(Node):
    """None"""

    title = 'Clamp_minNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp_min())



"""
WARNING: Module Clamp_min_Node was generated using fallback option. May contain bugs
"""

class Clamp_min_Node(Node):
    """None"""

    title = 'Clamp_min_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clamp_min_())



"""
WARNING: Module ClassesNode was generated using fallback option. May contain bugs
"""

class ClassesNode(Node):
    """None"""

    title = 'ClassesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.classes())



"""
WARNING: Module Clear_autocast_cacheNode was generated using fallback option. May contain bugs
"""

class Clear_autocast_cacheNode(Node):
    """None"""

    title = 'Clear_autocast_cacheNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clear_autocast_cache())



"""
WARNING: Module ClipNode was generated using fallback option. May contain bugs
"""

class ClipNode(Node):
    """
clip(input, min=None, max=None, *, out=None) -> Tensor

Alias for :func:`torch.clamp`.
"""

    title = 'ClipNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clip())



"""
WARNING: Module Clip_Node was generated using fallback option. May contain bugs
"""

class Clip_Node(Node):
    """None"""

    title = 'Clip_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clip_())



"""
WARNING: Module CloneNode was generated using fallback option. May contain bugs
"""

class CloneNode(Node):
    """
clone(input, *, memory_format=torch.preserve_format) -> Tensor

Returns a copy of :attr:`input`.

.. note::

    This function is differentiable, so gradients will flow back from the
    result of this operation to :attr:`input`. To create a tensor without an
    autograd relationship to :attr:`input` see :meth:`~Tensor.detach`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned tensor. Default: ``torch.preserve_format``.
"""

    title = 'CloneNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.clone())



"""
WARNING: Module Column_stackNode was generated using fallback option. May contain bugs
"""

class Column_stackNode(Node):
    """
column_stack(tensors, *, out=None) -> Tensor

Creates a new tensor by horizontally stacking the tensors in :attr:`tensors`.

Equivalent to ``torch.hstack(tensors)``, except each zero or one dimensional tensor ``t``
in :attr:`tensors` is first reshaped into a ``(t.numel(), 1)`` column before being stacked horizontally.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.column_stack((a, b))
    tensor([[1, 4],
        [2, 5],
        [3, 6]])
    >>> a = torch.arange(5)
    >>> b = torch.arange(10).reshape(5, 2)
    >>> torch.column_stack((a, b, b))
    tensor([[0, 0, 1, 0, 1],
            [1, 2, 3, 2, 3],
            [2, 4, 5, 4, 5],
            [3, 6, 7, 6, 7],
            [4, 8, 9, 8, 9]])

"""

    title = 'Column_stackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.column_stack())



"""
WARNING: Module CombinationsNode was generated using fallback option. May contain bugs
"""

class CombinationsNode(Node):
    """
combinations(input, r=2, with_replacement=False) -> seq

Compute combinations of length :math:`r` of the given tensor. The behavior is similar to
python's `itertools.combinations` when `with_replacement` is set to `False`, and
`itertools.combinations_with_replacement` when `with_replacement` is set to `True`.

Arguments:
    input (Tensor): 1D vector.
    r (int, optional): number of elements to combine
    with_replacement (boolean, optional): whether to allow duplication in combination

Returns:
    Tensor: A tensor equivalent to converting all the input tensors into lists, do
    `itertools.combinations` or `itertools.combinations_with_replacement` on these
    lists, and finally convert the resulting list into tensor.

Example::

    >>> a = [1, 2, 3]
    >>> list(itertools.combinations(a, r=2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(itertools.combinations(a, r=3))
    [(1, 2, 3)]
    >>> list(itertools.combinations_with_replacement(a, r=2))
    [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    >>> tensor_a = torch.tensor(a)
    >>> torch.combinations(tensor_a)
    tensor([[1, 2],
            [1, 3],
            [2, 3]])
    >>> torch.combinations(tensor_a, r=3)
    tensor([[1, 2, 3]])
    >>> torch.combinations(tensor_a, with_replacement=True)
    tensor([[1, 1],
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
            [3, 3]])
"""

    title = 'CombinationsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.combinations())


class Compiled_with_cxx11_abiNode(Node):
    """Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""

    title = 'Compiled_with_cxx11_abiNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.compiled_with_cxx11_abi())



"""
WARNING: Module ComplexNode was generated using fallback option. May contain bugs
"""

class ComplexNode(Node):
    """
complex(real, imag, *, out=None) -> Tensor

Constructs a complex tensor with its real part equal to :attr:`real` and its
imaginary part equal to :attr:`imag`.

Args:
    real (Tensor): The real part of the complex tensor. Must be float or double.
    imag (Tensor): The imaginary part of the complex tensor. Must be same dtype
        as :attr:`real`.

Keyword args:
    out (Tensor): If the inputs are ``torch.float32``, must be
        ``torch.complex64``. If the inputs are ``torch.float64``, must be
        ``torch.complex128``.

Example::

    >>> real = torch.tensor([1, 2], dtype=torch.float32)
    >>> imag = torch.tensor([3, 4], dtype=torch.float32)
    >>> z = torch.complex(real, imag)
    >>> z
    tensor([(1.+3.j), (2.+4.j)])
    >>> z.dtype
    torch.complex64

"""

    title = 'ComplexNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.complex())



"""
WARNING: Module Complex128Node was generated using fallback option. May contain bugs
"""

class Complex128Node(Node):
    """None"""

    title = 'Complex128Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.complex128())



"""
WARNING: Module Complex32Node was generated using fallback option. May contain bugs
"""

class Complex32Node(Node):
    """None"""

    title = 'Complex32Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.complex32())



"""
WARNING: Module Complex64Node was generated using fallback option. May contain bugs
"""

class Complex64Node(Node):
    """None"""

    title = 'Complex64Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.complex64())



"""
WARNING: Module ConjNode was generated using fallback option. May contain bugs
"""

class ConjNode(Node):
    """
conj(input, *, out=None) -> Tensor

Computes the element-wise conjugate of the given :attr:`input` tensor. If :attr:`input` has a non-complex dtype,
this function just returns :attr:`input`.

.. warning:: In the future, :func:`torch.conj` may return a non-writeable view for an :attr:`input` of
             non-complex dtype. It's recommended that programs not modify the tensor returned by :func:`torch.conj`
             when :attr:`input` is of non-complex dtype to be compatible with this change.

.. math::
    \text{out}_{i} = conj(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
    tensor([-1 - 1j, -2 - 2j, 3 + 3j])
"""

    title = 'ConjNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conj())



"""
WARNING: Module Constant_pad_ndNode was generated using fallback option. May contain bugs
"""

class Constant_pad_ndNode(Node):
    """None"""

    title = 'Constant_pad_ndNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.constant_pad_nd())



"""
WARNING: Module Contiguous_formatNode was generated using fallback option. May contain bugs
"""

class Contiguous_formatNode(Node):
    """None"""

    title = 'Contiguous_formatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.contiguous_format())



"""
WARNING: Module Conv1dNode was generated using fallback option. May contain bugs
"""

class Conv1dNode(Node):
    """
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.Conv1d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a one-element tuple `(padW,)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
      a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> inputs = torch.randn(33, 16, 30)
    >>> filters = torch.randn(20, 16, 5)
    >>> F.conv1d(inputs, filters)
"""

    title = 'Conv1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv1d())



"""
WARNING: Module Conv2dNode was generated using fallback option. May contain bugs
"""

class Conv2dNode(Node):
    """
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.Conv2d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
"""

    title = 'Conv2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv2d())



"""
WARNING: Module Conv3dNode was generated using fallback option. May contain bugs
"""

class Conv3dNode(Node):
    """
conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.Conv3d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padT, padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> filters = torch.randn(33, 16, 3, 3, 3)
    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> F.conv3d(inputs, filters)
"""

    title = 'Conv3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv3d())



"""
WARNING: Module Conv_tbcNode was generated using fallback option. May contain bugs
"""

class Conv_tbcNode(Node):
    """
Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

Args:
    input: input tensor of shape :math:`(\text{sequence length} \times batch \times \text{in\_channels})`
    weight: filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
    bias: bias of shape (:math:`\text{out\_channels}`)
    pad: number of timesteps to pad. Default: 0
"""

    title = 'Conv_tbcNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv_tbc())



"""
WARNING: Module Conv_transpose1dNode was generated using fallback option. May contain bugs
"""

class Conv_transpose1dNode(Node):
    """
conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padW,)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dW,)``. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50)
    >>> weights = torch.randn(16, 33, 5)
    >>> F.conv_transpose1d(inputs, weights)
"""

    title = 'Conv_transpose1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv_transpose1d())



"""
WARNING: Module Conv_transpose2dNode was generated using fallback option. May contain bugs
"""

class Conv_transpose2dNode(Node):
    """
conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
      Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dH, dW)``. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> weights = torch.randn(4, 8, 3, 3)
    >>> F.conv_transpose2d(inputs, weights, padding=1)
"""

    title = 'Conv_transpose2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv_transpose2d())



"""
WARNING: Module Conv_transpose3dNode was generated using fallback option. May contain bugs
"""

class Conv_transpose3dNode(Node):
    """
conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution"

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.ConvTranspose3d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sT, sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padT, padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple
      ``(out_padT, out_padH, out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> weights = torch.randn(16, 33, 3, 3, 3)
    >>> F.conv_transpose3d(inputs, weights)
"""

    title = 'Conv_transpose3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.conv_transpose3d())



"""
WARNING: Module ConvolutionNode was generated using fallback option. May contain bugs
"""

class ConvolutionNode(Node):
    """None"""

    title = 'ConvolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.convolution())



"""
WARNING: Module CopysignNode was generated using fallback option. May contain bugs
"""

class CopysignNode(Node):
    """
copysign(input, other, *, out=None) -> Tensor

Create a new floating-point tensor with the magnitude of :attr:`input` and the sign of :attr:`other`, elementwise.

.. math::
    \text{out}_{i} = \begin{cases}
        -|\text{input}_{i}| & \text{if} \text{other}_{i} \leq -0.0 \\
        |\text{input}_{i}| & \text{if} \text{other}_{i} \geq 0.0 \\
    \end{cases}


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
and integer and float inputs.

Args:
    input (Tensor): magnitudes.
    other (Tensor or Number): contains value(s) whose signbit(s) are
        applied to the magnitudes in :attr:`input`.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-1.2557, -0.0026, -0.5387,  0.4740, -0.9244])
    >>> torch.copysign(a, 1)
    tensor([1.2557, 0.0026, 0.5387, 0.4740, 0.9244])
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.7079,  0.2778, -1.0249,  0.5719],
            [-0.0059, -0.2600, -0.4475, -1.3948],
            [ 0.3667, -0.9567, -2.5757, -0.1751],
            [ 0.2046, -0.0742,  0.2998, -0.1054]])
    >>> b = torch.randn(4)
    tensor([ 0.2373,  0.3120,  0.3190, -1.1128])
    >>> torch.copysign(a, b)
    tensor([[ 0.7079,  0.2778,  1.0249, -0.5719],
            [ 0.0059,  0.2600,  0.4475, -1.3948],
            [ 0.3667,  0.9567,  2.5757, -0.1751],
            [ 0.2046,  0.0742,  0.2998, -0.1054]])

"""

    title = 'CopysignNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.copysign())



"""
WARNING: Module CosNode was generated using fallback option. May contain bugs
"""

class CosNode(Node):
    """
cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
"""

    title = 'CosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cos())



"""
WARNING: Module Cos_Node was generated using fallback option. May contain bugs
"""

class Cos_Node(Node):
    """None"""

    title = 'Cos_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cos_())



"""
WARNING: Module CoshNode was generated using fallback option. May contain bugs
"""

class CoshNode(Node):
    """
cosh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic cosine  of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \cosh(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
    >>> torch.cosh(a)
    tensor([ 1.0133,  1.7860,  1.2536,  1.2805])

.. note::
   When :attr:`input` is on the CPU, the implementation of torch.cosh may use
   the Sleef library, which rounds very large results to infinity or negative
   infinity. See `here <https://sleef.org/purec.xhtml>`_ for details.
"""

    title = 'CoshNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cosh())



"""
WARNING: Module Cosh_Node was generated using fallback option. May contain bugs
"""

class Cosh_Node(Node):
    """None"""

    title = 'Cosh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cosh_())



"""
WARNING: Module Cosine_embedding_lossNode was generated using fallback option. May contain bugs
"""

class Cosine_embedding_lossNode(Node):
    """None"""

    title = 'Cosine_embedding_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cosine_embedding_loss())



"""
WARNING: Module Cosine_similarityNode was generated using fallback option. May contain bugs
"""

class Cosine_similarityNode(Node):
    """
cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor

Returns cosine similarity between x1 and x2, computed along dim.

.. math ::
    \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

Args:
    x1 (Tensor): First input.
    x2 (Tensor): Second input (of size matching x1).
    dim (int, optional): Dimension of vectors. Default: 1
    eps (float, optional): Small value to avoid division by zero.
        Default: 1e-8

Shape:
    - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
    - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

Example::

    >>> input1 = torch.randn(100, 128)
    >>> input2 = torch.randn(100, 128)
    >>> output = F.cosine_similarity(input1, input2)
    >>> print(output)
"""

    title = 'Cosine_similarityNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cosine_similarity())



"""
WARNING: Module Count_nonzeroNode was generated using fallback option. May contain bugs
"""

class Count_nonzeroNode(Node):
    """
count_nonzero(input, dim=None) -> Tensor

Counts the number of non-zero values in the tensor :attr:`input` along the given :attr:`dim`.
If no dim is specified then all non-zeros in the tensor are counted.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints, optional): Dim or tuple of dims along which to count non-zeros.

Example::

    >>> x = torch.zeros(3,3)
    >>> x[torch.randn(3,3) > 0.5] = 1
    >>> x
    tensor([[0., 1., 1.],
            [0., 0., 0.],
            [0., 0., 1.]])
    >>> torch.count_nonzero(x)
    tensor(3)
    >>> torch.count_nonzero(x, dim=0)
    tensor([0, 1, 2])
"""

    title = 'Count_nonzeroNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.count_nonzero())



"""
WARNING: Module CppNode was generated using fallback option. May contain bugs
"""

class CppNode(Node):
    """None"""

    title = 'CppNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cpp())



"""
WARNING: Module CrossNode was generated using fallback option. May contain bugs
"""

class CrossNode(Node):
    """
cross(input, other, dim=None, *, out=None) -> Tensor


Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
and :attr:`other`.

:attr:`input` and :attr:`other` must have the same size, and the size of their
:attr:`dim` dimension should be 3.

If :attr:`dim` is not given, it defaults to the first dimension found with the
size 3. Note that this might be unexpected.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor
    dim  (int, optional): the dimension to take the cross-product in.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.cross(a, b, dim=1)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> torch.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
"""

    title = 'CrossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cross())



"""
WARNING: Module Ctc_lossNode was generated using fallback option. May contain bugs
"""

class Ctc_lossNode(Node):
    """None"""

    title = 'Ctc_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ctc_loss())



"""
WARNING: Module CtypesNode was generated using fallback option. May contain bugs
"""

class CtypesNode(Node):
    """create and manipulate C data types in Python"""

    title = 'CtypesNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ctypes(self.input(0)))



"""
WARNING: Module CudaNode was generated using fallback option. May contain bugs
"""

class CudaNode(Node):
    """
This package adds support for CUDA tensor types, that implement the same
function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

    title = 'CudaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cuda())



"""
WARNING: Module Cudnn_affine_grid_generatorNode was generated using fallback option. May contain bugs
"""

class Cudnn_affine_grid_generatorNode(Node):
    """None"""

    title = 'Cudnn_affine_grid_generatorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_affine_grid_generator())



"""
WARNING: Module Cudnn_batch_normNode was generated using fallback option. May contain bugs
"""

class Cudnn_batch_normNode(Node):
    """None"""

    title = 'Cudnn_batch_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_batch_norm())



"""
WARNING: Module Cudnn_convolutionNode was generated using fallback option. May contain bugs
"""

class Cudnn_convolutionNode(Node):
    """None"""

    title = 'Cudnn_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_convolution())



"""
WARNING: Module Cudnn_convolution_add_reluNode was generated using fallback option. May contain bugs
"""

class Cudnn_convolution_add_reluNode(Node):
    """None"""

    title = 'Cudnn_convolution_add_reluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_convolution_add_relu())



"""
WARNING: Module Cudnn_convolution_reluNode was generated using fallback option. May contain bugs
"""

class Cudnn_convolution_reluNode(Node):
    """None"""

    title = 'Cudnn_convolution_reluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_convolution_relu())



"""
WARNING: Module Cudnn_convolution_transposeNode was generated using fallback option. May contain bugs
"""

class Cudnn_convolution_transposeNode(Node):
    """None"""

    title = 'Cudnn_convolution_transposeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_convolution_transpose())



"""
WARNING: Module Cudnn_grid_samplerNode was generated using fallback option. May contain bugs
"""

class Cudnn_grid_samplerNode(Node):
    """None"""

    title = 'Cudnn_grid_samplerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_grid_sampler())



"""
WARNING: Module Cudnn_is_acceptableNode was generated using fallback option. May contain bugs
"""

class Cudnn_is_acceptableNode(Node):
    """None"""

    title = 'Cudnn_is_acceptableNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cudnn_is_acceptable())



"""
WARNING: Module CummaxNode was generated using fallback option. May contain bugs
"""

class CummaxNode(Node):
    """
cummax(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative maximum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each maximum value found in the dimension :attr:`dim`.

.. math::
    y_i = max(x_1, x_2, x_3, \dots, x_i)

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.3449, -1.5447,  0.0685, -1.5104, -1.1706,  0.2259,  1.4696, -1.3284,
         1.9946, -0.8209])
    >>> torch.cummax(a, dim=0)
    torch.return_types.cummax(
        values=tensor([-0.3449, -0.3449,  0.0685,  0.0685,  0.0685,  0.2259,  1.4696,  1.4696,
         1.9946,  1.9946]),
        indices=tensor([0, 0, 2, 2, 2, 5, 6, 6, 8, 8]))
"""

    title = 'CummaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cummax())



"""
WARNING: Module CumminNode was generated using fallback option. May contain bugs
"""

class CumminNode(Node):
    """
cummin(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each maximum value found in the dimension :attr:`dim`.

.. math::
    y_i = min(x_1, x_2, x_3, \dots, x_i)

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220, -0.3885,  1.1762,
         0.9165,  1.6684])
    >>> torch.cummin(a, dim=0)
    torch.return_types.cummin(
        values=tensor([-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298, -1.3298, -1.3298,
        -1.3298, -1.3298]),
        indices=tensor([0, 1, 1, 1, 4, 4, 4, 4, 4, 4]))
"""

    title = 'CumminNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cummin())



"""
WARNING: Module CumprodNode was generated using fallback option. May contain bugs
"""

class CumprodNode(Node):
    """
cumprod(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 \times x_2\times x_3\times \dots \times x_i

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
            -0.2129, -0.4206,  0.1968])
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
             0.0014, -0.0006, -0.0001])

    >>> a[5] = 0.0
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
             0.0000, -0.0000, -0.0000])
"""

    title = 'CumprodNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cumprod())



"""
WARNING: Module CumsumNode was generated using fallback option. May contain bugs
"""

class CumsumNode(Node):
    """
cumsum(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 + x_2 + x_3 + \dots + x_i

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
             0.1850, -1.1571, -0.4243])
    >>> torch.cumsum(a, dim=0)
    tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
            -1.8209, -2.9780, -3.4022])
"""

    title = 'CumsumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.cumsum())



"""
WARNING: Module Default_generatorNode was generated using fallback option. May contain bugs
"""

class Default_generatorNode(Node):
    """None"""

    title = 'Default_generatorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.default_generator())



"""
WARNING: Module Deg2radNode was generated using fallback option. May contain bugs
"""

class Deg2radNode(Node):
    """
deg2rad(input, *, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input`
converted from angles in degrees to radians.

Args:
    input (Tensor): the input tensor.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
    >>> torch.deg2rad(a)
    tensor([[ 3.1416, -3.1416],
            [ 6.2832, -6.2832],
            [ 1.5708, -1.5708]])

"""

    title = 'Deg2radNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.deg2rad())



"""
WARNING: Module Deg2rad_Node was generated using fallback option. May contain bugs
"""

class Deg2rad_Node(Node):
    """None"""

    title = 'Deg2rad_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.deg2rad_())



"""
WARNING: Module DequantizeNode was generated using fallback option. May contain bugs
"""

class DequantizeNode(Node):
    """
dequantize(tensor) -> Tensor

Returns an fp32 Tensor by dequantizing a quantized Tensor

Args:
    tensor (Tensor): A quantized Tensor

.. function:: dequantize(tensors) -> sequence of Tensors

Given a list of quantized Tensors, dequantize them and return a list of fp32 Tensors

Args:
     tensors (sequence of Tensors): A list of quantized Tensors
"""

    title = 'DequantizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dequantize())



"""
WARNING: Module DetNode was generated using fallback option. May contain bugs
"""

class DetNode(Node):
    """
det(input) -> Tensor

Alias for :func:`torch.linalg.det`
"""

    title = 'DetNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.det())



"""
WARNING: Module DetachNode was generated using fallback option. May contain bugs
"""

class DetachNode(Node):
    """None"""

    title = 'DetachNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.detach())



"""
WARNING: Module Detach_Node was generated using fallback option. May contain bugs
"""

class Detach_Node(Node):
    """None"""

    title = 'Detach_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.detach_())



"""
WARNING: Module DeviceNode was generated using fallback option. May contain bugs
"""

class DeviceNode(Node):
    """None"""

    title = 'DeviceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.device())



"""
WARNING: Module DiagNode was generated using fallback option. May contain bugs
"""

class DiagNode(Node):
    """
diag(input, diagonal=0, *, out=None) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a matrix (2-D tensor), then returns a 1-D tensor with
  the diagonal elements of :attr:`input`.

The argument :attr:`diagonal` controls which diagonal to consider:

- If :attr:`diagonal` = 0, it is the main diagonal.
- If :attr:`diagonal` > 0, it is above the main diagonal.
- If :attr:`diagonal` < 0, it is below the main diagonal.

Args:
    input (Tensor): the input tensor.
    diagonal (int, optional): the diagonal to consider

Keyword args:
    out (Tensor, optional): the output tensor.

.. seealso::

        :func:`torch.diagonal` always returns the diagonal of its input.

        :func:`torch.diagflat` always constructs a tensor with diagonal elements
        specified by the input.

Examples:

Get the square matrix where the input vector is the diagonal::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.5950,-0.0872, 2.3298])
    >>> torch.diag(a)
    tensor([[ 0.5950, 0.0000, 0.0000],
            [ 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 2.3298]])
    >>> torch.diag(a, 1)
    tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
            [ 0.0000, 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 0.0000, 2.3298],
            [ 0.0000, 0.0000, 0.0000, 0.0000]])

Get the k-th diagonal of a given matrix::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-0.4264, 0.0255,-0.1064],
            [ 0.8795,-0.2429, 0.1374],
            [ 0.1029,-0.6482,-1.6300]])
    >>> torch.diag(a, 0)
    tensor([-0.4264,-0.2429,-1.6300])
    >>> torch.diag(a, 1)
    tensor([ 0.0255, 0.1374])
"""

    title = 'DiagNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.diag())



"""
WARNING: Module Diag_embedNode was generated using fallback option. May contain bugs
"""

class Diag_embedNode(Node):
    """
diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor

Creates a tensor whose diagonals of certain 2D planes (specified by
:attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
To facilitate creating batched diagonal matrices, the 2D planes formed by
the last two dimensions of the returned tensor are chosen by default.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

The size of the new matrix will be calculated to make the specified diagonal
of the size of the last input dimension.
Note that for :attr:`offset` other than :math:`0`, the order of :attr:`dim1`
and :attr:`dim2` matters. Exchanging them is equivalent to changing the
sign of :attr:`offset`.

Applying :meth:`torch.diagonal` to the output of this function with
the same arguments yields a matrix identical to input. However,
:meth:`torch.diagonal` has different default dimensions, so those
need to be explicitly specified.

Args:
    input (Tensor): the input tensor. Must be at least 1-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: -2.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: -1.

Example::

    >>> a = torch.randn(2, 3)
    >>> torch.diag_embed(a)
    tensor([[[ 1.5410,  0.0000,  0.0000],
             [ 0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -2.1788]],

            [[ 0.5684,  0.0000,  0.0000],
             [ 0.0000, -1.0845,  0.0000],
             [ 0.0000,  0.0000, -1.3986]]])

    >>> torch.diag_embed(a, offset=1, dim1=0, dim2=2)
    tensor([[[ 0.0000,  1.5410,  0.0000,  0.0000],
             [ 0.0000,  0.5684,  0.0000,  0.0000]],

            [[ 0.0000,  0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -1.0845,  0.0000]],

            [[ 0.0000,  0.0000,  0.0000, -2.1788],
             [ 0.0000,  0.0000,  0.0000, -1.3986]],

            [[ 0.0000,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000]]])
"""

    title = 'Diag_embedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.diag_embed())



"""
WARNING: Module DiagflatNode was generated using fallback option. May contain bugs
"""

class DiagflatNode(Node):
    """
diagflat(input, offset=0) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a tensor with more than one dimension, then returns a
  2-D tensor with diagonal elements equal to a flattened :attr:`input`.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Args:
    input (Tensor): the input tensor.
    offset (int, optional): the diagonal to consider. Default: 0 (main
        diagonal).

Examples::

    >>> a = torch.randn(3)
    >>> a
    tensor([-0.2956, -0.9068,  0.1695])
    >>> torch.diagflat(a)
    tensor([[-0.2956,  0.0000,  0.0000],
            [ 0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.1695]])
    >>> torch.diagflat(a, 1)
    tensor([[ 0.0000, -0.2956,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.1695],
            [ 0.0000,  0.0000,  0.0000,  0.0000]])

    >>> a = torch.randn(2, 2)
    >>> a
    tensor([[ 0.2094, -0.3018],
            [-0.1516,  1.9342]])
    >>> torch.diagflat(a)
    tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -0.3018,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.1516,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.9342]])
"""

    title = 'DiagflatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.diagflat())



"""
WARNING: Module DiagonalNode was generated using fallback option. May contain bugs
"""

class DiagonalNode(Node):
    """
diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor

Returns a partial view of :attr:`input` with the its diagonal elements
with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
at the end of the shape.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Applying :meth:`torch.diag_embed` to the output of this function with
the same arguments yields a diagonal matrix with the diagonal entries
of the input. However, :meth:`torch.diag_embed` has different default
dimensions, so those need to be explicitly specified.

Args:
    input (Tensor): the input tensor. Must be at least 2-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: 0.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: 1.

.. note::  To take a batch diagonal, pass in dim1=-2, dim2=-1.

Examples::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0854,  1.1431, -0.1752],
            [ 0.8536, -0.0905,  0.0360],
            [ 0.6927, -0.3735, -0.4945]])


    >>> torch.diagonal(a, 0)
    tensor([-1.0854, -0.0905, -0.4945])


    >>> torch.diagonal(a, 1)
    tensor([ 1.1431,  0.0360])


    >>> x = torch.randn(2, 5, 4, 2)
    >>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
    tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
             [-1.1065,  1.0401, -0.2235, -0.7938]],

            [[-1.7325, -0.3081,  0.6166,  0.2335],
             [ 1.0500,  0.7336, -0.3836, -1.1015]]])
"""

    title = 'DiagonalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.diagonal())



"""
WARNING: Module DiffNode was generated using fallback option. May contain bugs
"""

class DiffNode(Node):
    """
diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor

Computes the n-th forward difference along the given dimension.

The first-order differences are given by `out[i] = input[i + 1] - input[i]`. Higher-order
differences are calculated by using :func:`torch.diff` recursively.

.. note::  Only `n = 1` is currently supported

Args:
    input (Tensor): the tensor to compute the differences on
    n (int, optional): the number of times to recursively compute the difference
    dim (int, optional): the dimension to compute the difference along.
        Default is the last dimension.
    prepend, append (Tensor, optional): values to prepend or append to
        :attr:`input` along :attr:`dim` before computing the difference.
        Their dimensions must be equivalent to that of input, and their shapes
        must match input's shape except on :attr:`dim`.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 3, 2])
    >>> torch.diff(a)
    tensor([ 2, -1])
    >>> b = torch.tensor([4, 5])
    >>> torch.diff(a, append=b)
    tensor([ 2, -1,  2,  1])
    >>> c = torch.tensor([[1, 2, 3], [3, 4, 5]])
    >>> torch.diff(c, dim=0)
    tensor([[2, 2, 2]])
    >>> torch.diff(c, dim=1)
    tensor([[1, 1],
            [1, 1]])
"""

    title = 'DiffNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.diff())



"""
WARNING: Module DigammaNode was generated using fallback option. May contain bugs
"""

class DigammaNode(Node):
    """
digamma(input, *, out=None) -> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

.. math::
    \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}

Args:
    input (Tensor): the tensor to compute the digamma function on

Keyword args:
    out (Tensor, optional): the output tensor.

.. note::  This function is similar to SciPy's `scipy.special.digamma`.

.. note::  From PyTorch 1.8 onwards, the digamma function returns `-Inf` for `0`.
           Previously it returned `NaN` for `0`.

Example::

    >>> a = torch.tensor([1, 0.5])
    >>> torch.digamma(a)
    tensor([-0.5772, -1.9635])
"""

    title = 'DigammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.digamma())



"""
WARNING: Module DistNode was generated using fallback option. May contain bugs
"""

class DistNode(Node):
    """
dist(input, other, p=2) -> Tensor

Returns the p-norm of (:attr:`input` - :attr:`other`)

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the Right-hand-side input tensor
    p (float, optional): the norm to be computed

Example::

    >>> x = torch.randn(4)
    >>> x
    tensor([-1.5393, -0.8675,  0.5916,  1.6321])
    >>> y = torch.randn(4)
    >>> y
    tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
    >>> torch.dist(x, y, 3.5)
    tensor(1.6727)
    >>> torch.dist(x, y, 3)
    tensor(1.6973)
    >>> torch.dist(x, y, 0)
    tensor(inf)
    >>> torch.dist(x, y, 1)
    tensor(2.6537)
"""

    title = 'DistNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dist())



"""
WARNING: Module DistributedNode was generated using fallback option. May contain bugs
"""

class DistributedNode(Node):
    """None"""

    title = 'DistributedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.distributed())



"""
WARNING: Module DistributionsNode was generated using fallback option. May contain bugs
"""

class DistributionsNode(Node):
    """
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
"""

    title = 'DistributionsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.distributions())



"""
WARNING: Module DivNode was generated using fallback option. May contain bugs
"""

class DivNode(Node):
    """
div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

.. note::
    By default, this performs a "true" division like Python 3.
    See the :attr:`rounding_mode` argument for floor division.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    rounding_mode (str, optional): Type of rounding applied to the result:

        * None - default behavior. Performs no rounding and, if both :attr:`input` and
          :attr:`other` are integer types, promotes the inputs to the default scalar type.
          Equivalent to true division in Python (the ``/`` operator) and NumPy's ``np.true_divide``.
        * ``"trunc"`` - rounds the results of the division towards zero.
          Equivalent to C-style integer division.
        * ``"floor"`` - rounds the results of the division down.
          Equivalent to floor division in Python (the ``//`` operator) and NumPy's ``np.floor_divide``.

    out (Tensor, optional): the output tensor.

Examples::

    >>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(x, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

    >>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
    ...                   [ 0.1815, -1.0111,  0.9805, -1.5923],
    ...                   [ 0.1062,  1.4581,  0.7759, -1.2344],
    ...                   [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
            [ 0.2260, -3.4509, -1.2086,  6.8990],
            [ 0.1322,  4.9764, -0.9564,  5.3484],
            [-0.2278, -0.1068, -1.4678,  6.3938]])

    >>> torch.div(a, b, rounding_mode='trunc')
    tensor([[-0., -6.,  0.,  1.],
            [ 0., -3., -1.,  6.],
            [ 0.,  4., -0.,  5.],
            [-0., -0., -1.,  6.]])

    >>> torch.div(a, b, rounding_mode='floor')
    tensor([[-1., -7.,  0.,  1.],
            [ 0., -4., -2.,  6.],
            [ 0.,  4., -1.,  5.],
            [-1., -1., -2.,  6.]])

"""

    title = 'DivNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.div())



"""
WARNING: Module DivideNode was generated using fallback option. May contain bugs
"""

class DivideNode(Node):
    """
divide(input, other, *, rounding_mode=None, out=None) -> Tensor

Alias for :func:`torch.div`.
"""

    title = 'DivideNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.divide())



"""
WARNING: Module DotNode was generated using fallback option. May contain bugs
"""

class DotNode(Node):
    """
dot(input, other, *, out=None) -> Tensor

Computes the dot product of two 1D tensors.

.. note::

    Unlike NumPy's dot, torch.dot intentionally only supports computing the dot product
    of two 1D tensors with the same number of elements.

Args:
    input (Tensor): first tensor in the dot product, must be 1D.
    other (Tensor): second tensor in the dot product, must be 1D.

Keyword args:
    {out}

Example::

    >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)
"""

    title = 'DotNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dot())



"""
WARNING: Module DoubleNode was generated using fallback option. May contain bugs
"""

class DoubleNode(Node):
    """None"""

    title = 'DoubleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.double())



"""
WARNING: Module DropoutNode was generated using fallback option. May contain bugs
"""

class DropoutNode(Node):
    """None"""

    title = 'DropoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dropout())



"""
WARNING: Module Dropout_Node was generated using fallback option. May contain bugs
"""

class Dropout_Node(Node):
    """None"""

    title = 'Dropout_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dropout_())



"""
WARNING: Module DsmmNode was generated using fallback option. May contain bugs
"""

class DsmmNode(Node):
    """None"""

    title = 'DsmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dsmm())



"""
WARNING: Module DsplitNode was generated using fallback option. May contain bugs
"""

class DsplitNode(Node):
    """
dsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with three or more dimensions, into multiple tensors
depthwise according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=2)
(the split dimension is 1), except that if :attr:`indices_or_sections` is an integer
it must evenly divide the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.dsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (Tensor, int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::
    >>> t = torch.arange(16.0).reshape(2, 2, 4)
    >>> t
    tensor([[[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.]],
            [[ 8.,  9., 10., 11.],
             [12., 13., 14., 15.]]])
    >>> torch.dsplit(t, 2)
    (tensor([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]),
     tensor([[[ 2.,  3.],
              [ 6.,  7.]],
             [[10., 11.],
              [14., 15.]]]))

    >>> torch.dsplit(t, [3, 6])
    (tensor([[[ 0.,  1.,  2.],
              [ 4.,  5.,  6.]],
             [[ 8.,  9., 10.],
              [12., 13., 14.]]]),
     tensor([[[ 3.],
              [ 7.]],
             [[11.],
              [15.]]]),
     tensor([], size=(2, 2, 0)))

"""

    title = 'DsplitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dsplit())



"""
WARNING: Module DstackNode was generated using fallback option. May contain bugs
"""

class DstackNode(Node):
    """
dstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence depthwise (along third axis).

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by :func:`torch.atleast_3d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.dstack((a,b))
    tensor([[[1, 4],
             [2, 5],
             [3, 6]]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.dstack((a,b))
    tensor([[[1, 4]],
            [[2, 5]],
            [[3, 6]]])


"""

    title = 'DstackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dstack())


class DtypeNode(Node):
    """None"""

    title = 'DtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.dtype())



"""
WARNING: Module EigNode was generated using fallback option. May contain bugs
"""

class EigNode(Node):
    """
eig(input, eigenvectors=False, *, out=None) -> (Tensor, Tensor)

Computes the eigenvalues and eigenvectors of a real square matrix.

.. note::
    Since eigenvalues and eigenvectors might be complex, backward pass is supported only
    if eigenvalues and eigenvectors are all real valued.

    When :attr:`input` is on CUDA, :func:`torch.eig() <torch.eig>` causes
    host-device synchronization.

.. warning::

    :func:`torch.eig` is deprecated in favor of :func:`torch.linalg.eig`
    and will be removed in a future PyTorch release.
    :func:`torch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble`
    rather than real tensors mimicking complex tensors.

    ``L, _ = torch.eig(A)`` should be replaced with

    .. code :: python

        L_complex = torch.linalg.eigvals(A)

    ``L, V = torch.eig(A, eigenvectors=True)`` should be replaced with

    .. code :: python

        L_complex, V_complex = torch.linalg.eig(A)

Args:
    input (Tensor): the square matrix of shape :math:`(n \times n)` for which the eigenvalues and eigenvectors
        will be computed
    eigenvectors (bool): ``True`` to compute both eigenvalues and eigenvectors;
        otherwise, only eigenvalues will be computed

Keyword args:
    out (tuple, optional): the output tensors

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(n \times 2)`. Each row is an eigenvalue of ``input``,
          where the first element is the real part and the second element is the imaginary part.
          The eigenvalues are not necessarily ordered.
        - **eigenvectors** (*Tensor*): If ``eigenvectors=False``, it's an empty tensor.
          Otherwise, this tensor of shape :math:`(n \times n)` can be used to compute normalized (unit length)
          eigenvectors of corresponding eigenvalues as follows.
          If the corresponding `eigenvalues[j]` is a real number, column `eigenvectors[:, j]` is the eigenvector
          corresponding to `eigenvalues[j]`.
          If the corresponding `eigenvalues[j]` and `eigenvalues[j + 1]` form a complex conjugate pair, then the
          true eigenvectors can be computed as
          :math:`\text{true eigenvector}[j] = eigenvectors[:, j] + i \times eigenvectors[:, j + 1]`,
          :math:`\text{true eigenvector}[j + 1] = eigenvectors[:, j] - i \times eigenvectors[:, j + 1]`.

Example::

    Trivial example with a diagonal matrix. By default, only eigenvalues are computed:

    >>> a = torch.diag(torch.tensor([1, 2, 3], dtype=torch.double))
    >>> e, v = torch.eig(a)
    >>> e
    tensor([[1., 0.],
            [2., 0.],
            [3., 0.]], dtype=torch.float64)
    >>> v
    tensor([], dtype=torch.float64)

    Compute also the eigenvectors:

    >>> e, v = torch.eig(a, eigenvectors=True)
    >>> e
    tensor([[1., 0.],
            [2., 0.],
            [3., 0.]], dtype=torch.float64)
    >>> v
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]], dtype=torch.float64)

"""

    title = 'EigNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.eig())


class EinsumNode(Node):
    """einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `torch.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).

    Equation:

        The :attr:`equation` string specifies the subscripts (lower case letters `['a', 'z']`) for each dimension of
        the input :attr:`operands` in the same order as the dimensions, separating subcripts for each operand by a
        comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
        must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
        repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
        must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
        appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
        The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
        on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.

        Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation
        followed by the subscripts for the output. For instance, the following equation computes the transpose of a
        matrix multiplication: 'ij,jk->ki'. The output subscripts must appear at least once for some input operand and
        at most once for the output.

        Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
        Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
        e.g. for an input operand with 5 dimensions, the ellipsis in the equation `'ab...c'` cover the third and fourth
        dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
        'shape' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
        explicitly defined with the arrow ('->') notation, the ellipsis will come first in the output (left-most dimensions),
        before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
        batch matrix multiplication `'...ij,...jk'`.

        A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
        arrow and comma) but something like `'. . .'` is not valid. An empty string `''` is valid for scalar operands.

    .. note::

        ``torch.einsum`` handles ellipsis ('...') differently from NumPy in that it allows dimensions
        covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.

    .. note::

        This function does not optimize the given expression, so a different formula for the same computation may
        run faster or consume less memory. Projects like opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/)
        can optimize the formula for you.

    Args:
        equation (string): The subscripts for the Einstein summation.
        operands (Tensor): The operands to compute the Einstein sum of.

    Examples::

        # trace
        >>> torch.einsum('ii', torch.randn(4, 4))
        tensor(-1.2104)

        # diagonal
        >>> torch.einsum('ii->i', torch.randn(4, 4))
        tensor([-0.1034,  0.7952, -0.2433,  0.4545])

        # outer product
        >>> x = torch.randn(5)
        >>> y = torch.randn(4)
        >>> torch.einsum('i,j->ij', x, y)
        tensor([[ 0.1156, -0.2897, -0.3918,  0.4963],
                [-0.3744,  0.9381,  1.2685, -1.6070],
                [ 0.7208, -1.8058, -2.4419,  3.0936],
                [ 0.1713, -0.4291, -0.5802,  0.7350],
                [ 0.5704, -1.4290, -1.9323,  2.4480]])

        # batch matrix multiplication
        >>> As = torch.randn(3,2,5)
        >>> Bs = torch.randn(3,5,4)
        >>> torch.einsum('bij,bjk->bik', As, Bs)
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        # batch permute
        >>> A = torch.randn(2, 3, 4, 5)
        >>> torch.einsum('...ij->...ji', A).shape
        torch.Size([2, 3, 5, 4])

        # equivalent to torch.nn.functional.bilinear
        >>> A = torch.randn(3,5,4)
        >>> l = torch.randn(2,5)
        >>> r = torch.randn(2,4)
        >>> torch.einsum('bn,anm,bm->ba', l, A, r)
        tensor([[-0.3430, -5.2405,  0.4494],
                [ 0.3311,  5.5201, -3.0356]])
    """

    title = 'EinsumNode'
    init_inputs = [
        NodeInputBP('equation'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.einsum(self.input(0)))



"""
WARNING: Module EmbeddingNode was generated using fallback option. May contain bugs
"""

class EmbeddingNode(Node):
    """None"""

    title = 'EmbeddingNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.embedding())



"""
WARNING: Module Embedding_bagNode was generated using fallback option. May contain bugs
"""

class Embedding_bagNode(Node):
    """None"""

    title = 'Embedding_bagNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.embedding_bag())



"""
WARNING: Module Embedding_renorm_Node was generated using fallback option. May contain bugs
"""

class Embedding_renorm_Node(Node):
    """None"""

    title = 'Embedding_renorm_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.embedding_renorm_())



"""
WARNING: Module EmptyNode was generated using fallback option. May contain bugs
"""

class EmptyNode(Node):
    """
empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.

Example::

    >>> a=torch.empty((2,3), dtype=torch.int32, device = 'cuda')
    >>> torch.empty_like(a)
    tensor([[0, 0, 0],
            [0, 0, 0]], device='cuda:0', dtype=torch.int32)
"""

    title = 'EmptyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.empty())



"""
WARNING: Module Empty_likeNode was generated using fallback option. May contain bugs
"""

class Empty_likeNode(Node):
    """
empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.
``torch.empty_like(input)`` is equivalent to
``torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> torch.empty((2,3), dtype=torch.int64)
    tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])
"""

    title = 'Empty_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.empty_like())



"""
WARNING: Module Empty_quantizedNode was generated using fallback option. May contain bugs
"""

class Empty_quantizedNode(Node):
    """None"""

    title = 'Empty_quantizedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.empty_quantized())



"""
WARNING: Module Empty_stridedNode was generated using fallback option. May contain bugs
"""

class Empty_stridedNode(Node):
    """
empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a tensor filled with uninitialized data. The shape and strides of the tensor is
defined by the variable argument :attr:`size` and :attr:`stride` respectively.
``torch.empty_strided(size, stride)`` is equivalent to
``torch.empty(size).as_strided(size, stride)``.

.. warning::
    More than one element of the created tensor may refer to a single memory
    location. As a result, in-place operations (especially ones that are
    vectorized) may result in incorrect behavior. If you need to write to
    the tensors, please clone them first.

Args:
    size (tuple of ints): the shape of the output tensor
    stride (tuple of ints): the strides of the output tensor

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.

Example::

    >>> a = torch.empty_strided((2, 3), (1, 2))
    >>> a
    tensor([[8.9683e-44, 4.4842e-44, 5.1239e+07],
            [0.0000e+00, 0.0000e+00, 3.0705e-41]])
    >>> a.stride()
    (1, 2)
    >>> a.size()
    torch.Size([2, 3])
"""

    title = 'Empty_stridedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.empty_strided())


class Enable_gradNode(Node):
    """Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Example::

        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...   with torch.enable_grad():
        ...     y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True

    """

    title = 'Enable_gradNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.enable_grad())



"""
WARNING: Module EqNode was generated using fallback option. May contain bugs
"""

class EqNode(Node):
    """
eq(input, other, *, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

Example::

    >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[ True, False],
            [False, True]])
"""

    title = 'EqNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.eq())



"""
WARNING: Module EqualNode was generated using fallback option. May contain bugs
"""

class EqualNode(Node):
    """
equal(input, other) -> bool

``True`` if two tensors have the same size and elements, ``False`` otherwise.

Example::

    >>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    True
"""

    title = 'EqualNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.equal())



"""
WARNING: Module ErfNode was generated using fallback option. May contain bugs
"""

class ErfNode(Node):
    """
erf(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erf`.
"""

    title = 'ErfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.erf())



"""
WARNING: Module Erf_Node was generated using fallback option. May contain bugs
"""

class Erf_Node(Node):
    """None"""

    title = 'Erf_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.erf_())



"""
WARNING: Module ErfcNode was generated using fallback option. May contain bugs
"""

class ErfcNode(Node):
    """
erfc(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erfc`.
"""

    title = 'ErfcNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.erfc())



"""
WARNING: Module Erfc_Node was generated using fallback option. May contain bugs
"""

class Erfc_Node(Node):
    """None"""

    title = 'Erfc_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.erfc_())



"""
WARNING: Module ErfinvNode was generated using fallback option. May contain bugs
"""

class ErfinvNode(Node):
    """
erfinv(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erfinv`.
"""

    title = 'ErfinvNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.erfinv())



"""
WARNING: Module ExpNode was generated using fallback option. May contain bugs
"""

class ExpNode(Node):
    """
exp(input, *, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.exp(torch.tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
"""

    title = 'ExpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.exp())



"""
WARNING: Module Exp2Node was generated using fallback option. May contain bugs
"""

class Exp2Node(Node):
    """
exp2(input, *, out=None) -> Tensor

Alias for :func:`torch.special.exp2`.
"""

    title = 'Exp2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.exp2())



"""
WARNING: Module Exp2_Node was generated using fallback option. May contain bugs
"""

class Exp2_Node(Node):
    """None"""

    title = 'Exp2_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.exp2_())



"""
WARNING: Module Exp_Node was generated using fallback option. May contain bugs
"""

class Exp_Node(Node):
    """None"""

    title = 'Exp_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.exp_())



"""
WARNING: Module Expm1Node was generated using fallback option. May contain bugs
"""

class Expm1Node(Node):
    """
expm1(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expm1`.
"""

    title = 'Expm1Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.expm1())



"""
WARNING: Module Expm1_Node was generated using fallback option. May contain bugs
"""

class Expm1_Node(Node):
    """None"""

    title = 'Expm1_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.expm1_())



"""
WARNING: Module EyeNode was generated using fallback option. May contain bugs
"""

class EyeNode(Node):
    """
eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): the number of rows
    m (int, optional): the number of columns with default being :attr:`n`

Keyword arguments:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Returns:
    Tensor: A 2-D tensor with ones on the diagonal and zeros elsewhere

Example::

    >>> torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
"""

    title = 'EyeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.eye())



"""
WARNING: Module Fake_quantize_per_channel_affineNode was generated using fallback option. May contain bugs
"""

class Fake_quantize_per_channel_affineNode(Node):
    """
fake_quantize_per_channel_affine(input, scale, zero_point, quant_min, quant_max) -> Tensor

Returns a new tensor with the data in :attr:`input` fake quantized per channel using :attr:`scale`,
:attr:`zero_point`, :attr:`quant_min` and :attr:`quant_max`, across the channel specified by :attr:`axis`.

.. math::
    \text{output} = min(
        \text{quant\_max},
        max(
            \text{quant\_min},
            \text{std::nearby\_int}(\text{input} / \text{scale}) + \text{zero\_point}
        )
    )

Args:
    input (Tensor): the input value(s), in ``torch.float32``.
    scale (Tensor): quantization scale, per channel
    zero_point (Tensor): quantization zero_point, per channel
    axis (int32): channel axis
    quant_min (int64): lower bound of the quantized domain
    quant_max (int64): upper bound of the quantized domain

Returns:
    Tensor: A newly fake_quantized per channel tensor

Example::

    >>> x = torch.randn(2, 2, 2)
    >>> x
    tensor([[[-0.2525, -0.0466],
             [ 0.3491, -0.2168]],

            [[-0.5906,  1.6258],
             [ 0.6444, -0.0542]]])
    >>> scales = (torch.randn(2) + 1) * 0.05
    >>> scales
    tensor([0.0475, 0.0486])
    >>> zero_points = torch.zeros(2).to(torch.long)
    >>> zero_points
    tensor([0, 0])
    >>> torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)
    tensor([[[0.0000, 0.0000],
             [0.3405, 0.0000]],

            [[0.0000, 1.6134],
            [0.6323, 0.0000]]])
"""

    title = 'Fake_quantize_per_channel_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fake_quantize_per_channel_affine())



"""
WARNING: Module Fake_quantize_per_tensor_affineNode was generated using fallback option. May contain bugs
"""

class Fake_quantize_per_tensor_affineNode(Node):
    """
fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max) -> Tensor

Returns a new tensor with the data in :attr:`input` fake quantized using :attr:`scale`,
:attr:`zero_point`, :attr:`quant_min` and :attr:`quant_max`.

.. math::
    \text{output} = min(
        \text{quant\_max},
        max(
            \text{quant\_min},
            \text{std::nearby\_int}(\text{input} / \text{scale}) + \text{zero\_point}
        )
    )

Args:
    input (Tensor): the input value(s), in ``torch.float32``.
    scale (double): quantization scale
    zero_point (int64): quantization zero_point
    quant_min (int64): lower bound of the quantized domain
    quant_max (int64): upper bound of the quantized domain

Returns:
    Tensor: A newly fake_quantized tensor

Example::

    >>> x = torch.randn(4)
    >>> x
    tensor([ 0.0552,  0.9730,  0.3973, -1.0780])
    >>> torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255)
    tensor([0.1000, 1.0000, 0.4000, 0.0000])
"""

    title = 'Fake_quantize_per_tensor_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fake_quantize_per_tensor_affine())



"""
WARNING: Module Fbgemm_linear_fp16_weightNode was generated using fallback option. May contain bugs
"""

class Fbgemm_linear_fp16_weightNode(Node):
    """None"""

    title = 'Fbgemm_linear_fp16_weightNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_linear_fp16_weight())



"""
WARNING: Module Fbgemm_linear_fp16_weight_fp32_activationNode was generated using fallback option. May contain bugs
"""

class Fbgemm_linear_fp16_weight_fp32_activationNode(Node):
    """None"""

    title = 'Fbgemm_linear_fp16_weight_fp32_activationNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_linear_fp16_weight_fp32_activation())



"""
WARNING: Module Fbgemm_linear_int8_weightNode was generated using fallback option. May contain bugs
"""

class Fbgemm_linear_int8_weightNode(Node):
    """None"""

    title = 'Fbgemm_linear_int8_weightNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_linear_int8_weight())



"""
WARNING: Module Fbgemm_linear_int8_weight_fp32_activationNode was generated using fallback option. May contain bugs
"""

class Fbgemm_linear_int8_weight_fp32_activationNode(Node):
    """None"""

    title = 'Fbgemm_linear_int8_weight_fp32_activationNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_linear_int8_weight_fp32_activation())



"""
WARNING: Module Fbgemm_linear_quantize_weightNode was generated using fallback option. May contain bugs
"""

class Fbgemm_linear_quantize_weightNode(Node):
    """None"""

    title = 'Fbgemm_linear_quantize_weightNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_linear_quantize_weight())



"""
WARNING: Module Fbgemm_pack_gemm_matrix_fp16Node was generated using fallback option. May contain bugs
"""

class Fbgemm_pack_gemm_matrix_fp16Node(Node):
    """None"""

    title = 'Fbgemm_pack_gemm_matrix_fp16Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_pack_gemm_matrix_fp16())



"""
WARNING: Module Fbgemm_pack_quantized_matrixNode was generated using fallback option. May contain bugs
"""

class Fbgemm_pack_quantized_matrixNode(Node):
    """None"""

    title = 'Fbgemm_pack_quantized_matrixNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fbgemm_pack_quantized_matrix())



"""
WARNING: Module Feature_alpha_dropoutNode was generated using fallback option. May contain bugs
"""

class Feature_alpha_dropoutNode(Node):
    """None"""

    title = 'Feature_alpha_dropoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.feature_alpha_dropout())



"""
WARNING: Module Feature_alpha_dropout_Node was generated using fallback option. May contain bugs
"""

class Feature_alpha_dropout_Node(Node):
    """None"""

    title = 'Feature_alpha_dropout_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.feature_alpha_dropout_())



"""
WARNING: Module Feature_dropoutNode was generated using fallback option. May contain bugs
"""

class Feature_dropoutNode(Node):
    """None"""

    title = 'Feature_dropoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.feature_dropout())



"""
WARNING: Module Feature_dropout_Node was generated using fallback option. May contain bugs
"""

class Feature_dropout_Node(Node):
    """None"""

    title = 'Feature_dropout_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.feature_dropout_())



"""
WARNING: Module FftNode was generated using fallback option. May contain bugs
"""

class FftNode(Node):
    """None"""

    title = 'FftNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fft())



"""
WARNING: Module Fill_Node was generated using fallback option. May contain bugs
"""

class Fill_Node(Node):
    """None"""

    title = 'Fill_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fill_())



"""
WARNING: Module FinfoNode was generated using fallback option. May contain bugs
"""

class FinfoNode(Node):
    """None"""

    title = 'FinfoNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.finfo())



"""
WARNING: Module FixNode was generated using fallback option. May contain bugs
"""

class FixNode(Node):
    """
fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`
"""

    title = 'FixNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fix())



"""
WARNING: Module Fix_Node was generated using fallback option. May contain bugs
"""

class Fix_Node(Node):
    """None"""

    title = 'Fix_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fix_())



"""
WARNING: Module FlattenNode was generated using fallback option. May contain bugs
"""

class FlattenNode(Node):
    """
flatten(input, start_dim=0, end_dim=-1) -> Tensor

Flattens :attr:`input` by reshaping it into a one-dimensional tensor. If :attr:`start_dim` or :attr:`end_dim`
are passed, only dimensions starting with :attr:`start_dim` and ending with :attr:`end_dim` are flattened.
The order of elements in :attr:`input` is unchanged.

Unlike NumPy's flatten, which always copies input's data, this function may return the original object, a view,
or copy. If no dimensions are flattened, then the original object :attr:`input` is returned. Otherwise, if input can
be viewed as the flattened shape, then that view is returned. Finally, only if the input cannot be viewed as the
flattened shape is input's data copied. See :meth:`torch.Tensor.view` for details on when a view will be returned.

.. note::
    Flattening a zero-dimensional tensor will return a one-dimensional view.

Args:
    input (Tensor): the input tensor.
    start_dim (int): the first dim to flatten
    end_dim (int): the last dim to flatten

Example::

    >>> t = torch.tensor([[[1, 2],
    ...                    [3, 4]],
    ...                   [[5, 6],
    ...                    [7, 8]]])
    >>> torch.flatten(t)
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
    >>> torch.flatten(t, start_dim=1)
    tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])
"""

    title = 'FlattenNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.flatten())



"""
WARNING: Module FlipNode was generated using fallback option. May contain bugs
"""

class FlipNode(Node):
    """
flip(input, dims) -> Tensor

Reverse the order of a n-D tensor along given axis in dims.

.. note::
    `torch.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`,
    which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
    `torch.flip` is expected to be slower than `np.flip`.

Args:
    input (Tensor): the input tensor.
    dims (a list or tuple): axis to flip on

Example::

    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[ 0,  1],
             [ 2,  3]],

            [[ 4,  5],
             [ 6,  7]]])
    >>> torch.flip(x, [0, 1])
    tensor([[[ 6,  7],
             [ 4,  5]],

            [[ 2,  3],
             [ 0,  1]]])
"""

    title = 'FlipNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.flip())



"""
WARNING: Module FliplrNode was generated using fallback option. May contain bugs
"""

class FliplrNode(Node):
    """
fliplr(input) -> Tensor

Flip tensor in the left/right direction, returning a new tensor.

Flip the entries in each row in the left/right direction.
Columns are preserved, but appear in a different order than before.

Note:
    Requires the tensor to be at least 2-D.

.. note::
    `torch.fliplr` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.fliplr`,
    which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
    `torch.fliplr` is expected to be slower than `np.fliplr`.

Args:
    input (Tensor): Must be at least 2-dimensional.

Example::

    >>> x = torch.arange(4).view(2, 2)
    >>> x
    tensor([[0, 1],
            [2, 3]])
    >>> torch.fliplr(x)
    tensor([[1, 0],
            [3, 2]])
"""

    title = 'FliplrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fliplr())



"""
WARNING: Module FlipudNode was generated using fallback option. May contain bugs
"""

class FlipudNode(Node):
    """
flipud(input) -> Tensor

Flip tensor in the up/down direction, returning a new tensor.

Flip the entries in each column in the up/down direction.
Rows are preserved, but appear in a different order than before.

Note:
    Requires the tensor to be at least 1-D.

.. note::
    `torch.flipud` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flipud`,
    which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
    `torch.flipud` is expected to be slower than `np.flipud`.

Args:
    input (Tensor): Must be at least 1-dimensional.

Example::

    >>> x = torch.arange(4).view(2, 2)
    >>> x
    tensor([[0, 1],
            [2, 3]])
    >>> torch.flipud(x)
    tensor([[2, 3],
            [0, 1]])
"""

    title = 'FlipudNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.flipud())



"""
WARNING: Module FloatNode was generated using fallback option. May contain bugs
"""

class FloatNode(Node):
    """None"""

    title = 'FloatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.float())



"""
WARNING: Module Float16Node was generated using fallback option. May contain bugs
"""

class Float16Node(Node):
    """None"""

    title = 'Float16Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.float16())



"""
WARNING: Module Float32Node was generated using fallback option. May contain bugs
"""

class Float32Node(Node):
    """None"""

    title = 'Float32Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.float32())



"""
WARNING: Module Float64Node was generated using fallback option. May contain bugs
"""

class Float64Node(Node):
    """None"""

    title = 'Float64Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.float64())



"""
WARNING: Module Float_powerNode was generated using fallback option. May contain bugs
"""

class Float_powerNode(Node):
    """
float_power(input, exponent, *, out=None) -> Tensor

Raises :attr:`input` to the power of :attr:`exponent`, elementwise, in double precision.
If neither input is complex returns a ``torch.float64`` tensor,
and if one or more inputs is complex returns a ``torch.complex128`` tensor.

.. note::
    This function always computes in double precision, unlike :func:`torch.pow`,
    which implements more typical :ref:`type promotion <type-promotion-doc>`.
    This is useful when the computation needs to be performed in a wider or more precise dtype,
    or the results of the computation may contain fractional values not representable in the input dtypes,
    like when an integer base is raised to a negative integer exponent.

Args:
    input (Tensor or Number): the base value(s)
    exponent (Tensor or Number): the exponent value(s)

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randint(10, (4,))
    >>> a
    tensor([6, 4, 7, 1])
    >>> torch.float_power(a, 2)
    tensor([36., 16., 49.,  1.], dtype=torch.float64)

    >>> a = torch.arange(1, 5)
    >>> a
    tensor([ 1,  2,  3,  4])
    >>> exp = torch.tensor([2, -3, 4, -5])
    >>> exp
    tensor([ 2, -3,  4, -5])
    >>> torch.float_power(a, exp)
    tensor([1.0000e+00, 1.2500e-01, 8.1000e+01, 9.7656e-04], dtype=torch.float64)
"""

    title = 'Float_powerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.float_power())



"""
WARNING: Module FloorNode was generated using fallback option. May contain bugs
"""

class FloorNode(Node):
    """
floor(input, *, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> torch.floor(a)
    tensor([-1.,  1., -1., -1.])
"""

    title = 'FloorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.floor())



"""
WARNING: Module Floor_Node was generated using fallback option. May contain bugs
"""

class Floor_Node(Node):
    """None"""

    title = 'Floor_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.floor_())



"""
WARNING: Module Floor_divideNode was generated using fallback option. May contain bugs
"""

class Floor_divideNode(Node):
    """
floor_divide(input, other, *, out=None) -> Tensor

.. warning::

    :func:`torch.floor_divide` is deprecated and will be removed in a future PyTorch
    release. Its name is a misnomer because it actually rounds the quotient
    towards zero instead of taking its floor. To keep the current behavior use
    :func:`torch.div` with ``rounding_mode='trunc'``. To actually perform floor
    division, use :func:`torch.div` with ``rounding_mode='floor'``.

Computes :attr:`input` divided by :attr:`other`, elementwise, and rounds each
quotient towards zero. Equivalently, it truncates the quotient(s):

.. math::
    \text{{out}}_i = \text{trunc} \left( \frac{{\text{{input}}_i}}{{\text{{other}}_i}} \right)



Supports broadcasting to a common shape, type promotion, and integer and float inputs.

Args:
    input (Tensor or Number): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([4.0, 3.0])
    >>> b = torch.tensor([2.0, 2.0])
    >>> torch.floor_divide(a, b)
    tensor([2.0, 1.0])
    >>> torch.floor_divide(a, 1.4)
    tensor([2.0, 2.0])
"""

    title = 'Floor_divideNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.floor_divide())



"""
WARNING: Module FmaxNode was generated using fallback option. May contain bugs
"""

class FmaxNode(Node):
    """
fmax(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

This is like :func:`torch.maximum` except it handles NaNs differently:
if exactly one of the two elements being compared is a NaN then the non-NaN element is taken as the maximum.
Only if both elements are NaN is NaN propagated.

This function is a wrapper around C++'s ``std::fmax`` and is similar to NumPy's ``fmax`` function.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and floating-point inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([9.7, float('nan'), 3.1, float('nan')])
    >>> b = torch.tensor([-2.2, 0.5, float('nan'), float('nan')])
    >>> torch.fmax(a, b)
    tensor([9.7000, 0.5000, 3.1000,    nan])
"""

    title = 'FmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fmax())



"""
WARNING: Module FminNode was generated using fallback option. May contain bugs
"""

class FminNode(Node):
    """
fmin(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

This is like :func:`torch.minimum` except it handles NaNs differently:
if exactly one of the two elements being compared is a NaN then the non-NaN element is taken as the minimum.
Only if both elements are NaN is NaN propagated.

This function is a wrapper around C++'s ``std::fmin`` and is similar to NumPy's ``fmin`` function.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and floating-point inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([2.2, float('nan'), 2.1, float('nan')])
    >>> b = torch.tensor([-9.3, 0.1, float('nan'), float('nan')])
    >>> torch.fmin(a, b)
    tensor([-9.3000, 0.1000, 2.1000,    nan])
"""

    title = 'FminNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fmin())



"""
WARNING: Module FmodNode was generated using fallback option. May contain bugs
"""

class FmodNode(Node):
    """
fmod(input, other, *, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend :attr:`input`.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

.. note::

    When the divisor is zero, returns ``NaN`` for floating point dtypes
    on both CPU and GPU; raises ``RuntimeError`` for integer division by
    zero on CPU; Integer division by zero on GPU may return any value.

Args:
    input (Tensor): the dividend
    other (Tensor or Scalar): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([-1., -0., -1.,  1.,  0.,  1.])
    >>> torch.fmod(torch.tensor([1, 2, 3, 4, 5]), 1.5)
    tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000])

"""

    title = 'FmodNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fmod())



"""
WARNING: Module ForkNode was generated using fallback option. May contain bugs
"""

class ForkNode(Node):
    """fork(*args, **kwargs) -> torch._C.Future
"""

    title = 'ForkNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.fork(self.input(0), self.input(1)))



"""
WARNING: Module FracNode was generated using fallback option. May contain bugs
"""

class FracNode(Node):
    """
frac(input, *, out=None) -> Tensor

Computes the fractional portion of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \operatorname{sgn}(\text{input}_{i})

Example::

    >>> torch.frac(torch.tensor([1, 2.5, -3.2]))
    tensor([ 0.0000,  0.5000, -0.2000])
"""

    title = 'FracNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.frac())



"""
WARNING: Module Frac_Node was generated using fallback option. May contain bugs
"""

class Frac_Node(Node):
    """None"""

    title = 'Frac_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.frac_())



"""
WARNING: Module FrexpNode was generated using fallback option. May contain bugs
"""

class FrexpNode(Node):
    """
frexp(input, *, out=None) -> (Tensor mantissa, Tensor exponent)

Decomposes :attr:`input` into mantissa and exponent tensors
such that :math:`\text{input} = \text{mantissa} \times 2^{\text{exponent}}`.

The range of mantissa is the open interval (-1, 1).

Supports float inputs.

Args:
    input (Tensor): the input tensor


Keyword args:
    out (tuple, optional): the output tensors

Example::

    >>> x = torch.arange(9.)
    >>> mantissa, exponent = torch.frexp(x)
    >>> mantissa
    tensor([0.0000, 0.5000, 0.5000, 0.7500, 0.5000, 0.6250, 0.7500, 0.8750, 0.5000])
    >>> exponent
    tensor([0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=torch.int32)
    >>> torch.ldexp(mantissa, exponent)
    tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])
"""

    title = 'FrexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.frexp())



"""
WARNING: Module Frobenius_normNode was generated using fallback option. May contain bugs
"""

class Frobenius_normNode(Node):
    """None"""

    title = 'Frobenius_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.frobenius_norm())



"""
WARNING: Module From_fileNode was generated using fallback option. May contain bugs
"""

class From_fileNode(Node):
    """None"""

    title = 'From_fileNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.from_file())



"""
WARNING: Module From_numpyNode was generated using fallback option. May contain bugs
"""

class From_numpyNode(Node):
    """
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and :attr:`ndarray` share the same memory. Modifications to
the tensor will be reflected in the :attr:`ndarray` and vice versa. The returned
tensor is not resizable.

It currently accepts :attr:`ndarray` with dtypes of ``numpy.float64``,
``numpy.float32``, ``numpy.float16``, ``numpy.complex64``, ``numpy.complex128``,
``numpy.int64``, ``numpy.int32``, ``numpy.int16``, ``numpy.int8``, ``numpy.uint8``,
and ``numpy.bool``.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
"""

    title = 'From_numpyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.from_numpy())



"""
WARNING: Module FullNode was generated using fallback option. May contain bugs
"""

class FullNode(Node):
    """
full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Creates a tensor of size :attr:`size` filled with :attr:`fill_value`. The
tensor's dtype is inferred from :attr:`fill_value`.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    fill_value (Scalar): the value to fill the output tensor with.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.full((2, 3), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416]])
"""

    title = 'FullNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.full())



"""
WARNING: Module Full_likeNode was generated using fallback option. May contain bugs
"""

class Full_likeNode(Node):
    """
full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
``torch.full_like(input, fill_value)`` is equivalent to
``torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    fill_value: the number to fill the output tensor with.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""

    title = 'Full_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.full_like())



"""
WARNING: Module FunctionalNode was generated using fallback option. May contain bugs
"""

class FunctionalNode(Node):
    """None"""

    title = 'FunctionalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.functional())



"""
WARNING: Module FuturesNode was generated using fallback option. May contain bugs
"""

class FuturesNode(Node):
    """None"""

    title = 'FuturesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.futures())



"""
WARNING: Module GatherNode was generated using fallback option. May contain bugs
"""

class GatherNode(Node):
    """
gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

:attr:`input` and :attr:`index` must have the same number of dimensions.
It is also required that ``index.size(d) <= input.size(d)`` for all
dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
Note that ``input`` and ``index`` do not broadcast against each other.

Args:
    input (Tensor): the source tensor
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to gather

Keyword arguments:
    sparse_grad (bool, optional): If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.
    out (Tensor, optional): the destination tensor

Example::

    >>> t = torch.tensor([[1, 2], [3, 4]])
    >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
"""

    title = 'GatherNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gather())



"""
WARNING: Module GcdNode was generated using fallback option. May contain bugs
"""

class GcdNode(Node):
    """
gcd(input, other, *, out=None) -> Tensor

Computes the element-wise greatest common divisor (GCD) of :attr:`input` and :attr:`other`.

Both :attr:`input` and :attr:`other` must have integer types.

.. note::
    This defines :math:`gcd(0, 0) = 0`.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([5, 10, 15])
    >>> b = torch.tensor([3, 4, 5])
    >>> torch.gcd(a, b)
    tensor([1, 2, 5])
    >>> c = torch.tensor([3])
    >>> torch.gcd(a, c)
    tensor([1, 1, 3])
"""

    title = 'GcdNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gcd())



"""
WARNING: Module Gcd_Node was generated using fallback option. May contain bugs
"""

class Gcd_Node(Node):
    """None"""

    title = 'Gcd_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gcd_())



"""
WARNING: Module GeNode was generated using fallback option. May contain bugs
"""

class GeNode(Node):
    """
ge(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \geq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is greater than or equal to :attr:`other` and False elsewhere

Example::

    >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
"""

    title = 'GeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ge())



"""
WARNING: Module GeqrfNode was generated using fallback option. May contain bugs
"""

class GeqrfNode(Node):
    """
geqrf(input, *, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK's geqrf directly. This function
returns a namedtuple (a, tau) as defined in `LAPACK documentation for geqrf`_ .

Computes a QR decomposition of :attr:`input`.
Both `Q` and `R` matrices are stored in the same output tensor `a`.
The elements of `R` are stored on and above the diagonal.
Elementary reflectors (or Householder vectors) implicitly defining matrix `Q`
are stored below the diagonal.
The results of this function can be used together with :func:`torch.linalg.householder_product`
to obtain the `Q` matrix or
with :func:`torch.ormqr`, which uses an implicit representation of the `Q` matrix,
for an efficient matrix-matrix multiplication.

See `LAPACK documentation for geqrf`_ for further details.

.. note::
    See also :func:`torch.linalg.qr`, which computes Q and R matrices, and :func:`torch.linalg.lstsq`
    with the ``driver="gels"`` option for a function that can solve matrix equations using a QR decomposition.

Args:
    input (Tensor): the input matrix

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, Tensor). Ignored if `None`. Default: `None`.

.. _LAPACK documentation for geqrf:
    http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html

"""

    title = 'GeqrfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.geqrf())



"""
WARNING: Module GerNode was generated using fallback option. May contain bugs
"""

class GerNode(Node):
    """
ger(input, vec2, *, out=None) -> Tensor

Alias of :func:`torch.outer`.

.. warning::
    This function is deprecated and will be removed in a future PyTorch release.
    Use :func:`torch.outer` instead.
"""

    title = 'GerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ger())



"""
WARNING: Module Get_default_dtypeNode was generated using fallback option. May contain bugs
"""

class Get_default_dtypeNode(Node):
    """
get_default_dtype() -> torch.dtype

Get the current default floating point :class:`torch.dtype`.

Example::

    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_dtype(torch.float64)
    >>> torch.get_default_dtype()  # default is now changed to torch.float64
    torch.float64
    >>> torch.set_default_tensor_type(torch.FloatTensor)  # setting tensor type also affects this
    >>> torch.get_default_dtype()  # changed to torch.float32, the dtype for torch.FloatTensor
    torch.float32

"""

    title = 'Get_default_dtypeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_default_dtype())



"""
WARNING: Module Get_deviceNode was generated using fallback option. May contain bugs
"""

class Get_deviceNode(Node):
    """None"""

    title = 'Get_deviceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_device())


class Get_file_pathNode(Node):
    """None"""

    title = 'Get_file_pathNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_file_path())



"""
WARNING: Module Get_num_interop_threadsNode was generated using fallback option. May contain bugs
"""

class Get_num_interop_threadsNode(Node):
    """
get_num_interop_threads() -> int

Returns the number of threads used for inter-op parallelism on CPU
(e.g. in JIT interpreter)
"""

    title = 'Get_num_interop_threadsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_num_interop_threads())



"""
WARNING: Module Get_num_threadsNode was generated using fallback option. May contain bugs
"""

class Get_num_threadsNode(Node):
    """
get_num_threads() -> int

Returns the number of threads used for parallelizing CPU operations
"""

    title = 'Get_num_threadsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_num_threads())


class Get_rng_stateNode(Node):
    """Returns the random number generator state as a `torch.ByteTensor`."""

    title = 'Get_rng_stateNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.get_rng_state())



"""
WARNING: Module GradientNode was generated using fallback option. May contain bugs
"""

class GradientNode(Node):
    """
gradient(input, *, spacing=None, dim=None, edge_order=1) -> List of Tensors

This function is analogous to NumPy's gradient function.

Args:
    {input}

Keyword args:
    spacing (scalar, list of scalar, list of Tensor, optional): implicitly or explicitly represents
    the coordinates the function is evaluated at
    dim (int, list of int, optional): the dimension or dimensions to approximate the gradient over.
    edge_order (int, optional): unsupported (must be equal to its default value which is 1.)

Example:

    >>> t = torch.tensor([1, 2, 4, 7, 11, 16], dtype=torch.float)
    >>> torch.gradient(t)
    tensor([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    >>> coords = torch.tensor([0., 1., 1.5, 3.5, 4., 6.], dtype=torch.float)
    >>> torch.gradient(t, spacing=(coords,))
    tensor([1. ,  3. ,  3.5,  6.7,  6.9,  2.5])

"""

    title = 'GradientNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gradient())



"""
WARNING: Module GreaterNode was generated using fallback option. May contain bugs
"""

class GreaterNode(Node):
    """
greater(input, other, *, out=None) -> Tensor

Alias for :func:`torch.gt`.
"""

    title = 'GreaterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.greater())



"""
WARNING: Module Greater_equalNode was generated using fallback option. May contain bugs
"""

class Greater_equalNode(Node):
    """
greater_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ge`.
"""

    title = 'Greater_equalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.greater_equal())



"""
WARNING: Module Grid_samplerNode was generated using fallback option. May contain bugs
"""

class Grid_samplerNode(Node):
    """None"""

    title = 'Grid_samplerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.grid_sampler())



"""
WARNING: Module Grid_sampler_2dNode was generated using fallback option. May contain bugs
"""

class Grid_sampler_2dNode(Node):
    """None"""

    title = 'Grid_sampler_2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.grid_sampler_2d())



"""
WARNING: Module Grid_sampler_3dNode was generated using fallback option. May contain bugs
"""

class Grid_sampler_3dNode(Node):
    """None"""

    title = 'Grid_sampler_3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.grid_sampler_3d())



"""
WARNING: Module Group_normNode was generated using fallback option. May contain bugs
"""

class Group_normNode(Node):
    """None"""

    title = 'Group_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.group_norm())



"""
WARNING: Module GruNode was generated using fallback option. May contain bugs
"""

class GruNode(Node):
    """None"""

    title = 'GruNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gru())



"""
WARNING: Module Gru_cellNode was generated using fallback option. May contain bugs
"""

class Gru_cellNode(Node):
    """None"""

    title = 'Gru_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gru_cell())



"""
WARNING: Module GtNode was generated using fallback option. May contain bugs
"""

class GtNode(Node):
    """
gt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} > \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is greater than :attr:`other` and False elsewhere

Example::

    >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [False, False]])
"""

    title = 'GtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.gt())



"""
WARNING: Module HalfNode was generated using fallback option. May contain bugs
"""

class HalfNode(Node):
    """None"""

    title = 'HalfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.half())



"""
WARNING: Module Hamming_windowNode was generated using fallback option. May contain bugs
"""

class Hamming_windowNode(Node):
    """
hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Hamming window function.

.. math::
    w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hamming_window(L, periodic=True)`` equal to
``torch.hamming_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

.. note::
    This is a generalized version of :meth:`torch.hann_window`.

Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above
    beta (float, optional): The coefficient :math:`\beta` in the equation above

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{window\_length},)` containing the window

"""

    title = 'Hamming_windowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hamming_window())



"""
WARNING: Module Hann_windowNode was generated using fallback option. May contain bugs
"""

class Hann_windowNode(Node):
    """
hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Hann window function.

.. math::
    w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
            \sin^2 \left( \frac{\pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hann_window(L, periodic=True)`` equal to
``torch.hann_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{window\_length},)` containing the window

"""

    title = 'Hann_windowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hann_window())



"""
WARNING: Module HardshrinkNode was generated using fallback option. May contain bugs
"""

class HardshrinkNode(Node):
    """None"""

    title = 'HardshrinkNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hardshrink())



"""
WARNING: Module Has_cudaNode was generated using fallback option. May contain bugs
"""

class Has_cudaNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_cudaNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_cuda(self.input(0)))



"""
WARNING: Module Has_cudnnNode was generated using fallback option. May contain bugs
"""

class Has_cudnnNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_cudnnNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_cudnn(self.input(0)))



"""
WARNING: Module Has_lapackNode was generated using fallback option. May contain bugs
"""

class Has_lapackNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_lapackNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_lapack(self.input(0)))



"""
WARNING: Module Has_mklNode was generated using fallback option. May contain bugs
"""

class Has_mklNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_mklNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_mkl(self.input(0)))



"""
WARNING: Module Has_mkldnnNode was generated using fallback option. May contain bugs
"""

class Has_mkldnnNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_mkldnnNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_mkldnn(self.input(0)))



"""
WARNING: Module Has_mlcNode was generated using fallback option. May contain bugs
"""

class Has_mlcNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_mlcNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_mlc(self.input(0)))



"""
WARNING: Module Has_openmpNode was generated using fallback option. May contain bugs
"""

class Has_openmpNode(Node):
    """bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed."""

    title = 'Has_openmpNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.has_openmp(self.input(0)))



"""
WARNING: Module HeavisideNode was generated using fallback option. May contain bugs
"""

class HeavisideNode(Node):
    """
heaviside(input, values, *, out=None) -> Tensor

Computes the Heaviside step function for each element in :attr:`input`.
The Heaviside step function is defined as:

.. math::
    \text{{heaviside}}(input, values) = \begin{cases}
        0, & \text{if input < 0}\\
        values, & \text{if input == 0}\\
        1, & \text{if input > 0}
    \end{cases}


Args:
    input (Tensor): the input tensor.
    values (Tensor): The values to use where :attr:`input` is zero.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> input = torch.tensor([-1.5, 0, 2.0])
    >>> values = torch.tensor([0.5])
    >>> torch.heaviside(input, values)
    tensor([0.0000, 0.5000, 1.0000])
    >>> values = torch.tensor([1.2, -2.0, 3.5])
    >>> torch.heaviside(input, values)
    tensor([0., -2., 1.])

"""

    title = 'HeavisideNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.heaviside())



"""
WARNING: Module Hinge_embedding_lossNode was generated using fallback option. May contain bugs
"""

class Hinge_embedding_lossNode(Node):
    """None"""

    title = 'Hinge_embedding_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hinge_embedding_loss())



"""
WARNING: Module HistcNode was generated using fallback option. May contain bugs
"""

class HistcNode(Node):
    """
histc(input, bins=100, min=0, max=0, *, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between :attr:`min` and
:attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
maximum values of the data are used.

Elements lower than min and higher than max are ignored.

Args:
    input (Tensor): the input tensor.
    bins (int): number of histogram bins
    min (int): lower end of the range (inclusive)
    max (int): upper end of the range (inclusive)

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    Tensor: Histogram represented as a tensor

Example::

    >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
    tensor([ 0.,  2.,  1.,  0.])
"""

    title = 'HistcNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.histc())



"""
WARNING: Module HsmmNode was generated using fallback option. May contain bugs
"""

class HsmmNode(Node):
    """None"""

    title = 'HsmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hsmm())



"""
WARNING: Module HsplitNode was generated using fallback option. May contain bugs
"""

class HsplitNode(Node):
    """
hsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with one or more dimensions, into multiple tensors
horizontally according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

If :attr:`input` is one dimensional this is equivalent to calling
torch.tensor_split(input, indices_or_sections, dim=0) (the split dimension is
zero), and if :attr:`input` has two or more dimensions it's equivalent to calling
torch.tensor_split(input, indices_or_sections, dim=1) (the split dimension is 1),
except that if :attr:`indices_or_sections` is an integer it must evenly divide
the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.hsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (Tensor, int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::
    >>> t = torch.arange(16.0).reshape(4,4)
    >>> t
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])
    >>> torch.hsplit(t, 2)
    (tensor([[ 0.,  1.],
             [ 4.,  5.],
             [ 8.,  9.],
             [12., 13.]]),
     tensor([[ 2.,  3.],
             [ 6.,  7.],
             [10., 11.],
             [14., 15.]]))
    >>> torch.hsplit(t, [3, 6])
    (tensor([[ 0.,  1.,  2.],
             [ 4.,  5.,  6.],
             [ 8.,  9., 10.],
             [12., 13., 14.]]),
     tensor([[ 3.],
             [ 7.],
             [11.],
             [15.]]),
     tensor([], size=(4, 0)))

"""

    title = 'HsplitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hsplit())



"""
WARNING: Module HspmmNode was generated using fallback option. May contain bugs
"""

class HspmmNode(Node):
    """
hspmm(mat1, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of a :ref:`sparse COO matrix
<sparse-coo-docs>` :attr:`mat1` and a strided matrix :attr:`mat2`. The
result is a (1 + 1)-dimensional :ref:`hybrid COO matrix
<sparse-hybrid-coo-docs>`.

Args:
    mat1 (Tensor): the first sparse matrix to be matrix multiplied
    mat2 (Tensor): the second strided matrix to be matrix multiplied

Keyword args:
    {out}
"""

    title = 'HspmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hspmm())



"""
WARNING: Module HstackNode was generated using fallback option. May contain bugs
"""

class HstackNode(Node):
    """
hstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.hstack((a,b))
    tensor([1, 2, 3, 4, 5, 6])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.hstack((a,b))
    tensor([[1, 4],
            [2, 5],
            [3, 6]])

"""

    title = 'HstackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hstack())



"""
WARNING: Module HubNode was generated using fallback option. May contain bugs
"""

class HubNode(Node):
    """None"""

    title = 'HubNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hub())



"""
WARNING: Module HypotNode was generated using fallback option. May contain bugs
"""

class HypotNode(Node):
    """
hypot(input, other, *, out=None) -> Tensor

Given the legs of a right triangle, return its hypotenuse.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}^{2} + \text{other}_{i}^{2}}

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0]))
    tensor([5.0000, 5.6569, 6.4031])

"""

    title = 'HypotNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.hypot())



"""
WARNING: Module I0Node was generated using fallback option. May contain bugs
"""

class I0Node(Node):
    """
i0(input, *, out=None) -> Tensor

Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

.. math::
    \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}


Args:
    input (Tensor): the input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.i0(torch.arange(5, dtype=torch.float32))
    tensor([ 1.0000,  1.2661,  2.2796,  4.8808, 11.3019])

"""

    title = 'I0Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.i0())



"""
WARNING: Module I0_Node was generated using fallback option. May contain bugs
"""

class I0_Node(Node):
    """None"""

    title = 'I0_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.i0_())



"""
WARNING: Module IgammaNode was generated using fallback option. May contain bugs
"""

class IgammaNode(Node):
    """
igamma(input, other, *, out=None) -> Tensor

Computes the regularized lower incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_0^{\text{other}_i} t^{\text{input}_i-1} e^{-t} dt

where both :math:`\text{input}_i` and :math:`\text{other}_i` are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then :math:`\text{out}_i=\text{nan}`.
:math:`\Gamma(\cdot)` in the equation above is the gamma function,

.. math::
    \Gamma(\text{input}_i) = \int_0^\infty t^{(\text{input}_i-1)} e^{-t} dt.

See :func:`torch.igammac` and :func:`torch.lgamma` for related functions.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`
and float inputs.

.. note::
    The backward pass with respect to :attr:`input` is not yet supported.
    Please open an issue on PyTorch's Github to request it.


Args:
    input (Tensor): the first non-negative input tensor
    other (Tensor): the second non-negative input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a1 = torch.tensor([4.0])
    >>> a2 = torch.tensor([3.0, 4.0, 5.0])
    >>> a = torch.igammac(a1, a2)
    tensor([0.3528, 0.5665, 0.7350])
    tensor([0.3528, 0.5665, 0.7350])
    >>> b = torch.igamma(a1, a2) + torch.igammac(a1, a2)
    tensor([1., 1., 1.])

"""

    title = 'IgammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.igamma())



"""
WARNING: Module IgammacNode was generated using fallback option. May contain bugs
"""

class IgammacNode(Node):
    """
igammac(input, other, *, out=None) -> Tensor

Computes the regularized upper incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_{\text{other}_i}^{\infty} t^{\text{input}_i-1} e^{-t} dt

where both :math:`\text{input}_i` and :math:`\text{other}_i` are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then :math:`\text{out}_i=\text{nan}`.
:math:`\Gamma(\cdot)` in the equation above is the gamma function,

.. math::
    \Gamma(\text{input}_i) = \int_0^\infty t^{(\text{input}_i-1)} e^{-t} dt.

See :func:`torch.igamma` and :func:`torch.lgamma` for related functions.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`
and float inputs.

.. note::
    The backward pass with respect to :attr:`input` is not yet supported.
    Please open an issue on PyTorch's Github to request it.


Args:
    input (Tensor): the first non-negative input tensor
    other (Tensor): the second non-negative input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a1 = torch.tensor([4.0])
    >>> a2 = torch.tensor([3.0, 4.0, 5.0])
    >>> a = torch.igammac(a1, a2)
    tensor([0.6472, 0.4335, 0.2650])
    >>> b = torch.igamma(a1, a2) + torch.igammac(a1, a2)
    tensor([1., 1., 1.])

"""

    title = 'IgammacNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.igammac())



"""
WARNING: Module IinfoNode was generated using fallback option. May contain bugs
"""

class IinfoNode(Node):
    """None"""

    title = 'IinfoNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.iinfo())



"""
WARNING: Module ImagNode was generated using fallback option. May contain bugs
"""

class ImagNode(Node):
    """
imag(input) -> Tensor

Returns a new tensor containing imaginary values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

.. warning::
    :func:`imag` is only supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.imag
    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])

"""

    title = 'ImagNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.imag())



"""
WARNING: Module Import_ir_moduleNode was generated using fallback option. May contain bugs
"""

class Import_ir_moduleNode(Node):
    """import_ir_module(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict) -> torch._C.ScriptModule
"""

    title = 'Import_ir_moduleNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
NodeInputBP('c'),
NodeInputBP('d'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.import_ir_module(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module Import_ir_module_from_bufferNode was generated using fallback option. May contain bugs
"""

class Import_ir_module_from_bufferNode(Node):
    """import_ir_module_from_buffer(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict) -> torch._C.ScriptModule
"""

    title = 'Import_ir_module_from_bufferNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
NodeInputBP('c'),
NodeInputBP('d'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.import_ir_module_from_buffer(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module Index_addNode was generated using fallback option. May contain bugs
"""

class Index_addNode(Node):
    """None"""

    title = 'Index_addNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_add())



"""
WARNING: Module Index_copyNode was generated using fallback option. May contain bugs
"""

class Index_copyNode(Node):
    """None"""

    title = 'Index_copyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_copy())



"""
WARNING: Module Index_fillNode was generated using fallback option. May contain bugs
"""

class Index_fillNode(Node):
    """None"""

    title = 'Index_fillNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_fill())



"""
WARNING: Module Index_putNode was generated using fallback option. May contain bugs
"""

class Index_putNode(Node):
    """None"""

    title = 'Index_putNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_put())



"""
WARNING: Module Index_put_Node was generated using fallback option. May contain bugs
"""

class Index_put_Node(Node):
    """None"""

    title = 'Index_put_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_put_())



"""
WARNING: Module Index_selectNode was generated using fallback option. May contain bugs
"""

class Index_selectNode(Node):
    """
index_select(input, dim, index, *, out=None) -> Tensor

Returns a new tensor which indexes the :attr:`input` tensor along dimension
:attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.

The returned tensor has the same number of dimensions as the original tensor
(:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
of :attr:`index`; other dimensions have the same size as in the original tensor.

.. note:: The returned tensor does **not** use the same storage as the original
          tensor.  If :attr:`out` has a different shape than expected, we
          silently change it to the correct shape, reallocating the underlying
          storage if necessary.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension in which we index
    index (IntTensor or LongTensor): the 1-D tensor containing the indices to index

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> torch.index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
"""

    title = 'Index_selectNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.index_select())


class Inference_modeNode(Node):
    """Context-manager that enables or disables inference mode

    InferenceMode is a new context manager analogous to :class:`~no_grad`
    to be used when you are certain your operations will have no interactions
    with autograd (e.g., model training). Code run under this mode gets better
    performance by disabling view tracking and version counter bumps.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. note::
        Inference mode is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Args:
        mode (bool): Flag whether to enable or disable inference mode

    Example::
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...   y = x * x
        >>> y.requires_grad
        False
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...   return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False

    """

    title = 'Inference_modeNode'
    init_inputs = [
        NodeInputBP('self'),
NodeInputBP('mode'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.inference_mode(self.input(0), self.input(1)))



"""
WARNING: Module Init_num_threadsNode was generated using fallback option. May contain bugs
"""

class Init_num_threadsNode(Node):
    """init_num_threads() -> None


init_num_threads()

Initializes the number of parallel threads used on the current thread.

Call this whenever a new thread is created in order to propagate values from
:func:`torch.set_num_threads` onto the new thread.

"""

    title = 'Init_num_threadsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.init_num_threads())


class Initial_seedNode(Node):
    """Returns the initial seed for generating random numbers as a
    Python `long`.
    """

    title = 'Initial_seedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.initial_seed())



"""
WARNING: Module InnerNode was generated using fallback option. May contain bugs
"""

class InnerNode(Node):
    """
inner(input, other, *, out=None) -> Tensor

Computes the dot product for 1D tensors. For higher dimensions, sums the product
of elements from :attr:`input` and :attr:`other` along their last dimension.

.. note::

    If either :attr:`input` or :attr:`other` is a scalar, the result is equivalent
    to `torch.mul(input, other)`.

    If both :attr:`input` and :attr:`other` are non-scalars, the size of their last
    dimension must match and the result is equivalent to `torch.tensordot(input,
    other, dims=([-1], [-1]))`

Args:
    input (Tensor): First input tensor
    other (Tensor): Second input tensor

Keyword args:
    out (Tensor, optional): Optional output tensor to write result into. The output
                            shape is `input.shape[:-1] + other.shape[:-1]`.

Example::

    # Dot product
    >>> torch.inner(torch.tensor([1, 2, 3]), torch.tensor([0, 2, 1]))
    tensor(7)

    # Multidimensional input tensors
    >>> a = torch.randn(2, 3)
    >>> a
    tensor([[0.8173, 1.0874, 1.1784],
            [0.3279, 0.1234, 2.7894]])
    >>> b = torch.randn(2, 4, 3)
    >>> b
    tensor([[[-0.4682, -0.7159,  0.1506],
            [ 0.4034, -0.3657,  1.0387],
            [ 0.9892, -0.6684,  0.1774],
            [ 0.9482,  1.3261,  0.3917]],

            [[ 0.4537,  0.7493,  1.1724],
            [ 0.2291,  0.5749, -0.2267],
            [-0.7920,  0.3607, -0.3701],
            [ 1.3666, -0.5850, -1.7242]]])
    >>> torch.inner(a, b)
    tensor([[[-0.9837,  1.1560,  0.2907,  2.6785],
            [ 2.5671,  0.5452, -0.6912, -1.5509]],

            [[ 0.1782,  2.9843,  0.7366,  1.5672],
            [ 3.5115, -0.4864, -1.2476, -4.4337]]])

    # Scalar input
    >>> torch.inner(a, torch.tensor(2))
    tensor([[1.6347, 2.1748, 2.3567],
            [0.6558, 0.2469, 5.5787]])
"""

    title = 'InnerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.inner())



"""
WARNING: Module Instance_normNode was generated using fallback option. May contain bugs
"""

class Instance_normNode(Node):
    """None"""

    title = 'Instance_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.instance_norm())



"""
WARNING: Module IntNode was generated using fallback option. May contain bugs
"""

class IntNode(Node):
    """None"""

    title = 'IntNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int())



"""
WARNING: Module Int16Node was generated using fallback option. May contain bugs
"""

class Int16Node(Node):
    """None"""

    title = 'Int16Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int16())



"""
WARNING: Module Int32Node was generated using fallback option. May contain bugs
"""

class Int32Node(Node):
    """None"""

    title = 'Int32Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int32())



"""
WARNING: Module Int64Node was generated using fallback option. May contain bugs
"""

class Int64Node(Node):
    """None"""

    title = 'Int64Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int64())



"""
WARNING: Module Int8Node was generated using fallback option. May contain bugs
"""

class Int8Node(Node):
    """None"""

    title = 'Int8Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int8())



"""
WARNING: Module Int_reprNode was generated using fallback option. May contain bugs
"""

class Int_reprNode(Node):
    """None"""

    title = 'Int_reprNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.int_repr())



"""
WARNING: Module InverseNode was generated using fallback option. May contain bugs
"""

class InverseNode(Node):
    """
inverse(input, *, out=None) -> Tensor

Alias for :func:`torch.linalg.inv`
"""

    title = 'InverseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.inverse())



"""
WARNING: Module Is_anomaly_enabledNode was generated using fallback option. May contain bugs
"""

class Is_anomaly_enabledNode(Node):
    """None"""

    title = 'Is_anomaly_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_anomaly_enabled())



"""
WARNING: Module Is_autocast_enabledNode was generated using fallback option. May contain bugs
"""

class Is_autocast_enabledNode(Node):
    """None"""

    title = 'Is_autocast_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_autocast_enabled())



"""
WARNING: Module Is_complexNode was generated using fallback option. May contain bugs
"""

class Is_complexNode(Node):
    """
is_complex(input) -> (bool)

Returns True if the data type of :attr:`input` is a complex data type i.e.,
one of ``torch.complex64``, and ``torch.complex128``.

Args:
    input (Tensor): the input tensor.
"""

    title = 'Is_complexNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_complex())


class Is_deterministicNode(Node):
    """This function is deprecated and will be removed in a future release.
    Please use :func:`torch.are_deterministic_algorithms_enabled` instead.
    """

    title = 'Is_deterministicNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_deterministic())



"""
WARNING: Module Is_distributedNode was generated using fallback option. May contain bugs
"""

class Is_distributedNode(Node):
    """None"""

    title = 'Is_distributedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_distributed())



"""
WARNING: Module Is_floating_pointNode was generated using fallback option. May contain bugs
"""

class Is_floating_pointNode(Node):
    """
is_floating_point(input) -> (bool)

Returns True if the data type of :attr:`input` is a floating point data type i.e.,
one of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.

Args:
    input (Tensor): the input tensor.
"""

    title = 'Is_floating_pointNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_floating_point())



"""
WARNING: Module Is_grad_enabledNode was generated using fallback option. May contain bugs
"""

class Is_grad_enabledNode(Node):
    """
is_grad_enabled() -> (bool)

Returns True if grad mode is currently enabled.
"""

    title = 'Is_grad_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_grad_enabled())



"""
WARNING: Module Is_inference_mode_enabledNode was generated using fallback option. May contain bugs
"""

class Is_inference_mode_enabledNode(Node):
    """
is_inference_mode_enabled() -> (bool)

Returns True if inference mode is currently enabled.
"""

    title = 'Is_inference_mode_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_inference_mode_enabled())



"""
WARNING: Module Is_nonzeroNode was generated using fallback option. May contain bugs
"""

class Is_nonzeroNode(Node):
    """
is_nonzero(input) -> (bool)

Returns True if the :attr:`input` is a single element tensor which is not equal to zero
after type conversions.
i.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor([0])`` or
``torch.tensor([False])``.
Throws a ``RuntimeError`` if ``torch.numel() != 1`` (even in case
of sparse tensors).

Args:
    input (Tensor): the input tensor.

Examples::

    >>> torch.is_nonzero(torch.tensor([0.]))
    False
    >>> torch.is_nonzero(torch.tensor([1.5]))
    True
    >>> torch.is_nonzero(torch.tensor([False]))
    False
    >>> torch.is_nonzero(torch.tensor([3]))
    True
    >>> torch.is_nonzero(torch.tensor([1, 3, 5]))
    Traceback (most recent call last):
    ...
    RuntimeError: bool value of Tensor with more than one value is ambiguous
    >>> torch.is_nonzero(torch.tensor([]))
    Traceback (most recent call last):
    ...
    RuntimeError: bool value of Tensor with no values is ambiguous
"""

    title = 'Is_nonzeroNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_nonzero())



"""
WARNING: Module Is_same_sizeNode was generated using fallback option. May contain bugs
"""

class Is_same_sizeNode(Node):
    """None"""

    title = 'Is_same_sizeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_same_size())



"""
WARNING: Module Is_signedNode was generated using fallback option. May contain bugs
"""

class Is_signedNode(Node):
    """None"""

    title = 'Is_signedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_signed())


class Is_storageNode(Node):
    """Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    """

    title = 'Is_storageNode'
    init_inputs = [
        NodeInputBP('obj'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_storage(self.input(0)))


class Is_tensorNode(Node):
    """Returns True if `obj` is a PyTorch tensor.

    Note that this function is simply doing ``isinstance(obj, Tensor)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_tensor``.

    Args:
        obj (Object): Object to test
    Example::

        >>> x=torch.tensor([1,2,3])
        >>> torch.is_tensor(x)
        True

    """

    title = 'Is_tensorNode'
    init_inputs = [
        NodeInputBP('obj'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_tensor(self.input(0)))



"""
WARNING: Module Is_vulkan_availableNode was generated using fallback option. May contain bugs
"""

class Is_vulkan_availableNode(Node):
    """None"""

    title = 'Is_vulkan_availableNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_vulkan_available())


class Is_warn_always_enabledNode(Node):
    """Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """

    title = 'Is_warn_always_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.is_warn_always_enabled())



"""
WARNING: Module IscloseNode was generated using fallback option. May contain bugs
"""

class IscloseNode(Node):
    """
isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

Returns a new tensor with boolean elements representing if each element of
:attr:`input` is "close" to the corresponding element of :attr:`other`.
Closeness is defined as:

.. math::
    \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert


where :attr:`input` and :attr:`other` are finite. Where :attr:`input`
and/or :attr:`other` are nonfinite they are close if and only if
they are equal, with NaNs being considered equal to each other when
:attr:`equal_nan` is True.

Args:
    input (Tensor): first tensor to compare
    other (Tensor): second tensor to compare
    atol (float, optional): absolute tolerance. Default: 1e-08
    rtol (float, optional): relative tolerance. Default: 1e-05
    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

Examples::

    >>> torch.isclose(torch.tensor((1., 2, 3)), torch.tensor((1 + 1e-10, 3, 4)))
    tensor([ True, False, False])
    >>> torch.isclose(torch.tensor((float('inf'), 4)), torch.tensor((float('inf'), 6)), rtol=.5)
    tensor([True, True])
"""

    title = 'IscloseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isclose())



"""
WARNING: Module IsfiniteNode was generated using fallback option. May contain bugs
"""

class IsfiniteNode(Node):
    """
isfinite(input) -> Tensor

Returns a new tensor with boolean elements representing if each element is `finite` or not.

Real values are finite when they are not NaN, negative infinity, or infinity.
Complex values are finite when both their real and imaginary parts are finite.

Args:
    input (Tensor): the input tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is finite and False elsewhere

Example::

    >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([True,  False,  True,  False,  False])
"""

    title = 'IsfiniteNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isfinite())



"""
WARNING: Module IsinfNode was generated using fallback option. May contain bugs
"""

class IsinfNode(Node):
    """
isinf(input) -> Tensor

Tests if each element of :attr:`input` is infinite
(positive or negative infinity) or not.

.. note::
    Complex values are infinite when their real or imaginary part is
    infinite.

Args:
    {input}

Returns:
    A boolean tensor that is True where :attr:`input` is infinite and False elsewhere

Example::

    >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([False,  True,  False,  True,  False])
"""

    title = 'IsinfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isinf())



"""
WARNING: Module IsnanNode was generated using fallback option. May contain bugs
"""

class IsnanNode(Node):
    """
isnan(input) -> Tensor

Returns a new tensor with boolean elements representing if each element of :attr:`input`
is NaN or not. Complex values are considered NaN when either their real
and/or imaginary part is NaN.

Arguments:
    input (Tensor): the input tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is NaN and False elsewhere

Example::

    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
    tensor([False, True, False])
"""

    title = 'IsnanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isnan())



"""
WARNING: Module IsneginfNode was generated using fallback option. May contain bugs
"""

class IsneginfNode(Node):
    """
isneginf(input, *, out=None) -> Tensor
Tests if each element of :attr:`input` is negative infinity or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([-float('inf'), float('inf'), 1.2])
    >>> torch.isneginf(a)
    tensor([ True, False, False])
"""

    title = 'IsneginfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isneginf())



"""
WARNING: Module IsposinfNode was generated using fallback option. May contain bugs
"""

class IsposinfNode(Node):
    """
isposinf(input, *, out=None) -> Tensor
Tests if each element of :attr:`input` is positive infinity or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([-float('inf'), float('inf'), 1.2])
    >>> torch.isposinf(a)
    tensor([False,  True, False])
"""

    title = 'IsposinfNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isposinf())



"""
WARNING: Module IsrealNode was generated using fallback option. May contain bugs
"""

class IsrealNode(Node):
    """
isreal(input) -> Tensor

Returns a new tensor with boolean elements representing if each element of :attr:`input` is real-valued or not.
All real-valued types are considered real. Complex values are considered real when their imaginary part is 0.

Arguments:
    input (Tensor): the input tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is real and False elsewhere

Example::

    >>> torch.isreal(torch.tensor([1, 1+1j, 2+0j]))
    tensor([True, False, True])
"""

    title = 'IsrealNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.isreal())


class IstftNode(Node):
    """Inverse short time Fourier Transform. This is expected to be the inverse of :func:`~torch.stft`.
    It has the same parameters (+ additional optional parameter of :attr:`length`) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).

    Important consideration in the parameters :attr:`window` and :attr:`center` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} |w|^2[n-t\times hop\_length] \cancel{=} 0`.

    Since :func:`~torch.stft` discards elements at the end of the signal if they do not fit in a frame,
    ``istft`` may return a shorter signal than the original signal (can occur if :attr:`center` is False
    since the signal isn't padded).

    If :attr:`center` is ``True``, then there will be padding e.g. ``'constant'``, ``'reflect'``, etc.
    Left padding can be trimmed off exactly because they can be calculated but right padding cannot be
    calculated without additional information.

    Example: Suppose the last window is:
    ``[17, 18, 0, 0, 0]`` vs ``[18, 0, 0, 0, 0]``

    The :attr:`n_fft`, :attr:`hop_length`, :attr:`win_length` are all the same which prevents the calculation
    of right padding. These additional values could be zeros or a reflection of the signal so providing
    :attr:`length` could be useful. If :attr:`length` is ``None`` then padding will be aggressively removed
    (some loss of signal).

    [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
    IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

    Args:
        input (Tensor): The input tensor. Expected to be output of :func:`~torch.stft`,
            can either be complex (``channel``, ``fft_size``, ``n_frame``), or real
            (``channel``, ``fft_size``, ``n_frame``, 2) where the ``channel``
            dimension is optional.

            .. deprecated:: 1.8.0
               Real input is deprecated, use complex inputs as returned by
               ``stft(..., return_complex=True)`` instead.
        n_fft (int): Size of Fourier transform
        hop_length (Optional[int]): The distance between neighboring sliding window frames.
            (Default: ``n_fft // 4``)
        win_length (Optional[int]): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Optional[torch.Tensor]): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool): Whether :attr:`input` was padded on both sides so that the :math:`t`-th frame is
            centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        normalized (bool): Whether the STFT was normalized. (Default: ``False``)
        onesided (Optional[bool]): Whether the STFT was onesided.
            (Default: ``True`` if ``n_fft != fft_size`` in the input size)
        length (Optional[int]): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)
        return_complex (Optional[bool]):
            Whether the output should be complex, or if the input should be
            assumed to derive from a real signal and window.
            Note that this is incompatible with ``onesided=True``.
            (Default: ``False``)

    Returns:
        Tensor: Least squares estimation of the original signal of size (..., signal_length)
    """

    title = 'IstftNode'
    init_inputs = [
        NodeInputBP('input'),
NodeInputBP('n_fft'),
NodeInputBP('hop_length'),
NodeInputBP('win_length'),
NodeInputBP('window'),
NodeInputBP('center'),
NodeInputBP('normalized'),
NodeInputBP('onesided'),
NodeInputBP('length'),
NodeInputBP('return_complex'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.istft(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4), self.input(5), self.input(6), self.input(7), self.input(8), self.input(9)))



"""
WARNING: Module JitNode was generated using fallback option. May contain bugs
"""

class JitNode(Node):
    """None"""

    title = 'JitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.jit())



"""
WARNING: Module Kaiser_windowNode was generated using fallback option. May contain bugs
"""

class Kaiser_windowNode(Node):
    """
kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the Kaiser window with window length :attr:`window_length` and shape parameter :attr:`beta`.

Let I_0 be the zeroth order modified Bessel function of the first kind (see :func:`torch.i0`) and
``N = L - 1`` if :attr:`periodic` is False and ``L`` if :attr:`periodic` is True,
where ``L`` is the :attr:`window_length`. This function computes:

.. math::
    out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

Calling ``torch.kaiser_window(L, B, periodic=True)`` is equivalent to calling
``torch.kaiser_window(L + 1, B, periodic=False)[:-1])``.
The :attr:`periodic` argument is intended as a helpful shorthand
to produce a periodic window as input to functions like :func:`torch.stft`.

.. note::
    If :attr:`window_length` is one, then the returned window is a single element tensor containing a one.


Args:
    window_length (int): length of the window.
    periodic (bool, optional): If True, returns a periodic window suitable for use in spectral analysis.
        If False, returns a symmetric window suitable for use in filter design.
    beta (float, optional): shape parameter for the window.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

"""

    title = 'Kaiser_windowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.kaiser_window())



"""
WARNING: Module Kl_divNode was generated using fallback option. May contain bugs
"""

class Kl_divNode(Node):
    """None"""

    title = 'Kl_divNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.kl_div())



"""
WARNING: Module KronNode was generated using fallback option. May contain bugs
"""

class KronNode(Node):
    """
kron(input, other, *, out=None) -> Tensor

Computes the Kronecker product, denoted by :math:`\otimes`, of :attr:`input` and :attr:`other`.

If :attr:`input` is a :math:`(a_0 \times a_1 \times \dots \times a_n)` tensor and :attr:`other` is a
:math:`(b_0 \times b_1 \times \dots \times b_n)` tensor, the result will be a
:math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` tensor with the following entries:

.. math::
    (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
        \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

where :math:`k_t = i_t * b_t + j_t` for :math:`0 \leq t \leq n`.
If one tensor has fewer dimensions than the other it is unsqueezed until it has the same number of dimensions.

Supports real-valued and complex-valued inputs.

.. note::
    This function generalizes the typical definition of the Kronecker product for two matrices to two tensors,
    as described above. When :attr:`input` is a :math:`(m \times n)` matrix and :attr:`other` is a
    :math:`(p \times q)` matrix, the result will be a :math:`(p*m \times q*n)` block matrix:

    .. math::
        \mathbf{A} \otimes \mathbf{B}=\begin{bmatrix}
        a_{11} \mathbf{B} & \cdots & a_{1 n} \mathbf{B} \\
        \vdots & \ddots & \vdots \\
        a_{m 1} \mathbf{B} & \cdots & a_{m n} \mathbf{B} \end{bmatrix}

    where :attr:`input` is :math:`\mathbf{A}` and :attr:`other` is :math:`\mathbf{B}`.

Arguments:
    input (Tensor)
    other (Tensor)

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> mat1 = torch.eye(2)
    >>> mat2 = torch.ones(2, 2)
    >>> torch.kron(mat1, mat2)
    tensor([[1., 1., 0., 0.],
            [1., 1., 0., 0.],
            [0., 0., 1., 1.],
            [0., 0., 1., 1.]])

    >>> mat1 = torch.eye(2)
    >>> mat2 = torch.arange(1, 5).reshape(2, 2)
    >>> torch.kron(mat1, mat2)
    tensor([[1., 2., 0., 0.],
            [3., 4., 0., 0.],
            [0., 0., 1., 2.],
            [0., 0., 3., 4.]])
"""

    title = 'KronNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.kron())



"""
WARNING: Module KthvalueNode was generated using fallback option. May contain bugs
"""

class KthvalueNode(Node):
    """
kthvalue(input, k, dim=None, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th
smallest element of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each element found.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`keepdim` is ``True``, both the :attr:`values` and :attr:`indices` tensors
are the same size as :attr:`input`, except in the dimension :attr:`dim` where
they are of size 1. Otherwise, :attr:`dim` is squeezed
(see :func:`torch.squeeze`), resulting in both the :attr:`values` and
:attr:`indices` tensors having 1 fewer dimension than the :attr:`input` tensor.

.. note::
    When :attr:`input` is a CUDA tensor and there are multiple valid
    :attr:`k` th values, this function may nondeterministically return
    :attr:`indices` for any of them.

Args:
    input (Tensor): the input tensor.
    k (int): k for the k-th smallest element
    dim (int, optional): the dimension to find the kth value along
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.kthvalue(x, 4)
    torch.return_types.kthvalue(values=tensor(4.), indices=tensor(3))

    >>> x=torch.arange(1.,7.).resize_(2,3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.]])
    >>> torch.kthvalue(x, 2, 0, True)
    torch.return_types.kthvalue(values=tensor([[4., 5., 6.]]), indices=tensor([[1, 1, 1]]))
"""

    title = 'KthvalueNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.kthvalue())



"""
WARNING: Module Layer_normNode was generated using fallback option. May contain bugs
"""

class Layer_normNode(Node):
    """None"""

    title = 'Layer_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.layer_norm())


class LayoutNode(Node):
    """None"""

    title = 'LayoutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.layout())



"""
WARNING: Module LcmNode was generated using fallback option. May contain bugs
"""

class LcmNode(Node):
    """
lcm(input, other, *, out=None) -> Tensor

Computes the element-wise least common multiple (LCM) of :attr:`input` and :attr:`other`.

Both :attr:`input` and :attr:`other` must have integer types.

.. note::
    This defines :math:`lcm(0, 0) = 0` and :math:`lcm(0, a) = 0`.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([5, 10, 15])
    >>> b = torch.tensor([3, 4, 5])
    >>> torch.lcm(a, b)
    tensor([15, 20, 15])
    >>> c = torch.tensor([3])
    >>> torch.lcm(a, c)
    tensor([15, 30, 15])
"""

    title = 'LcmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lcm())



"""
WARNING: Module Lcm_Node was generated using fallback option. May contain bugs
"""

class Lcm_Node(Node):
    """None"""

    title = 'Lcm_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lcm_())



"""
WARNING: Module LdexpNode was generated using fallback option. May contain bugs
"""

class LdexpNode(Node):
    """
ldexp(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by 2**:attr:`other`.

.. math::
    \text{{out}}_i = \text{{input}}_i * 2^\text{{other}}_i


Typically this function is used to construct floating point numbers by multiplying
mantissas in :attr:`input` with integral powers of two created from the exponents
in :attr:'other'.

Args:
    input (Tensor): the input tensor.
    other (Tensor): a tensor of exponents, typically integers.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.ldexp(torch.tensor([1.]), torch.tensor([1]))
    tensor([2.])
    >>> torch.ldexp(torch.tensor([1.0]), torch.tensor([1, 2, 3, 4]))
    tensor([ 2.,  4.,  8., 16.])


"""

    title = 'LdexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ldexp())



"""
WARNING: Module Ldexp_Node was generated using fallback option. May contain bugs
"""

class Ldexp_Node(Node):
    """None"""

    title = 'Ldexp_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ldexp_())



"""
WARNING: Module LeNode was generated using fallback option. May contain bugs
"""

class LeNode(Node):
    """
le(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \leq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or Scalar): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is less than or equal to
    :attr:`other` and False elsewhere

Example::

    >>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, False], [True, True]])
"""

    title = 'LeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.le())



"""
WARNING: Module Legacy_contiguous_formatNode was generated using fallback option. May contain bugs
"""

class Legacy_contiguous_formatNode(Node):
    """None"""

    title = 'Legacy_contiguous_formatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.legacy_contiguous_format())



"""
WARNING: Module LerpNode was generated using fallback option. May contain bugs
"""

class LerpNode(Node):
    """
lerp(input, end, weight, *, out=None)

Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based
on a scalar or tensor :attr:`weight` and returns the resulting :attr:`out` tensor.

.. math::
    \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

The shapes of :attr:`start` and :attr:`end` must be
:ref:`broadcastable <broadcasting-semantics>`. If :attr:`weight` is a tensor, then
the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the tensor with the starting points
    end (Tensor): the tensor with the ending points
    weight (float or tensor): the weight for the interpolation formula

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> start = torch.arange(1., 5.)
    >>> end = torch.empty(4).fill_(10)
    >>> start
    tensor([ 1.,  2.,  3.,  4.])
    >>> end
    tensor([ 10.,  10.,  10.,  10.])
    >>> torch.lerp(start, end, 0.5)
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
    >>> torch.lerp(start, end, torch.full_like(start, 0.5))
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
"""

    title = 'LerpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lerp())



"""
WARNING: Module LessNode was generated using fallback option. May contain bugs
"""

class LessNode(Node):
    """
less(input, other, *, out=None) -> Tensor

Alias for :func:`torch.lt`.
"""

    title = 'LessNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.less())



"""
WARNING: Module Less_equalNode was generated using fallback option. May contain bugs
"""

class Less_equalNode(Node):
    """
less_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.le`.
"""

    title = 'Less_equalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.less_equal())



"""
WARNING: Module LgammaNode was generated using fallback option. May contain bugs
"""

class LgammaNode(Node):
    """
lgamma(input, *, out=None) -> Tensor

Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.lgamma(a)
    tensor([ 0.5724,  0.0000, -0.1208])
"""

    title = 'LgammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lgamma())



"""
WARNING: Module LinalgNode was generated using fallback option. May contain bugs
"""

class LinalgNode(Node):
    """None"""

    title = 'LinalgNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.linalg())



"""
WARNING: Module LinspaceNode was generated using fallback option. May contain bugs
"""

class LinspaceNode(Node):
    """
linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
spaced from :attr:`start` to :attr:`end`, inclusive. That is, the value are:

.. math::
    (\text{start},
    \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \ldots,
    \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \text{end})


.. warning::
    Not providing a value for :attr:`steps` is deprecated. For backwards
    compatibility, not providing a value for :attr:`steps` will create a tensor
    with 100 elements. Note that this behavior is not reflected in the
    documented function signature and should not be relied on. In a future
    PyTorch release, failing to provide a value for :attr:`steps` will throw a
    runtime error.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): size of the constructed tensor

Keyword arguments:
    out (Tensor, optional): the output tensor.
    dtype (torch.dtype, optional): the data type to perform the computation in.
        Default: if None, uses the global default dtype (see torch.get_default_dtype())
        when both :attr:`start` and :attr:`end` are real,
        and corresponding complex dtype when either is complex.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.


Example::

    >>> torch.linspace(3, 10, steps=5)
    tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
    >>> torch.linspace(-10, 10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=1)
    tensor([-10.])
"""

    title = 'LinspaceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.linspace())


class LoadNode(Node):
    """load(f, map_location=None, pickle_module=pickle, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

    :func:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``'cpu'``
    for CPU tensors and ``'cuda:device_id'`` (e.g. ``'cuda:2'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn't specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.

    .. warning::
        :func:`torch.load()` uses ``pickle`` module implicitly, which is known to be insecure.
        It is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never load data that could have come from an untrusted
        source, or that could have been tampered with. **Only load data you trust**.

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding='latin1'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding='bytes'` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.

    Example:
        >>> torch.load('tensors.pt')
        # Load all tensors onto the CPU
        >>> torch.load('tensors.pt', map_location=torch.device('cpu'))
        # Load all tensors onto the CPU, using a function
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
        # Load all tensors onto GPU 1
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
        # Load tensor from io.BytesIO object
        >>> with open('tensor.pt', 'rb') as f:
        ...     buffer = io.BytesIO(f.read())
        >>> torch.load(buffer)
        # Load a module with 'ascii' encoding for unpickling
        >>> torch.load('module.pt', encoding='ascii')
    """

    title = 'LoadNode'
    init_inputs = [
        NodeInputBP('f'),
NodeInputBP('map_location'),
NodeInputBP('pickle_module'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.load(self.input(0), self.input(1), self.input(2)))


class LobpcgNode(Node):
    """Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive defined generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.

    Supported inputs are dense, sparse, and batches of dense matrices.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    .. warning:: The backward method does not support sparse and complex inputs.
      It works only when `B` is not provided (i.e. `B == None`).
      We are actively working on extensions, and the details of
      the algorithms are going to be published promptly.

    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.
      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric
      in first-order optimization routines, prior to running `lobpcg`
      we do the following symmetrization map: `A -> (A + A.t()) / 2`.
      The map is performed only when the `A` requires gradients.

    Args:

      A (Tensor): the input tensor of size :math:`(*, m, m)`

      B (Tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When not specified, `B` is interpereted as
                  identity matrix.

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, it is used as
                  initial approximation of eigenvectors. X must be a
                  dense tensor.

      iK (tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When specified, it will be used as preconditioner.

      k (integer, optional): the number of requested
                  eigenpairs. Default is the number of :math:`X`
                  columns (when specified) or `1`.

      n (integer, optional): if :math:`X` is not specified then `n`
                  specifies the size of the generated random
                  approximation of eigenvectors. Default value for `n`
                  is `k`. If :math:`X` is specified, the value of `n`
                  (when specified) must be the number of :math:`X`
                  columns.

      tol (float, optional): residual tolerance for stopping
                 criterion. Default is `feps ** 0.5` where `feps` is
                 smallest non-zero floating-point number of the given
                 input tensor `A` data type.

      largest (bool, optional): when True, solve the eigenproblem for
                 the largest eigenvalues. Otherwise, solve the
                 eigenproblem for smallest eigenvalues. Default is
                 `True`.

      method (str, optional): select LOBPCG method. See the
                 description of the function above. Default is
                 "ortho".

      niter (int, optional): maximum number of iterations. When
                 reached, the iteration process is hard-stopped and
                 the current approximation of eigenpairs is returned.
                 For infinite iteration but until convergence criteria
                 is met, use `-1`.

      tracker (callable, optional) : a function for tracing the
                 iteration process. When specified, it is called at
                 each iteration step with LOBPCG instance as an
                 argument. The LOBPCG instance holds the full state of
                 the iteration process in the following attributes:

                   `iparams`, `fparams`, `bparams` - dictionaries of
                   integer, float, and boolean valued input
                   parameters, respectively

                   `ivars`, `fvars`, `bvars`, `tvars` - dictionaries
                   of integer, float, boolean, and Tensor valued
                   iteration variables, respectively.

                   `A`, `B`, `iK` - input Tensor arguments.

                   `E`, `X`, `S`, `R` - iteration Tensor variables.

                 For instance:

                   `ivars["istep"]` - the current iteration step
                   `X` - the current approximation of eigenvectors
                   `E` - the current approximation of eigenvalues
                   `R` - the current residual
                   `ivars["converged_count"]` - the current number of converged eigenpairs
                   `tvars["rerr"]` - the current state of convergence criteria

                 Note that when `tracker` stores Tensor objects from
                 the LOBPCG instance, it must make copies of these.

                 If `tracker` sets `bvars["force_stop"] = True`, the
                 iteration process will be hard-stopped.

      ortho_iparams, ortho_fparams, ortho_bparams (dict, optional):
                 various parameters to LOBPCG algorithm when using
                 `method="ortho"`.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    References:

      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
      Preconditioned Eigensolver: Locally Optimal Block Preconditioned
      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
      517-541. (25 pages)
      https://epubs.siam.org/doi/abs/10.1137/S1064827500366124

      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
      Wu. (2002) A Block Orthogonalization Procedure with Constant
      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
      2165-2182. (18 pages)
      https://epubs.siam.org/doi/10.1137/S1064827500370883

      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
      https://epubs.siam.org/doi/abs/10.1137/17M1129830

    """

    title = 'LobpcgNode'
    init_inputs = [
        NodeInputBP('A'),
NodeInputBP('k'),
NodeInputBP('B'),
NodeInputBP('X'),
NodeInputBP('n'),
NodeInputBP('iK'),
NodeInputBP('niter'),
NodeInputBP('tol'),
NodeInputBP('largest'),
NodeInputBP('method'),
NodeInputBP('tracker'),
NodeInputBP('ortho_iparams'),
NodeInputBP('ortho_fparams'),
NodeInputBP('ortho_bparams'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lobpcg(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4), self.input(5), self.input(6), self.input(7), self.input(8), self.input(9), self.input(10), self.input(11), self.input(12), self.input(13)))



"""
WARNING: Module LogNode was generated using fallback option. May contain bugs
"""

class LogNode(Node):
    """
log(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
    >>> torch.log(a)
    tensor([ nan,  nan,  nan,  nan,  nan])
"""

    title = 'LogNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log())



"""
WARNING: Module Log10Node was generated using fallback option. May contain bugs
"""

class Log10Node(Node):
    """
log10(input, *, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])


    >>> torch.log10(a)
    tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

"""

    title = 'Log10Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log10())



"""
WARNING: Module Log10_Node was generated using fallback option. May contain bugs
"""

class Log10_Node(Node):
    """None"""

    title = 'Log10_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log10_())



"""
WARNING: Module Log1pNode was generated using fallback option. May contain bugs
"""

class Log1pNode(Node):
    """
log1p(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

.. math::
    y_i = \log_{e} (x_i + 1)

.. note:: This function is more accurate than :func:`torch.log` for small
          values of :attr:`input`

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
    >>> torch.log1p(a)
    tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
"""

    title = 'Log1pNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log1p())



"""
WARNING: Module Log1p_Node was generated using fallback option. May contain bugs
"""

class Log1p_Node(Node):
    """None"""

    title = 'Log1p_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log1p_())



"""
WARNING: Module Log2Node was generated using fallback option. May contain bugs
"""

class Log2Node(Node):
    """
log2(input, *, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


    >>> torch.log2(a)
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

"""

    title = 'Log2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log2())



"""
WARNING: Module Log2_Node was generated using fallback option. May contain bugs
"""

class Log2_Node(Node):
    """None"""

    title = 'Log2_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log2_())



"""
WARNING: Module Log_Node was generated using fallback option. May contain bugs
"""

class Log_Node(Node):
    """None"""

    title = 'Log_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log_())



"""
WARNING: Module Log_softmaxNode was generated using fallback option. May contain bugs
"""

class Log_softmaxNode(Node):
    """None"""

    title = 'Log_softmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.log_softmax())



"""
WARNING: Module LogaddexpNode was generated using fallback option. May contain bugs
"""

class LogaddexpNode(Node):
    """
logaddexp(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs.

Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful
in statistics where the calculated probabilities of events may be so small as to
exceed the range of normal floating point numbers. In such cases the logarithm
of the calculated probability is stored. This function allows adding
probabilities stored in such a fashion.

This op should be disambiguated with :func:`torch.logsumexp` which performs a
reduction on a single tensor.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
    tensor([-0.3069, -0.6867, -0.8731])
    >>> torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
    tensor([-1., -2., -3.])
    >>> torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))
    tensor([1.1269e+00, 2.0000e+03, 3.0000e+04])
"""

    title = 'LogaddexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logaddexp())



"""
WARNING: Module Logaddexp2Node was generated using fallback option. May contain bugs
"""

class Logaddexp2Node(Node):
    """
logaddexp2(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs in base-2.

Calculates pointwise :math:`\log_2\left(2^x + 2^y\right)`. See
:func:`torch.logaddexp` for more details.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword arguments:
    out (Tensor, optional): the output tensor.
"""

    title = 'Logaddexp2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logaddexp2())



"""
WARNING: Module LogcumsumexpNode was generated using fallback option. May contain bugs
"""

class LogcumsumexpNode(Node):
    """
logcumsumexp(input, dim, *, out=None) -> Tensor
Returns the logarithm of the cumulative summation of the exponentiation of
elements of :attr:`input` in the dimension :attr:`dim`.

For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{logcumsumexp}(x)_{ij} = \log \sum\limits_{j=0}^{i} \exp(x_{ij})

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(10)
    >>> torch.logcumsumexp(a, dim=0)
    tensor([-0.42296738, -0.04462666,  0.86278635,  0.94622083,  1.05277811,
             1.39202815,  1.83525007,  1.84492621,  2.06084887,  2.06844475]))
"""

    title = 'LogcumsumexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logcumsumexp())



"""
WARNING: Module LogdetNode was generated using fallback option. May contain bugs
"""

class LogdetNode(Node):
    """
logdet(input) -> Tensor

Calculates log determinant of a square matrix or batches of square matrices.

.. note::
    Result is ``-inf`` if :attr:`input` has zero log determinant, and is ``nan`` if
    :attr:`input` has negative determinant.

.. note::
    Backward through :meth:`logdet` internally uses SVD results when :attr:`input`
    is not invertible. In this case, double backward through :meth:`logdet` will
    be unstable in when :attr:`input` doesn't have distinct singular values. See
    :meth:`~torch.svd` for details.

Arguments:
    input (Tensor): the input tensor of size ``(*, n, n)`` where ``*`` is zero or more
                batch dimensions.

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(0.2611)
    >>> torch.logdet(A)
    tensor(-1.3430)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
    >>> A.det().log()
    tensor([ 0.1815, -0.8917, -0.3031])
"""

    title = 'LogdetNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logdet())



"""
WARNING: Module Logical_andNode was generated using fallback option. May contain bugs
"""

class Logical_andNode(Node):
    """
logical_and(input, other, *, out=None) -> Tensor

Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the tensor to compute AND with

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([ True, False, False])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    >>> torch.logical_and(a, b)
    tensor([False, False,  True, False])
    >>> torch.logical_and(a.double(), b.double())
    tensor([False, False,  True, False])
    >>> torch.logical_and(a.double(), b)
    tensor([False, False,  True, False])
    >>> torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool))
    tensor([False, False,  True, False])
"""

    title = 'Logical_andNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logical_and())



"""
WARNING: Module Logical_notNode was generated using fallback option. May contain bugs
"""

class Logical_notNode(Node):
    """
logical_not(input, *, out=None) -> Tensor

Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.logical_not(torch.tensor([True, False]))
    tensor([False,  True])
    >>> torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16))
    tensor([1, 0, 0], dtype=torch.int16)
"""

    title = 'Logical_notNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logical_not())



"""
WARNING: Module Logical_orNode was generated using fallback option. May contain bugs
"""

class Logical_orNode(Node):
    """
logical_or(input, other, *, out=None) -> Tensor

Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the tensor to compute OR with

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([ True, False,  True])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    >>> torch.logical_or(a, b)
    tensor([ True,  True,  True, False])
    >>> torch.logical_or(a.double(), b.double())
    tensor([ True,  True,  True, False])
    >>> torch.logical_or(a.double(), b)
    tensor([ True,  True,  True, False])
    >>> torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool))
    tensor([ True,  True,  True, False])
"""

    title = 'Logical_orNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logical_or())



"""
WARNING: Module Logical_xorNode was generated using fallback option. May contain bugs
"""

class Logical_xorNode(Node):
    """
logical_xor(input, other, *, out=None) -> Tensor

Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the tensor to compute XOR with

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([False, False,  True])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    >>> torch.logical_xor(a, b)
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a.double(), b.double())
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a.double(), b)
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool))
    tensor([ True,  True, False, False])
"""

    title = 'Logical_xorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logical_xor())



"""
WARNING: Module LogitNode was generated using fallback option. May contain bugs
"""

class LogitNode(Node):
    """
logit(input, eps=None, *, out=None) -> Tensor

Alias for :func:`torch.special.logit`.
"""

    title = 'LogitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logit())



"""
WARNING: Module Logit_Node was generated using fallback option. May contain bugs
"""

class Logit_Node(Node):
    """None"""

    title = 'Logit_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logit_())



"""
WARNING: Module LogspaceNode was generated using fallback option. May contain bugs
"""

class LogspaceNode(Node):
    """
logspace(start, end, steps, base=10.0, *,          out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
spaced from :math:`{{\text{{base}}}}^{{\text{{start}}}}` to
:math:`{{\text{{base}}}}^{{\text{{end}}}}`, inclusive, on a logarithmic scale
with base :attr:`base`. That is, the values are:

.. math::
    (\text{base}^{\text{start}},
    \text{base}^{(\text{start} + \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \ldots,
    \text{base}^{(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \text{base}^{\text{end}})


.. warning::
    Not providing a value for :attr:`steps` is deprecated. For backwards
    compatibility, not providing a value for :attr:`steps` will create a tensor
    with 100 elements. Note that this behavior is not reflected in the
    documented function signature and should not be relied on. In a future
    PyTorch release, failing to provide a value for :attr:`steps` will throw a
    runtime error.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): size of the constructed tensor
    base (float, optional): base of the logarithm function. Default: ``10.0``.

Keyword arguments:
    out (Tensor, optional): the output tensor.
    dtype (torch.dtype, optional): the data type to perform the computation in.
        Default: if None, uses the global default dtype (see torch.get_default_dtype())
        when both :attr:`start` and :attr:`end` are real,
        and corresponding complex dtype when either is complex.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.logspace(start=-10, end=10, steps=5)
    tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
    >>> torch.logspace(start=0.1, end=1.0, steps=5)
    tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
    >>> torch.logspace(start=0.1, end=1.0, steps=1)
    tensor([1.2589])
    >>> torch.logspace(start=2, end=2, steps=1, base=2)
    tensor([4.0])
"""

    title = 'LogspaceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logspace())



"""
WARNING: Module LogsumexpNode was generated using fallback option. May contain bugs
"""

class LogsumexpNode(Node):
    """
logsumexp(input, dim, keepdim=False, *, out=None)

Returns the log of summed exponentials of each row of the :attr:`input`
tensor in the given dimension :attr:`dim`. The computation is numerically
stabilized.

For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{logsumexp}(x)_{i} = \log \sum_j \exp(x_{ij})


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.


Example::

    >>> a = torch.randn(3, 3)
    >>> torch.logsumexp(a, 1)
    tensor([ 0.8442,  1.4322,  0.8711])
"""

    title = 'LogsumexpNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.logsumexp())



"""
WARNING: Module LongNode was generated using fallback option. May contain bugs
"""

class LongNode(Node):
    """None"""

    title = 'LongNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.long())



"""
WARNING: Module LstmNode was generated using fallback option. May contain bugs
"""

class LstmNode(Node):
    """None"""

    title = 'LstmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lstm())



"""
WARNING: Module Lstm_cellNode was generated using fallback option. May contain bugs
"""

class Lstm_cellNode(Node):
    """None"""

    title = 'Lstm_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lstm_cell())



"""
WARNING: Module LstsqNode was generated using fallback option. May contain bugs
"""

class LstsqNode(Node):
    """
lstsq(input, A, *, out=None) -> (Tensor, Tensor)

Computes the solution to the least squares and least norm problems for a full
rank matrix :math:`A` of size :math:`(m \times n)` and a matrix :math:`B` of
size :math:`(m \times k)`.

If :math:`m \geq n`, :func:`lstsq` solves the least-squares problem:

.. math::

   \begin{array}{ll}
   \min_X & \|AX-B\|_2.
   \end{array}

If :math:`m < n`, :func:`lstsq` solves the least-norm problem:

.. math::

   \begin{array}{llll}
   \min_X & \|X\|_2 & \text{subject to} & AX = B.
   \end{array}

Returned tensor :math:`X` has shape :math:`(\max(m, n) \times k)`. The first :math:`n`
rows of :math:`X` contains the solution. If :math:`m \geq n`, the residual sum of squares
for the solution in each column is given by the sum of squares of elements in the
remaining :math:`m - n` rows of that column.

.. warning::

    :func:`torch.lstsq` is deprecated in favor of :func:`torch.linalg.lstsq`
    and will be removed in a future PyTorch release. :func:`torch.linalg.lstsq`
    has reversed arguments and does not return the QR decomposition in the returned tuple,
    (it returns other information about the problem).
    The returned `solution` in :func:`torch.lstsq` stores the residuals of the solution in the
    last `m - n` columns in the case `m > n`. In :func:`torch.linalg.lstsq`, the residuals
    are in the field 'residuals' of the returned named tuple.

    Unpacking the solution as``X = torch.lstsq(B, A).solution[:A.size(1)]`` should be replaced with

    .. code:: python

        X = torch.linalg.lstsq(A, B).solution

.. note::
    The case when :math:`m < n` is not supported on the GPU.

Args:
    input (Tensor): the matrix :math:`B`
    A (Tensor): the :math:`m` by :math:`n` matrix :math:`A`

Keyword args:
    out (tuple, optional): the optional destination tensor

Returns:
    (Tensor, Tensor): A namedtuple (solution, QR) containing:

        - **solution** (*Tensor*): the least squares solution
        - **QR** (*Tensor*): the details of the QR factorization

.. note::

    The returned matrices will always be transposed, irrespective of the strides
    of the input matrices. That is, they will have stride `(1, m)` instead of
    `(m, 1)`.

Example::

    >>> A = torch.tensor([[1., 1, 1],
    ...                   [2, 3, 4],
    ...                   [3, 5, 2],
    ...                   [4, 2, 5],
    ...                   [5, 4, 3]])
    >>> B = torch.tensor([[-10., -3],
    ...                   [ 12, 14],
    ...                   [ 14, 12],
    ...                   [ 16, 16],
    ...                   [ 18, 16]])
    >>> X, _ = torch.lstsq(B, A)
    >>> X
    tensor([[  2.0000,   1.0000],
            [  1.0000,   1.0000],
            [  1.0000,   2.0000],
            [ 10.9635,   4.8501],
            [  8.9332,   5.2418]])
"""

    title = 'LstsqNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lstsq())



"""
WARNING: Module LtNode was generated using fallback option. May contain bugs
"""

class LtNode(Node):
    """
lt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} < \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is less than :attr:`other` and False elsewhere

Example::

    >>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, False], [True, False]])
"""

    title = 'LtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lt())


class LuNode(Node):
    """Computes the LU factorization of a matrix or batches of matrices
    :attr:`A`. Returns a tuple containing the LU factorization and
    pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
    ``True``.

    .. note::
        The pivots returned by the function are 1-indexed. If :attr:`pivot` is ``False``,
        then the returned pivots is a tensor filled with zeros of the appropriate size.

    .. note::
        LU factorization with :attr:`pivot` = ``False`` is not available for CPU, and attempting
        to do so will throw an error. However, LU factorization with :attr:`pivot` = ``False`` is
        available for CUDA.

    .. note::
        This function does not check if the factorization was successful or not if
        :attr:`get_infos` is ``True`` since the status of the factorization is present in the
        third element of the return tuple.

    .. note::
        In the case of batches of square matrices with size less or
        equal to 32 on a CUDA device, the LU factorization is repeated
        for singular matrices due to the bug in the MAGMA library (see
        magma issue 13).

    .. note::
       ``L``, ``U``, and ``P`` can be derived using :func:`torch.lu_unpack`.

    .. warning::
        The LU factorization does have backward support,
        but only for square inputs of full rank.

    Args:
        A (Tensor): the tensor to factor of size :math:`(*, m, n)`
        pivot (bool, optional): controls whether pivoting is done. Default: ``True``
        get_infos (bool, optional): if set to ``True``, returns an info IntTensor.
                                    Default: ``False``
        out (tuple, optional): optional output tuple. If :attr:`get_infos` is ``True``,
                               then the elements in the tuple are Tensor, IntTensor,
                               and IntTensor. If :attr:`get_infos` is ``False``, then the
                               elements in the tuple are Tensor, IntTensor. Default: ``None``

    Returns:
        (Tensor, IntTensor, IntTensor (optional)): A tuple of tensors containing

            - **factorization** (*Tensor*): the factorization of size :math:`(*, m, n)`

            - **pivots** (*IntTensor*): the pivots of size :math:`(*, \text{min}(m, n))`.
              ``pivots`` stores all the intermediate transpositions of rows.
              The final permutation ``perm`` could be reconstructed by
              applying ``swap(perm[i], perm[pivots[i] - 1])`` for ``i = 0, ..., pivots.size(-1) - 1``,
              where ``perm`` is initially the identity permutation of :math:`m` elements
              (essentially this is what :func:`torch.lu_unpack` is doing).

            - **infos** (*IntTensor*, *optional*): if :attr:`get_infos` is ``True``, this is a tensor of
              size :math:`(*)` where non-zero values indicate whether factorization for the matrix or
              each minibatch has succeeded or failed

    Example::

        >>> A = torch.randn(2, 3, 3)
        >>> A_LU, pivots = torch.lu(A)
        >>> A_LU
        tensor([[[ 1.3506,  2.5558, -0.0816],
                 [ 0.1684,  1.1551,  0.1940],
                 [ 0.1193,  0.6189, -0.5497]],

                [[ 0.4526,  1.2526, -0.3285],
                 [-0.7988,  0.7175, -0.9701],
                 [ 0.2634, -0.9255, -0.3459]]])
        >>> pivots
        tensor([[ 3,  3,  3],
                [ 3,  3,  3]], dtype=torch.int32)
        >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
        >>> if info.nonzero().size(0) == 0:
        ...   print('LU factorization succeeded for all samples!')
        LU factorization succeeded for all samples!
    """

    title = 'LuNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lu())



"""
WARNING: Module Lu_solveNode was generated using fallback option. May contain bugs
"""

class Lu_solveNode(Node):
    """
lu_solve(b, LU_data, LU_pivots, *, out=None) -> Tensor

Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
LU factorization of A from :meth:`torch.lu`.

This function supports ``float``, ``double``, ``cfloat`` and ``cdouble`` dtypes for :attr:`input`.

Arguments:
    b (Tensor): the RHS tensor of size :math:`(*, m, k)`, where :math:`*`
                is zero or more batch dimensions.
    LU_data (Tensor): the pivoted LU factorization of A from :meth:`torch.lu` of size :math:`(*, m, m)`,
                       where :math:`*` is zero or more batch dimensions.
    LU_pivots (IntTensor): the pivots of the LU factorization from :meth:`torch.lu` of size :math:`(*, m)`,
                           where :math:`*` is zero or more batch dimensions.
                           The batch dimensions of :attr:`LU_pivots` must be equal to the batch dimensions of
                           :attr:`LU_data`.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(2, 3, 1)
    >>> A_LU = torch.lu(A)
    >>> x = torch.lu_solve(b, *A_LU)
    >>> torch.norm(torch.bmm(A, x) - b)
    tensor(1.00000e-07 *
           2.8312)
"""

    title = 'Lu_solveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lu_solve())



"""
WARNING: Module Lu_unpackNode was generated using fallback option. May contain bugs
"""

class Lu_unpackNode(Node):
    """
lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None) -> (Tensor, Tensor, Tensor)

Unpacks the data and pivots from a LU factorization of a tensor into tensors ``L`` and ``U`` and a permutation tensor ``P``
such that ``LU_data, LU_pivots = (P @ L @ U).lu()``.

Returns a tuple of tensors as ``(the P tensor (permutation matrix), the L tensor, the U tensor)``.

.. note:: ``P.dtype == LU_data.dtype`` and ``P.dtype`` is not an integer type so that matrix products with ``P``
          are possible without casting it to a floating type.

Args:
    LU_data (Tensor): the packed LU factorization data
    LU_pivots (Tensor): the packed LU factorization pivots
    unpack_data (bool): flag indicating if the data should be unpacked.
                        If ``False``, then the returned ``L`` and ``U`` are ``None``.
                        Default: ``True``
    unpack_pivots (bool): flag indicating if the pivots should be unpacked into a permutation matrix ``P``.
                          If ``False``, then the returned ``P`` is  ``None``.
                          Default: ``True``
    out (tuple, optional): a tuple of three tensors to use for the outputs ``(P, L, U)``.

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> A_LU, pivots = A.lu()
    >>> P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
    >>>
    >>> # can recover A from factorization
    >>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))

    >>> # LU factorization of a rectangular matrix:
    >>> A = torch.randn(2, 3, 2)
    >>> A_LU, pivots = A.lu()
    >>> P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
    >>> P
    tensor([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]],

            [[0., 0., 1.],
             [0., 1., 0.],
             [1., 0., 0.]]])
    >>> A_L
    tensor([[[ 1.0000,  0.0000],
             [ 0.4763,  1.0000],
             [ 0.3683,  0.1135]],

            [[ 1.0000,  0.0000],
             [ 0.2957,  1.0000],
             [-0.9668, -0.3335]]])
    >>> A_U
    tensor([[[ 2.1962,  1.0881],
             [ 0.0000, -0.8681]],

            [[-1.0947,  0.3736],
             [ 0.0000,  0.5718]]])
    >>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))
    >>> torch.norm(A_ - A)
    tensor(2.9802e-08)
"""

    title = 'Lu_unpackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.lu_unpack())


class Manual_seedNode(Node):
    """Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """

    title = 'Manual_seedNode'
    init_inputs = [
        NodeInputBP('seed'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.manual_seed(self.input(0)))



"""
WARNING: Module Margin_ranking_lossNode was generated using fallback option. May contain bugs
"""

class Margin_ranking_lossNode(Node):
    """None"""

    title = 'Margin_ranking_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.margin_ranking_loss())



"""
WARNING: Module Masked_fillNode was generated using fallback option. May contain bugs
"""

class Masked_fillNode(Node):
    """None"""

    title = 'Masked_fillNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.masked_fill())



"""
WARNING: Module Masked_scatterNode was generated using fallback option. May contain bugs
"""

class Masked_scatterNode(Node):
    """None"""

    title = 'Masked_scatterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.masked_scatter())



"""
WARNING: Module Masked_selectNode was generated using fallback option. May contain bugs
"""

class Masked_selectNode(Node):
    """
masked_select(input, mask, *, out=None) -> Tensor

Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
the boolean mask :attr:`mask` which is a `BoolTensor`.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. note:: The returned tensor does **not** use the same storage
          as the original tensor

Args:
    input (Tensor): the input tensor.
    mask  (BoolTensor): the tensor containing the binary mask to index with

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
            [-1.2035,  1.2252,  0.5002,  0.6248],
            [ 0.1307, -2.0608,  0.1244,  2.0139]])
    >>> mask = x.ge(0.5)
    >>> mask
    tensor([[False, False, False, False],
            [False, True, True, True],
            [False, False, False, True]])
    >>> torch.masked_select(x, mask)
    tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
"""

    title = 'Masked_selectNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.masked_select())



"""
WARNING: Module MatmulNode was generated using fallback option. May contain bugs
"""

class MatmulNode(Node):
    """
matmul(input, other, *, out=None) -> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
  must be broadcastable).  For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times n)` tensor and :attr:`other` is a :math:`(k \times n \times n)`
  tensor, :attr:`out` will be a :math:`(j \times k \times n \times n)` tensor.

  Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
  are broadcastable, and not the matrix dimensions. For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
  tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
  matrix dimensions) are different. :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    input (Tensor): the first tensor to be multiplied
    other (Tensor): the second tensor to be multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> # vector x vector
    >>> tensor1 = torch.randn(3)
    >>> tensor2 = torch.randn(3)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([])
    >>> # matrix x vector
    >>> tensor1 = torch.randn(3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(10, 4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])

"""

    title = 'MatmulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.matmul())



"""
WARNING: Module Matrix_expNode was generated using fallback option. May contain bugs
"""

class Matrix_expNode(Node):
    """
matrix_exp(input) -> Tensor

Computes the matrix exponential of a square matrix or of each square matrix in a batch.
For a matrix :attr:`input`, the matrix exponential is defined as

.. math::
    \mathrm{e}^\text{input} = \sum_{k=0}^\infty \text{input}^k / k!


The implementation is based on:

Bader, P.; Blanes, S.; Casas, F.
Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
Mathematics 2019, 7, 1174.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(2, 2, 2)
    >>> a[0, :, :] = torch.eye(2, 2)
    >>> a[1, :, :] = 2 * torch.eye(2, 2)
    >>> a
    tensor([[[1., 0.],
             [0., 1.]],

            [[2., 0.],
             [0., 2.]]])
    >>> torch.matrix_exp(a)
    tensor([[[2.7183, 0.0000],
             [0.0000, 2.7183]],

             [[7.3891, 0.0000],
              [0.0000, 7.3891]]])

    >>> import math
    >>> x = torch.tensor([[0, math.pi/3], [-math.pi/3, 0]])
    >>> x.matrix_exp() # should be [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
    tensor([[ 0.5000,  0.8660],
            [-0.8660,  0.5000]])
"""

    title = 'Matrix_expNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.matrix_exp())



"""
WARNING: Module Matrix_powerNode was generated using fallback option. May contain bugs
"""

class Matrix_powerNode(Node):
    """
matrix_power(input, n, *, out=None) -> Tensor

Alias for :func:`torch.linalg.matrix_power`
"""

    title = 'Matrix_powerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.matrix_power())



"""
WARNING: Module Matrix_rankNode was generated using fallback option. May contain bugs
"""

class Matrix_rankNode(Node):
    """
matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor

Returns the numerical rank of a 2-D tensor. The method to compute the
matrix rank is done using SVD by default. If :attr:`symmetric` is ``True``,
then :attr:`input` is assumed to be symmetric, and the computation of the
rank is done by obtaining the eigenvalues.

:attr:`tol` is the threshold below which the singular values (or the eigenvalues
when :attr:`symmetric` is ``True``) are considered to be 0. If :attr:`tol` is not
specified, :attr:`tol` is set to ``S.max() * max(S.size()) * eps`` where `S` is the
singular values (or the eigenvalues when :attr:`symmetric` is ``True``), and ``eps``
is the epsilon value for the datatype of :attr:`input`.

.. warning::

    :func:`torch.matrix_rank` is deprecated in favor of :func:`torch.linalg.matrix_rank`
    and will be removed in a future PyTorch release. The parameter :attr:`symmetric` was
    renamed in :func:`torch.linalg.matrix_rank` to :attr:`hermitian`.

Args:
    input (Tensor): the input 2-D tensor
    tol (float, optional): the tolerance value. Default: ``None``
    symmetric(bool, optional): indicates whether :attr:`input` is symmetric.
                               Default: ``False``

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.eye(10)
    >>> torch.matrix_rank(a)
    tensor(10)
    >>> b = torch.eye(10)
    >>> b[0, 0] = 0
    >>> torch.matrix_rank(b)
    tensor(9)
"""

    title = 'Matrix_rankNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.matrix_rank())



"""
WARNING: Module MaxNode was generated using fallback option. May contain bugs
"""

class MaxNode(Node):
    """
max(input) -> Tensor

Returns the maximum value of all elements in the ``input`` tensor.

.. warning::
    This function produces deterministic (sub)gradients unlike ``max(dim=0)``

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)

.. function:: max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If ``keepdim`` is ``True``, the output tensors are of the same size
as ``input`` except in the dimension ``dim`` where they are of size 1.
Otherwise, ``dim`` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than ``input``.

.. note:: If there are multiple maximal values in a reduced row then
          the indices of the first maximal value are returned.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

.. function:: max(input, other, *, out=None) -> Tensor

See :func:`torch.maximum`.

"""

    title = 'MaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.max())



"""
WARNING: Module Max_pool1dNode was generated using fallback option. May contain bugs
"""

class Max_pool1dNode(Node):
    """None"""

    title = 'Max_pool1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.max_pool1d())



"""
WARNING: Module Max_pool1d_with_indicesNode was generated using fallback option. May contain bugs
"""

class Max_pool1d_with_indicesNode(Node):
    """None"""

    title = 'Max_pool1d_with_indicesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.max_pool1d_with_indices())



"""
WARNING: Module Max_pool2dNode was generated using fallback option. May contain bugs
"""

class Max_pool2dNode(Node):
    """None"""

    title = 'Max_pool2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.max_pool2d())



"""
WARNING: Module Max_pool3dNode was generated using fallback option. May contain bugs
"""

class Max_pool3dNode(Node):
    """None"""

    title = 'Max_pool3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.max_pool3d())



"""
WARNING: Module MaximumNode was generated using fallback option. May contain bugs
"""

class MaximumNode(Node):
    """
maximum(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`maximum` is not supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.maximum(a, b)
    tensor([3, 2, 4])
"""

    title = 'MaximumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.maximum())



"""
WARNING: Module MeanNode was generated using fallback option. May contain bugs
"""

class MeanNode(Node):
    """
mean(input) -> Tensor

Returns the mean value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)

.. function:: mean(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
"""

    title = 'MeanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mean())



"""
WARNING: Module MedianNode was generated using fallback option. May contain bugs
"""

class MedianNode(Node):
    """
median(input) -> Tensor

Returns the median of the values in :attr:`input`.

.. note::
    The median is not unique for :attr:`input` tensors with an even number
    of elements. In this case the lower of the two medians is returned. To
    compute the mean of both medians, use :func:`torch.quantile` with ``q=0.5`` instead.

.. warning::
    This function produces deterministic (sub)gradients unlike ``median(dim=0)``

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.5219, -1.5212,  0.2202]])
    >>> torch.median(a)
    tensor(0.2202)

.. function:: median(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` contains the median of each row of :attr:`input`
in the dimension :attr:`dim`, and ``indices`` contains the index of the median values found in the dimension :attr:`dim`.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the outputs tensor having 1 fewer dimension than :attr:`input`.

.. note::
    The median is not unique for :attr:`input` tensors with an even number
    of elements in the dimension :attr:`dim`. In this case the lower of the
    two medians is returned. To compute the mean of both medians in
    :attr:`input`, use :func:`torch.quantile` with ``q=0.5`` instead.

.. warning::
    ``indices`` does not necessarily contain the first occurrence of each
    median value found, unless it is unique.
    The exact implementation details are device-specific.
    Do not expect the same result when run on CPU and GPU in general.
    For the same reason do not expect the gradients to be deterministic.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out ((Tensor, Tensor), optional): The first tensor will be populated with the median values and the second
                                      tensor, which must have dtype long, with their indices in the dimension
                                      :attr:`dim` of :attr:`input`.

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
            [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
            [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
            [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
    >>> torch.median(a, 1)
    torch.return_types.median(values=tensor([-0.3982,  0.2270,  0.2488,  0.4742]), indices=tensor([1, 4, 4, 3]))
"""

    title = 'MedianNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.median())


class Memory_formatNode(Node):
    """None"""

    title = 'Memory_formatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.memory_format())



"""
WARNING: Module Merge_type_from_type_commentNode was generated using fallback option. May contain bugs
"""

class Merge_type_from_type_commentNode(Node):
    """merge_type_from_type_comment(arg0: torch._C._jit_tree_views.Decl, arg1: torch._C._jit_tree_views.Decl, arg2: bool) -> torch._C._jit_tree_views.Decl
"""

    title = 'Merge_type_from_type_commentNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
NodeInputBP('c'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.merge_type_from_type_comment(self.input(0), self.input(1), self.input(2)))


class MeshgridNode(Node):
    """Take :math:`N` tensors, each of which can be either scalar or 1-dimensional
        vector, and create :math:`N` N-dimensional grids, where the :math:`i` :sup:`th` grid is defined by
        expanding the :math:`i` :sup:`th` input over dimensions defined by other inputs.

        Args:
            tensors (list of Tensor): list of scalars or 1 dimensional tensors. Scalars will be
                treated as tensors of size :math:`(1,)` automatically

        Returns:
            seq (sequence of Tensors): If the input has :math:`k` tensors of size
            :math:`(N_1,), (N_2,), \ldots , (N_k,)`, then the output would also have :math:`k` tensors,
            where all tensors are of size :math:`(N_1, N_2, \ldots , N_k)`.

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> y = torch.tensor([4, 5, 6])
            >>> grid_x, grid_y = torch.meshgrid(x, y)
            >>> grid_x
            tensor([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
            >>> grid_y
            tensor([[4, 5, 6],
                    [4, 5, 6],
                    [4, 5, 6]])
        """

    title = 'MeshgridNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.meshgrid())



"""
WARNING: Module MinNode was generated using fallback option. May contain bugs
"""

class MinNode(Node):
    """
min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

.. warning::
    This function produces deterministic (sub)gradients unlike ``min(dim=0)``

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)

.. function:: min(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: If there are multiple minimal values in a reduced row then
          the indices of the first minimal value are returned.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> torch.min(a, 1)
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

.. function:: min(input, other, *, out=None) -> Tensor

See :func:`torch.minimum`.
"""

    title = 'MinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.min())



"""
WARNING: Module MinimumNode was generated using fallback option. May contain bugs
"""

class MinimumNode(Node):
    """
minimum(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`minimum` is not supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.minimum(a, b)
    tensor([1, 0, -1])
"""

    title = 'MinimumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.minimum())



"""
WARNING: Module Miopen_batch_normNode was generated using fallback option. May contain bugs
"""

class Miopen_batch_normNode(Node):
    """None"""

    title = 'Miopen_batch_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.miopen_batch_norm())



"""
WARNING: Module Miopen_convolutionNode was generated using fallback option. May contain bugs
"""

class Miopen_convolutionNode(Node):
    """None"""

    title = 'Miopen_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.miopen_convolution())



"""
WARNING: Module Miopen_convolution_transposeNode was generated using fallback option. May contain bugs
"""

class Miopen_convolution_transposeNode(Node):
    """None"""

    title = 'Miopen_convolution_transposeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.miopen_convolution_transpose())



"""
WARNING: Module Miopen_depthwise_convolutionNode was generated using fallback option. May contain bugs
"""

class Miopen_depthwise_convolutionNode(Node):
    """None"""

    title = 'Miopen_depthwise_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.miopen_depthwise_convolution())



"""
WARNING: Module Miopen_rnnNode was generated using fallback option. May contain bugs
"""

class Miopen_rnnNode(Node):
    """None"""

    title = 'Miopen_rnnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.miopen_rnn())



"""
WARNING: Module Mkldnn_adaptive_avg_pool2dNode was generated using fallback option. May contain bugs
"""

class Mkldnn_adaptive_avg_pool2dNode(Node):
    """None"""

    title = 'Mkldnn_adaptive_avg_pool2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_adaptive_avg_pool2d())



"""
WARNING: Module Mkldnn_convolutionNode was generated using fallback option. May contain bugs
"""

class Mkldnn_convolutionNode(Node):
    """None"""

    title = 'Mkldnn_convolutionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_convolution())



"""
WARNING: Module Mkldnn_convolution_backward_weightsNode was generated using fallback option. May contain bugs
"""

class Mkldnn_convolution_backward_weightsNode(Node):
    """None"""

    title = 'Mkldnn_convolution_backward_weightsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_convolution_backward_weights())



"""
WARNING: Module Mkldnn_linear_backward_weightsNode was generated using fallback option. May contain bugs
"""

class Mkldnn_linear_backward_weightsNode(Node):
    """None"""

    title = 'Mkldnn_linear_backward_weightsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_linear_backward_weights())



"""
WARNING: Module Mkldnn_max_pool2dNode was generated using fallback option. May contain bugs
"""

class Mkldnn_max_pool2dNode(Node):
    """None"""

    title = 'Mkldnn_max_pool2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_max_pool2d())



"""
WARNING: Module Mkldnn_max_pool3dNode was generated using fallback option. May contain bugs
"""

class Mkldnn_max_pool3dNode(Node):
    """None"""

    title = 'Mkldnn_max_pool3dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mkldnn_max_pool3d())



"""
WARNING: Module MmNode was generated using fallback option. May contain bugs
"""

class MmNode(Node):
    """
mm(input, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Supports strided and sparse 2-D tensors as inputs, autograd with
respect to strided inputs.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Args:
    input (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])
"""

    title = 'MmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mm())



"""
WARNING: Module ModeNode was generated using fallback option. May contain bugs
"""

class ModeNode(Node):
    """
mode(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`, i.e. a value which appears most often
in that row, and ``indices`` is the index location of each mode value found.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randint(10, (5,))
    >>> a
    tensor([6, 5, 1, 0, 2])
    >>> b = a + (torch.randn(50, 1) * 5).long()
    >>> torch.mode(b, 0)
    torch.return_types.mode(values=tensor([6, 5, 1, 0, 2]), indices=tensor([2, 2, 2, 2, 2]))
"""

    title = 'ModeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mode())



"""
WARNING: Module MoveaxisNode was generated using fallback option. May contain bugs
"""

class MoveaxisNode(Node):
    """
moveaxis(input, source, destination) -> Tensor

Alias for :func:`torch.movedim`.

This function is equivalent to NumPy's moveaxis function.

Examples::

    >>> t = torch.randn(3,2,1)
    >>> t
    tensor([[[-0.3362],
            [-0.8437]],

            [[-0.9627],
            [ 0.1727]],

            [[ 0.5173],
            [-0.1398]]])
    >>> torch.moveaxis(t, 1, 0).shape
    torch.Size([2, 3, 1])
    >>> torch.moveaxis(t, 1, 0)
    tensor([[[-0.3362],
            [-0.9627],
            [ 0.5173]],

            [[-0.8437],
            [ 0.1727],
            [-0.1398]]])
    >>> torch.moveaxis(t, (1, 2), (0, 1)).shape
    torch.Size([2, 1, 3])
    >>> torch.moveaxis(t, (1, 2), (0, 1))
    tensor([[[-0.3362, -0.9627,  0.5173]],

            [[-0.8437,  0.1727, -0.1398]]])
"""

    title = 'MoveaxisNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.moveaxis())



"""
WARNING: Module MovedimNode was generated using fallback option. May contain bugs
"""

class MovedimNode(Node):
    """
movedim(input, source, destination) -> Tensor

Moves the dimension(s) of :attr:`input` at the position(s) in :attr:`source`
to the position(s) in :attr:`destination`.

Other dimensions of :attr:`input` that are not explicitly moved remain in
their original order and appear at the positions not specified in :attr:`destination`.

Args:
    input (Tensor): the input tensor.
    source (int or tuple of ints): Original positions of the dims to move. These must be unique.
    destination (int or tuple of ints): Destination positions for each of the original dims. These must also be unique.

Examples::

    >>> t = torch.randn(3,2,1)
    >>> t
    tensor([[[-0.3362],
            [-0.8437]],

            [[-0.9627],
            [ 0.1727]],

            [[ 0.5173],
            [-0.1398]]])
    >>> torch.movedim(t, 1, 0).shape
    torch.Size([2, 3, 1])
    >>> torch.movedim(t, 1, 0)
    tensor([[[-0.3362],
            [-0.9627],
            [ 0.5173]],

            [[-0.8437],
            [ 0.1727],
            [-0.1398]]])
    >>> torch.movedim(t, (1, 2), (0, 1)).shape
    torch.Size([2, 1, 3])
    >>> torch.movedim(t, (1, 2), (0, 1))
    tensor([[[-0.3362, -0.9627,  0.5173]],

            [[-0.8437,  0.1727, -0.1398]]])
"""

    title = 'MovedimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.movedim())



"""
WARNING: Module MsortNode was generated using fallback option. May contain bugs
"""

class MsortNode(Node):
    """
msort(input, *, out=None) -> Tensor

Sorts the elements of the :attr:`input` tensor along its first dimension
in ascending order by value.

.. note:: `torch.msort(t)` is equivalent to `torch.sort(t, dim=0)[0]`.
          See also :func:`torch.sort`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.randn(3, 4)
    >>> t
    tensor([[-0.1321,  0.4370, -1.2631, -1.1289],
            [-2.0527, -1.1250,  0.2275,  0.3077],
            [-0.0881, -0.1259, -0.5495,  1.0284]])
    >>> torch.msort(t)
    tensor([[-2.0527, -1.1250, -1.2631, -1.1289],
            [-0.1321, -0.1259, -0.5495,  0.3077],
            [-0.0881,  0.4370,  0.2275,  1.0284]])
"""

    title = 'MsortNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.msort())



"""
WARNING: Module MulNode was generated using fallback option. May contain bugs
"""

class MulNode(Node):
    """
mul(input, other, *, out=None) -> Tensor

Multiplies each element of the input :attr:`input` with the scalar
:attr:`other` and returns a new resulting tensor.

.. math::
    \text{out}_i = \text{other} \times \text{input}_i

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`other`
should be a real number, otherwise it should be an integer

Args:
    input (Tensor): the input tensor.
    other (Number): the number to be multiplied to each element of :attr:`input`

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> torch.mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])

.. function:: mul(input, other, *, out=None) -> Tensor

Each element of the tensor :attr:`input` is multiplied by the corresponding
element of the Tensor :attr:`other`. The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i


Args:
    input (Tensor): the first multiplicand tensor
    other (Tensor): the second multiplicand tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 1)
    >>> a
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> b = torch.randn(1, 4)
    >>> b
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> torch.mul(a, b)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
"""

    title = 'MulNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mul())



"""
WARNING: Module MultinomialNode was generated using fallback option. May contain bugs
"""

class MultinomialNode(Node):
    """
multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial probability distribution located in the corresponding row
of tensor :attr:`input`.

.. note::
    The rows of :attr:`input` do not need to sum to one (in which case we use
    the values as weights), but must be non-negative, finite and have
    a non-zero sum.

Indices are ordered from left to right according to when each was sampled
(first samples are placed in first column).

If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
:math:`(m \times \text{num\_samples})`.

If replacement is ``True``, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

.. note::
    When drawn without replacement, :attr:`num_samples` must be lower than
    number of non-zero elements in :attr:`input` (or the min number of non-zero
    elements in each row of :attr:`input` if it is a matrix).

Args:
    input (Tensor): the input tensor containing probabilities
    num_samples (int): number of samples to draw
    replacement (bool, optional): whether to draw with replacement or not

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.

Example::

    >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    >>> torch.multinomial(weights, 2)
    tensor([1, 2])
    >>> torch.multinomial(weights, 4) # ERROR!
    RuntimeError: invalid argument 2: invalid multinomial distribution (with replacement=False,
    not enough non-negative category to sample) at ../aten/src/TH/generic/THTensorRandom.cpp:320
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 2,  1,  1,  1])
"""

    title = 'MultinomialNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.multinomial())



"""
WARNING: Module MultiplyNode was generated using fallback option. May contain bugs
"""

class MultiplyNode(Node):
    """
multiply(input, other, *, out=None)

Alias for :func:`torch.mul`.
"""

    title = 'MultiplyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.multiply())



"""
WARNING: Module MultiprocessingNode was generated using fallback option. May contain bugs
"""

class MultiprocessingNode(Node):
    """
torch.multiprocessing is a wrapper around the native :mod:`multiprocessing`
module. It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""

    title = 'MultiprocessingNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.multiprocessing())



"""
WARNING: Module MvNode was generated using fallback option. May contain bugs
"""

class MvNode(Node):
    """
mv(input, vec, *, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`input` and the vector
:attr:`vec`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size :math:`m`, :attr:`out` will be 1-D of size :math:`n`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    input (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    tensor([ 1.0404, -0.6361])
"""

    title = 'MvNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mv())



"""
WARNING: Module MvlgammaNode was generated using fallback option. May contain bugs
"""

class MvlgammaNode(Node):
    """
mvlgamma(input, p) -> Tensor

Computes the `multivariate log-gamma function
<https://en.wikipedia.org/wiki/Multivariate_gamma_function>`_) with dimension
:math:`p` element-wise, given by

.. math::
    \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)

where :math:`C = \log(\pi) \times \frac{p (p - 1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

All elements must be greater than :math:`\frac{p - 1}{2}`, otherwise an error would be thrown.

Args:
    input (Tensor): the tensor to compute the multivariate log-gamma function
    p (int): the number of dimensions

Example::

    >>> a = torch.empty(2, 3).uniform_(1, 2)
    >>> a
    tensor([[1.6835, 1.8474, 1.1929],
            [1.0475, 1.7162, 1.4180]])
    >>> torch.mvlgamma(a, 2)
    tensor([[0.3928, 0.4007, 0.7586],
            [1.0311, 0.3901, 0.5049]])
"""

    title = 'MvlgammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.mvlgamma())



"""
WARNING: Module NameNode was generated using fallback option. May contain bugs
"""

class NameNode(Node):
    """str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'."""

    title = 'NameNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.name(self.input(0)))



"""
WARNING: Module Nan_to_numNode was generated using fallback option. May contain bugs
"""

class Nan_to_numNode(Node):
    """
nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None) -> Tensor

Replaces :literal:`NaN`, positive infinity, and negative infinity values in :attr:`input`
with the values specified by :attr:`nan`, :attr:`posinf`, and :attr:`neginf`, respectively.
By default, :literal:`NaN`s are replaced with zero, positive infinity is replaced with the
greatest finite value representable by :attr:`input`'s dtype, and negative infinity
is replaced with the least finite value representable by :attr:`input`'s dtype.

Args:
    input (Tensor): the input tensor.
    nan (Number, optional): the value to replace :literal:`NaN`\s with. Default is zero.
    posinf (Number, optional): if a Number, the value to replace positive infinity values with.
        If None, positive infinity values are replaced with the greatest finite value representable by :attr:`input`'s dtype.
        Default is None.
    neginf (Number, optional): if a Number, the value to replace negative infinity values with.
        If None, negative infinity values are replaced with the lowest finite value representable by :attr:`input`'s dtype.
        Default is None.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
    >>> torch.nan_to_num(x)
    tensor([ 0.0000e+00,  3.4028e+38, -3.4028e+38,  3.1400e+00])
    >>> torch.nan_to_num(x, nan=2.0)
    tensor([ 2.0000e+00,  3.4028e+38, -3.4028e+38,  3.1400e+00])
    >>> torch.nan_to_num(x, nan=2.0, posinf=1.0)
    tensor([ 2.0000e+00,  1.0000e+00, -3.4028e+38,  3.1400e+00])

"""

    title = 'Nan_to_numNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nan_to_num())



"""
WARNING: Module Nan_to_num_Node was generated using fallback option. May contain bugs
"""

class Nan_to_num_Node(Node):
    """None"""

    title = 'Nan_to_num_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nan_to_num_())



"""
WARNING: Module NanmedianNode was generated using fallback option. May contain bugs
"""

class NanmedianNode(Node):
    """
nanmedian(input) -> Tensor

Returns the median of the values in :attr:`input`, ignoring ``NaN`` values.

This function is identical to :func:`torch.median` when there are no ``NaN`` values in :attr:`input`.
When :attr:`input` has one or more ``NaN`` values, :func:`torch.median` will always return ``NaN``,
while this function will return the median of the non-``NaN`` elements in :attr:`input`.
If all the elements in :attr:`input` are ``NaN`` it will also return ``NaN``.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.tensor([1, float('nan'), 3, 2])
    >>> a.median()
    tensor(nan)
    >>> a.nanmedian()
    tensor(2.)

.. function:: nanmedian(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` contains the median of each row of :attr:`input`
in the dimension :attr:`dim`, ignoring ``NaN`` values, and ``indices`` contains the index of the median values
found in the dimension :attr:`dim`.

This function is identical to :func:`torch.median` when there are no ``NaN`` values in a reduced row. When a reduced row has
one or more ``NaN`` values, :func:`torch.median` will always reduce it to ``NaN``, while this function will reduce it to the
median of the non-``NaN`` elements. If all the elements in a reduced row are ``NaN`` then it will be reduced to ``NaN``, too.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out ((Tensor, Tensor), optional): The first tensor will be populated with the median values and the second
                                      tensor, which must have dtype long, with their indices in the dimension
                                      :attr:`dim` of :attr:`input`.

Example::

    >>> a = torch.tensor([[2, 3, 1], [float('nan'), 1, float('nan')]])
    >>> a
    tensor([[2., 3., 1.],
            [nan, 1., nan]])
    >>> a.median(0)
    torch.return_types.median(values=tensor([nan, 1., nan]), indices=tensor([1, 1, 1]))
    >>> a.nanmedian(0)
    torch.return_types.nanmedian(values=tensor([2., 1., 1.]), indices=tensor([0, 1, 0]))
"""

    title = 'NanmedianNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nanmedian())



"""
WARNING: Module NanquantileNode was generated using fallback option. May contain bugs
"""

class NanquantileNode(Node):
    """
nanquantile(input, q, dim=None, keepdim=False, *, out=None) -> Tensor

This is a variant of :func:`torch.quantile` that "ignores" ``NaN`` values,
computing the quantiles :attr:`q` as if ``NaN`` values in :attr:`input` did
not exist. If all values in a reduced row are ``NaN`` then the quantiles for
that reduction will be ``NaN``. See the documentation for :func:`torch.quantile`.

Args:
    input (Tensor): the input tensor.
    q (float or Tensor): a scalar or 1D tensor of quantile values in the range [0, 1]
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.tensor([float('nan'), 1, 2])
    >>> t.quantile(0.5)
    tensor(nan)
    >>> t.nanquantile(0.5)
    tensor(1.5000)
    >>> t = torch.tensor([[float('nan'), float('nan')], [1, 2]])
    >>> t
    tensor([[nan, nan],
            [1., 2.]])
    >>> t.nanquantile(0.5, dim=0)
    tensor([1., 2.])
    >>> t.nanquantile(0.5, dim=1)
    tensor([   nan, 1.5000])
"""

    title = 'NanquantileNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nanquantile())



"""
WARNING: Module NansumNode was generated using fallback option. May contain bugs
"""

class NansumNode(Node):
    """
nansum(input, *, dtype=None) -> Tensor

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.tensor([1., 2., float('nan'), 4.])
    >>> torch.nansum(a)
    tensor(7.)

.. function:: nansum(input, dim, keepdim=False, *, dtype=None) -> Tensor

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`, treating Not a Numbers (NaNs) as zero.
If :attr:`dim` is a list of dimensions, reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> torch.nansum(torch.tensor([1., float("nan")]))
    1.0
    >>> a = torch.tensor([[1, 2], [3., float("nan")]])
    >>> torch.nansum(a)
    tensor(6.)
    >>> torch.nansum(a, dim=0)
    tensor([4., 2.])
    >>> torch.nansum(a, dim=1)
    tensor([3., 3.])
"""

    title = 'NansumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nansum())



"""
WARNING: Module NarrowNode was generated using fallback option. May contain bugs
"""

class NarrowNode(Node):
    """
narrow(input, dim, start, length) -> Tensor

Returns a new tensor that is a narrowed version of :attr:`input` tensor. The
dimension :attr:`dim` is input from :attr:`start` to :attr:`start + length`. The
returned tensor and :attr:`input` tensor share the same underlying storage.

Args:
    input (Tensor): the tensor to narrow
    dim (int): the dimension along which to narrow
    start (int): the starting dimension
    length (int): the distance to the ending dimension

Example::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    >>> torch.narrow(x, 1, 1, 2)
    tensor([[ 2,  3],
            [ 5,  6],
            [ 8,  9]])
"""

    title = 'NarrowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.narrow())



"""
WARNING: Module Narrow_copyNode was generated using fallback option. May contain bugs
"""

class Narrow_copyNode(Node):
    """None"""

    title = 'Narrow_copyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.narrow_copy())



"""
WARNING: Module Native_batch_normNode was generated using fallback option. May contain bugs
"""

class Native_batch_normNode(Node):
    """None"""

    title = 'Native_batch_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.native_batch_norm())



"""
WARNING: Module Native_group_normNode was generated using fallback option. May contain bugs
"""

class Native_group_normNode(Node):
    """None"""

    title = 'Native_group_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.native_group_norm())



"""
WARNING: Module Native_layer_normNode was generated using fallback option. May contain bugs
"""

class Native_layer_normNode(Node):
    """None"""

    title = 'Native_layer_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.native_layer_norm())



"""
WARNING: Module Native_normNode was generated using fallback option. May contain bugs
"""

class Native_normNode(Node):
    """None"""

    title = 'Native_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.native_norm())



"""
WARNING: Module NeNode was generated using fallback option. May contain bugs
"""

class NeNode(Node):
    """
ne(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \neq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

Example::

    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
"""

    title = 'NeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ne())



"""
WARNING: Module NegNode was generated using fallback option. May contain bugs
"""

class NegNode(Node):
    """
neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
"""

    title = 'NegNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.neg())



"""
WARNING: Module Neg_Node was generated using fallback option. May contain bugs
"""

class Neg_Node(Node):
    """None"""

    title = 'Neg_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.neg_())



"""
WARNING: Module NegativeNode was generated using fallback option. May contain bugs
"""

class NegativeNode(Node):
    """
negative(input, *, out=None) -> Tensor

Alias for :func:`torch.neg`
"""

    title = 'NegativeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.negative())



"""
WARNING: Module Negative_Node was generated using fallback option. May contain bugs
"""

class Negative_Node(Node):
    """None"""

    title = 'Negative_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.negative_())



"""
WARNING: Module NextafterNode was generated using fallback option. May contain bugs
"""

class NextafterNode(Node):
    """
nextafter(input, other, *, out=None) -> Tensor

Return the next floating-point value after :attr:`input` towards :attr:`other`, elementwise.

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> eps = torch.finfo(torch.float32).eps
    >>> torch.nextafter(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])) == torch.tensor([eps + 1, 2 - eps])
    tensor([True, True])

"""

    title = 'NextafterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nextafter())



"""
WARNING: Module NnNode was generated using fallback option. May contain bugs
"""

class NnNode(Node):
    """None"""

    title = 'NnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nn())


class No_gradNode(Node):
    """Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> with torch.no_grad():
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
    """

    title = 'No_gradNode'
    init_inputs = [
        NodeInputBP('self'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.no_grad(self.input(0)))



"""
WARNING: Module NonzeroNode was generated using fallback option. May contain bugs
"""

class NonzeroNode(Node):
    """
nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

.. note::
    :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
    2-D tensor where each row is the index for a nonzero value.

    :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
    index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
    gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
    contains nonzero indices for a certain dimension.

    See below for more details on the two behaviors.

    When :attr:`input` is on CUDA, :func:`torch.nonzero() <torch.nonzero>` causes
    host-device synchronization.

**When** :attr:`as_tuple` **is ``False`` (default)**:

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The result is sorted lexicographically, with
the last index changing the fastest (C-style).

If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

**When** :attr:`as_tuple` **is ``True``**:

Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
each containing the indices (in that dimension) of all non-zero elements of
:attr:`input` .

If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
tensors of size :math:`z`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
value, it is treated as a one-dimensional tensor with one element.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (LongTensor, optional): the output tensor containing indices

Returns:
    LongTensor or tuple of LongTensor: If :attr:`as_tuple` is ``False``, the output
    tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
    each dimension, containing the indices of each nonzero element along that
    dimension.

Example::

    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
    (tensor([0, 1, 2, 4]),)
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
    >>> torch.nonzero(torch.tensor(5), as_tuple=True)
    (tensor([0]),)
"""

    title = 'NonzeroNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nonzero())


class NormNode(Node):
    """Returns the matrix norm or vector norm of a given tensor.

    .. warning::

        torch.norm is deprecated and may be removed in a future PyTorch release.

        Use :func:`torch.linalg.norm`, instead, or :func:`torch.linalg.vector_norm`
        when computing vector norms and :func:`torch.linalg.matrix_norm` when
        computing matrix norms. Note, however, the signature for these functions
        is slightly different than the signature for torch.norm.

    Args:
        input (Tensor): The input tensor. Its data type must be either a floating
            point or complex type. For complex inputs, the norm is calculated using the
            absolute value of each element. If the input is complex and neither
            :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is
            complexfloat).

        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            ======  ==============  ==========================
            ord     matrix norm     vector norm
            ======  ==============  ==========================
            'fro'   Frobenius norm  --
            'nuc'   nuclear norm    --
            Number  --              sum(abs(x)**ord)**(1./ord)
            ======  ==============  ==========================

            The vector norm can be calculated across any number of dimensions.
            The corresponding dimensions of :attr:`input` are flattened into
            one dimension, and the norm is calculated on the flattened
            dimension.

            Frobenius norm produces the same result as ``p=2`` in all cases
            except when :attr:`dim` is a list of three or more dims, in which
            case Frobenius norm throws an error.

            Nuclear norm can only be calculated across exactly two dimensions.

        dim (int, tuple of ints, list of ints, optional):
            Specifies which dimension or dimensions of :attr:`input` to
            calculate the norm across. If :attr:`dim` is ``None``, the norm will
            be calculated across all dimensions of :attr:`input`. If the norm
            type indicated by :attr:`p` does not support the specified number of
            dimensions, an error will occur.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        out (Tensor, optional): the output tensor. Ignored if
            :attr:`dim` = ``None`` and :attr:`out` = ``None``.
        dtype (:class:`torch.dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:'dtype' while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`torch.linalg.norm` with ``ord='fro'`` aligns
        with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example::

        >>> import torch
        >>> a = torch.arange(9, dtype= torch.float) - 4
        >>> b = a.reshape((3, 3))
        >>> torch.norm(a)
        tensor(7.7460)
        >>> torch.norm(b)
        tensor(7.7460)
        >>> torch.norm(a, float('inf'))
        tensor(4.)
        >>> torch.norm(b, float('inf'))
        tensor(4.)
        >>> c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
        >>> torch.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000])
        >>> torch.norm(c, dim=1)
        tensor([3.7417, 4.2426])
        >>> torch.norm(c, p=1, dim=1)
        tensor([6., 6.])
        >>> d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
        >>> torch.norm(d, dim=(1,2))
        tensor([ 3.7417, 11.2250])
        >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        (tensor(3.7417), tensor(11.2250))
    """

    title = 'NormNode'
    init_inputs = [
        NodeInputBP('input'),
NodeInputBP('p'),
NodeInputBP('dim'),
NodeInputBP('keepdim'),
NodeInputBP('out'),
NodeInputBP('dtype'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.norm(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4), self.input(5)))



"""
WARNING: Module Norm_except_dimNode was generated using fallback option. May contain bugs
"""

class Norm_except_dimNode(Node):
    """None"""

    title = 'Norm_except_dimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.norm_except_dim())



"""
WARNING: Module NormalNode was generated using fallback option. May contain bugs
"""

class NormalNode(Node):
    """
normal(mean, std, *, generator=None, out=None) -> Tensor

Returns a tensor of random numbers drawn from separate normal distributions
whose mean and standard deviation are given.

The :attr:`mean` is a tensor with the mean of
each output element's normal distribution

The :attr:`std` is a tensor with the standard deviation of
each output element's normal distribution

The shapes of :attr:`mean` and :attr:`std` don't need to match, but the
total number of elements in each tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`mean`
          is used as the shape for the returned output tensor

.. note:: When :attr:`std` is a CUDA tensor, this function synchronizes
          its device with the CPU.

Args:
    mean (Tensor): the tensor of per-element means
    std (Tensor): the tensor of per-element standard deviations

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
              8.0505,   8.1408,   9.0563,  10.0566])

.. function:: normal(mean=0.0, std, *, out=None) -> Tensor

Similar to the function above, but the means are shared among all drawn
elements.

Args:
    mean (float, optional): the mean for all distributions
    std (Tensor): the tensor of per-element standard deviations

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
    tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])

.. function:: normal(mean, std=1.0, *, out=None) -> Tensor

Similar to the function above, but the standard deviations are shared among
all drawn elements.

Args:
    mean (Tensor): the tensor of per-element means
    std (float, optional): the standard deviation for all distributions

Keyword args:
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=torch.arange(1., 6.))
    tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])

.. function:: normal(mean, std, size, *, out=None) -> Tensor

Similar to the function above, but the means and standard deviations are shared
among all drawn elements. The resulting tensor has size given by :attr:`size`.

Args:
    mean (float): the mean for all distributions
    std (float): the standard deviation for all distributions
    size (int...): a sequence of integers defining the shape of the output tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(2, 3, size=(1, 4))
    tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
"""

    title = 'NormalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.normal())



"""
WARNING: Module Not_equalNode was generated using fallback option. May contain bugs
"""

class Not_equalNode(Node):
    """
not_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ne`.
"""

    title = 'Not_equalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.not_equal())



"""
WARNING: Module Nuclear_normNode was generated using fallback option. May contain bugs
"""

class Nuclear_normNode(Node):
    """None"""

    title = 'Nuclear_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.nuclear_norm())



"""
WARNING: Module NumelNode was generated using fallback option. May contain bugs
"""

class NumelNode(Node):
    """
numel(input) -> int

Returns the total number of elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 2, 3, 4, 5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16

"""

    title = 'NumelNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.numel())



"""
WARNING: Module OnesNode was generated using fallback option. May contain bugs
"""

class OnesNode(Node):
    """
ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `1`, with the shape defined
by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword arguments:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.ones(2, 3)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> torch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])

"""

    title = 'OnesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ones())



"""
WARNING: Module Ones_likeNode was generated using fallback option. May contain bugs
"""

class Ones_likeNode(Node):
    """
ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`. ``torch.ones_like(input)`` is equivalent to
``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.ones_like(input, out=output)`` is equivalent to
    ``torch.ones(input.size(), out=output)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword arguments:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
"""

    title = 'Ones_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ones_like())



"""
WARNING: Module OnnxNode was generated using fallback option. May contain bugs
"""

class OnnxNode(Node):
    """None"""

    title = 'OnnxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.onnx())



"""
WARNING: Module OpsNode was generated using fallback option. May contain bugs
"""

class OpsNode(Node):
    """None"""

    title = 'OpsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ops())



"""
WARNING: Module OptimNode was generated using fallback option. May contain bugs
"""

class OptimNode(Node):
    """
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

    title = 'OptimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.optim())



"""
WARNING: Module OrgqrNode was generated using fallback option. May contain bugs
"""

class OrgqrNode(Node):
    """
orgqr(input, tau) -> Tensor

Alias for :func:`torch.linalg.householder_product`.
"""

    title = 'OrgqrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.orgqr())



"""
WARNING: Module OrmqrNode was generated using fallback option. May contain bugs
"""

class OrmqrNode(Node):
    """
ormqr(input, tau, other, left=True, transpose=False, *, out=None) -> Tensor

Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.

Multiplies a :math:`m \times n` matrix `C` (given by :attr:`other`) with a matrix `Q`,
where `Q` is represented using Householder reflectors `(input, tau)`.
See `Representation of Orthogonal or Unitary Matrices`_ for further details.

If :attr:`left` is `True` then `op(Q)` times `C` is computed, otherwise the result is `C` times `op(Q)`.
When :attr:`left` is `True`, the implicit matrix `Q` has size :math:`m \times m`.
It has size :math:`n \times n` otherwise.
If :attr:`transpose` is `True` then `op` is the conjugate transpose operation, otherwise it's a no-op.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batched inputs, and, if the input is batched, the output is batched with the same dimensions.

.. seealso::

        :func:`torch.geqrf` can be used to form the Householder representation `(input, tau)` of matrix `Q`
        from the QR decomposition.

Args:
    input (Tensor): tensor of shape `(*, mn, k)` where `*` is zero or more batch dimensions
                    and `mn` equals to `m` or `n` depending on the :attr:`left`.
    tau (Tensor): tensor of shape `(*, min(mn, k))` where `*` is zero or more batch dimensions.
    other (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    left (bool): controls the order of multiplication.
    transpose (bool): controls whether the matrix `Q` is conjugate transposed or not.

Keyword args:
    out (Tensor, optional): the output Tensor. Ignored if `None`. Default: `None`.

.. _Representation of Orthogonal or Unitary Matrices:
    https://www.netlib.org/lapack/lug/node128.html
"""

    title = 'OrmqrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ormqr())



"""
WARNING: Module OsNode was generated using fallback option. May contain bugs
"""

class OsNode(Node):
    """OS routines for NT or Posix depending on what system we're on.

This exports:
  - all functions from posix or nt, e.g. unlink, stat, etc.
  - os.path is either posixpath or ntpath
  - os.name is either 'posix' or 'nt'
  - os.curdir is a string representing the current directory (always '.')
  - os.pardir is a string representing the parent directory (always '..')
  - os.sep is the (or a most common) pathname separator ('/' or '\\')
  - os.extsep is the extension separator (always '.')
  - os.altsep is the alternate pathname separator (None or '/')
  - os.pathsep is the component separator used in $PATH etc
  - os.linesep is the line separator in text files ('\r' or '\n' or '\r\n')
  - os.defpath is the default search path for executables
  - os.devnull is the file path of the null device ('/dev/null', etc.)

Programs that import and use 'os' stand a better chance of being
portable between different platforms.  Of course, they must then
only use functions that are defined by all platforms (e.g., unlink
and opendir), and leave all pathname manipulation to os.path
(e.g., split and join).
"""

    title = 'OsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.os(self.input(0)))



"""
WARNING: Module OuterNode was generated using fallback option. May contain bugs
"""

class OuterNode(Node):
    """
outer(input, vec2, *, out=None) -> Tensor

Outer product of :attr:`input` and :attr:`vec2`.
If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \times m)`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    input (Tensor): 1-D input vector
    vec2 (Tensor): 1-D input vector

Keyword args:
    out (Tensor, optional): optional output matrix

Example::

    >>> v1 = torch.arange(1., 5.)
    >>> v2 = torch.arange(1., 4.)
    >>> torch.outer(v1, v2)
    tensor([[  1.,   2.,   3.],
            [  2.,   4.,   6.],
            [  3.,   6.,   9.],
            [  4.,   8.,  12.]])
"""

    title = 'OuterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.outer())



"""
WARNING: Module OverridesNode was generated using fallback option. May contain bugs
"""

class OverridesNode(Node):
    """
Python implementation of ``__torch_function__``

While most of the torch API and handling for ``__torch_function__`` happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for ``__torch_function__`` overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""

    title = 'OverridesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.overrides())



"""
WARNING: Module PackageNode was generated using fallback option. May contain bugs
"""

class PackageNode(Node):
    """None"""

    title = 'PackageNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.package())



"""
WARNING: Module Pairwise_distanceNode was generated using fallback option. May contain bugs
"""

class Pairwise_distanceNode(Node):
    """None"""

    title = 'Pairwise_distanceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pairwise_distance())



"""
WARNING: Module Parse_irNode was generated using fallback option. May contain bugs
"""

class Parse_irNode(Node):
    """parse_ir(arg0: str) -> torch::jit::Graph
"""

    title = 'Parse_irNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.parse_ir(self.input(0)))



"""
WARNING: Module Parse_schemaNode was generated using fallback option. May contain bugs
"""

class Parse_schemaNode(Node):
    """parse_schema(arg0: str) -> c10::FunctionSchema
"""

    title = 'Parse_schemaNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.parse_schema(self.input(0)))



"""
WARNING: Module Parse_type_commentNode was generated using fallback option. May contain bugs
"""

class Parse_type_commentNode(Node):
    """parse_type_comment(arg0: str) -> torch._C._jit_tree_views.Decl
"""

    title = 'Parse_type_commentNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.parse_type_comment(self.input(0)))


class Pca_lowrankNode(Node):
    """Performs linear Principal Component Analysis (PCA) on a low-rank
    matrix, batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :math:`A` such that :math:`A = U diag(S) V^T`.

    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:

                - :math:`A` is a data matrix with ``m`` samples and
                  ``n`` features

                - the :math:`V` columns represent the principal directions

                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of
                  :math:`A^T A / (m - 1)` which is the covariance of
                  ``A`` when ``center=True`` is provided.

                - ``matmul(A, V[:, :k])`` projects data to the first k
                  principal components

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:

                - :math:`U` is m x q matrix

                - :math:`S` is q-vector

                - :math:`V` is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of
                           :math:`A`. By default, ``q = min(6, m,
                           n)``.

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2.

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """

    title = 'Pca_lowrankNode'
    init_inputs = [
        NodeInputBP('A'),
NodeInputBP('q'),
NodeInputBP('center'),
NodeInputBP('niter'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pca_lowrank(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module PdistNode was generated using fallback option. May contain bugs
"""

class PdistNode(Node):
    """
pdist(input, p=2) -> Tensor

Computes the p-norm distance between every pair of row vectors in the input.
This is identical to the upper triangular portion, excluding the diagonal, of
`torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
if the rows are contiguous.

If input has shape :math:`N \times M` then the output will have shape
:math:`\frac{1}{2} N (N - 1)`.

This function is equivalent to `scipy.spatial.distance.pdist(input,
'minkowski', p=p)` if :math:`p \in (0, \infty)`. When :math:`p = 0` it is
equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`.
When :math:`p = \infty`, the closest scipy function is
`scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.

Args:
    input: input tensor of shape :math:`N \times M`.
    p: p value for the p-norm distance to calculate between each vector pair
        :math:`\in [0, \infty]`.
"""

    title = 'PdistNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pdist())



"""
WARNING: Module Per_channel_affineNode was generated using fallback option. May contain bugs
"""

class Per_channel_affineNode(Node):
    """None"""

    title = 'Per_channel_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.per_channel_affine())



"""
WARNING: Module Per_channel_affine_float_qparamsNode was generated using fallback option. May contain bugs
"""

class Per_channel_affine_float_qparamsNode(Node):
    """None"""

    title = 'Per_channel_affine_float_qparamsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.per_channel_affine_float_qparams())



"""
WARNING: Module Per_channel_symmetricNode was generated using fallback option. May contain bugs
"""

class Per_channel_symmetricNode(Node):
    """None"""

    title = 'Per_channel_symmetricNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.per_channel_symmetric())



"""
WARNING: Module Per_tensor_affineNode was generated using fallback option. May contain bugs
"""

class Per_tensor_affineNode(Node):
    """None"""

    title = 'Per_tensor_affineNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.per_tensor_affine())



"""
WARNING: Module Per_tensor_symmetricNode was generated using fallback option. May contain bugs
"""

class Per_tensor_symmetricNode(Node):
    """None"""

    title = 'Per_tensor_symmetricNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.per_tensor_symmetric())



"""
WARNING: Module PermuteNode was generated using fallback option. May contain bugs
"""

class PermuteNode(Node):
    """
permute(input, dims) -> Tensor

Returns a view of the original tensor :attr:`input` with its dimensions permuted.

Args:
    {input}
    dims (tuple of ints): The desired ordering of dimensions

Example:
    >>> x = torch.randn(2, 3, 5)
    >>> x.size()
    torch.Size([2, 3, 5])
    >>> torch.permute(x, (2, 0, 1)).size()
    torch.Size([5, 2, 3])
"""

    title = 'PermuteNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.permute())



"""
WARNING: Module PinverseNode was generated using fallback option. May contain bugs
"""

class PinverseNode(Node):
    """
pinverse(input, rcond=1e-15) -> Tensor

Alias for :func:`torch.linalg.pinv`
"""

    title = 'PinverseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pinverse())



"""
WARNING: Module Pixel_shuffleNode was generated using fallback option. May contain bugs
"""

class Pixel_shuffleNode(Node):
    """
pixel_shuffle(input, upscale_factor) -> Tensor

Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :attr:`upscale_factor`.

See :class:`~torch.nn.PixelShuffle` for details.

Args:
    input (Tensor): the input tensor
    upscale_factor (int): factor to increase spatial resolution by

Examples::

    >>> input = torch.randn(1, 9, 4, 4)
    >>> output = torch.nn.functional.pixel_shuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])
"""

    title = 'Pixel_shuffleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pixel_shuffle())



"""
WARNING: Module Pixel_unshuffleNode was generated using fallback option. May contain bugs
"""

class Pixel_unshuffleNode(Node):
    """
pixel_unshuffle(input, downscale_factor) -> Tensor

Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a
tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
:math:`(*, C \times r^2, H, W)`, where r is the :attr:`downscale_factor`.

See :class:`~torch.nn.PixelUnshuffle` for details.

Args:
    input (Tensor): the input tensor
    downscale_factor (int): factor to increase spatial resolution by

Examples::

    >>> input = torch.randn(1, 1, 12, 12)
    >>> output = torch.nn.functional.pixel_unshuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 9, 4, 4])
"""

    title = 'Pixel_unshuffleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pixel_unshuffle())



"""
WARNING: Module PlatformNode was generated using fallback option. May contain bugs
"""

class PlatformNode(Node):
    """ This module tries to retrieve as much platform-identifying data as
    possible. It makes this information available via function APIs.

    If called from the command line, it prints the platform
    information concatenated as single string to stdout. The output
    format is useable as part of a filename.

"""

    title = 'PlatformNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.platform(self.input(0)))



"""
WARNING: Module PoissonNode was generated using fallback option. May contain bugs
"""

class PoissonNode(Node):
    """
poisson(input, generator=None) -> Tensor

Returns a tensor of the same size as :attr:`input` with each element
sampled from a Poisson distribution with rate parameter given by the corresponding
element in :attr:`input` i.e.,

.. math::
    \text{out}_i \sim \text{Poisson}(\text{input}_i)

Args:
    input (Tensor): the input tensor containing the rates of the Poisson distribution

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling

Example::

    >>> rates = torch.rand(4, 4) * 5  # rate parameter between 0 and 5
    >>> torch.poisson(rates)
    tensor([[9., 1., 3., 5.],
            [8., 6., 6., 0.],
            [0., 4., 5., 3.],
            [2., 1., 4., 2.]])
"""

    title = 'PoissonNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.poisson())



"""
WARNING: Module Poisson_nll_lossNode was generated using fallback option. May contain bugs
"""

class Poisson_nll_lossNode(Node):
    """None"""

    title = 'Poisson_nll_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.poisson_nll_loss())



"""
WARNING: Module PolarNode was generated using fallback option. May contain bugs
"""

class PolarNode(Node):
    """
polar(abs, angle, *, out=None) -> Tensor

Constructs a complex tensor whose elements are Cartesian coordinates
corresponding to the polar coordinates with absolute value :attr:`abs` and angle
:attr:`angle`.

.. math::
    \text{out} = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j

Args:
    abs (Tensor): The absolute value the complex tensor. Must be float or
        double.
    angle (Tensor): The angle of the complex tensor. Must be same dtype as
        :attr:`abs`.

Keyword args:
    out (Tensor): If the inputs are ``torch.float32``, must be
        ``torch.complex64``. If the inputs are ``torch.float64``, must be
        ``torch.complex128``.

Example::

    >>> import numpy as np
    >>> abs = torch.tensor([1, 2], dtype=torch.float64)
    >>> angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
    >>> z = torch.polar(abs, angle)
    >>> z
    tensor([(0.0000+1.0000j), (-1.4142-1.4142j)], dtype=torch.complex128)
"""

    title = 'PolarNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.polar())



"""
WARNING: Module PolygammaNode was generated using fallback option. May contain bugs
"""

class PolygammaNode(Node):
    """
polygamma(n, input, *, out=None) -> Tensor

Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`.
:math:`n \geq 0` is called the order of the polygamma function.

.. math::
    \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)

.. note::
    This function is implemented only for nonnegative integers :math:`n \geq 0`.

Args:
    n (int): the order of the polygamma function
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 0.5])
    >>> torch.polygamma(1, a)
    tensor([1.64493, 4.9348])
    >>> torch.polygamma(2, a)
    tensor([ -2.4041, -16.8288])
    >>> torch.polygamma(3, a)
    tensor([ 6.4939, 97.4091])
    >>> torch.polygamma(4, a)
    tensor([ -24.8863, -771.4742])
"""

    title = 'PolygammaNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.polygamma())



"""
WARNING: Module PositiveNode was generated using fallback option. May contain bugs
"""

class PositiveNode(Node):
    """
positive(input) -> Tensor

Returns :attr:`input`.
Throws a runtime error if :attr:`input` is a bool tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> t = torch.randn(5)
    >>> t
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.positive(t)
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
"""

    title = 'PositiveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.positive())



"""
WARNING: Module PowNode was generated using fallback option. May contain bugs
"""

class PowNode(Node):
    """
pow(input, exponent, *, out=None) -> Tensor

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a `Tensor`
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

.. math::
    \text{out}_i = x_i ^ \text{exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    \text{out}_i = x_i ^ {\text{exponent}_i}

When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor.
    exponent (float or tensor): the exponent value

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
    >>> torch.pow(a, 2)
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
    >>> exp = torch.arange(1., 5.)

    >>> a = torch.arange(1., 5.)
    >>> a
    tensor([ 1.,  2.,  3.,  4.])
    >>> exp
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.pow(a, exp)
    tensor([   1.,    4.,   27.,  256.])

.. function:: pow(self, exponent, *, out=None) -> Tensor

:attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
The returned tensor :attr:`out` is of the same shape as :attr:`exponent`

The operation applied is:

.. math::
    \text{out}_i = \text{self} ^ {\text{exponent}_i}

Args:
    self (float): the scalar base value for the power operation
    exponent (Tensor): the exponent tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> exp = torch.arange(1., 5.)
    >>> base = 2
    >>> torch.pow(base, exp)
    tensor([  2.,   4.,   8.,  16.])
"""

    title = 'PowNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.pow())



"""
WARNING: Module PreluNode was generated using fallback option. May contain bugs
"""

class PreluNode(Node):
    """None"""

    title = 'PreluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.prelu())


class Prepare_multiprocessing_environmentNode(Node):
    """None"""

    title = 'Prepare_multiprocessing_environmentNode'
    init_inputs = [
        NodeInputBP('path'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.prepare_multiprocessing_environment(self.input(0)))



"""
WARNING: Module Preserve_formatNode was generated using fallback option. May contain bugs
"""

class Preserve_formatNode(Node):
    """None"""

    title = 'Preserve_formatNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.preserve_format())



"""
WARNING: Module ProdNode was generated using fallback option. May contain bugs
"""

class ProdNode(Node):
    """
prod(input, *, dtype=None) -> Tensor

Returns the product of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8020,  0.5428, -1.5854]])
    >>> torch.prod(a)
    tensor(0.6902)

.. function:: prod(input, dim, keepdim=False, *, dtype=None) -> Tensor

Returns the product of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 0.5261, -0.3837],
            [ 1.1857, -0.2498],
            [-1.1646,  0.0705],
            [ 1.1131, -1.0629]])
    >>> torch.prod(a, 1)
    tensor([-0.2018, -0.2962, -0.0821, -1.1831])
"""

    title = 'ProdNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.prod())



"""
WARNING: Module ProfilerNode was generated using fallback option. May contain bugs
"""

class ProfilerNode(Node):
    """
PyTorch Profiler is a tool that allows the collecton of the performance metrics during the training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""

    title = 'ProfilerNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.profiler())



"""
WARNING: Module Promote_typesNode was generated using fallback option. May contain bugs
"""

class Promote_typesNode(Node):
    """
promote_types(type1, type2) -> dtype

Returns the :class:`torch.dtype` with the smallest size and scalar kind that is
not smaller nor of lower kind than either `type1` or `type2`. See type promotion
:ref:`documentation <type-promotion-doc>` for more information on the type
promotion logic.

Args:
    type1 (:class:`torch.dtype`)
    type2 (:class:`torch.dtype`)

Example::

    >>> torch.promote_types(torch.int32, torch.float32)
    torch.float32
    >>> torch.promote_types(torch.uint8, torch.long)
    torch.long
"""

    title = 'Promote_typesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.promote_types())



"""
WARNING: Module PutNode was generated using fallback option. May contain bugs
"""

class PutNode(Node):
    """None"""

    title = 'PutNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.put())



"""
WARNING: Module Q_per_channel_axisNode was generated using fallback option. May contain bugs
"""

class Q_per_channel_axisNode(Node):
    """None"""

    title = 'Q_per_channel_axisNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.q_per_channel_axis())



"""
WARNING: Module Q_per_channel_scalesNode was generated using fallback option. May contain bugs
"""

class Q_per_channel_scalesNode(Node):
    """None"""

    title = 'Q_per_channel_scalesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.q_per_channel_scales())



"""
WARNING: Module Q_per_channel_zero_pointsNode was generated using fallback option. May contain bugs
"""

class Q_per_channel_zero_pointsNode(Node):
    """None"""

    title = 'Q_per_channel_zero_pointsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.q_per_channel_zero_points())



"""
WARNING: Module Q_scaleNode was generated using fallback option. May contain bugs
"""

class Q_scaleNode(Node):
    """None"""

    title = 'Q_scaleNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.q_scale())



"""
WARNING: Module Q_zero_pointNode was generated using fallback option. May contain bugs
"""

class Q_zero_pointNode(Node):
    """None"""

    title = 'Q_zero_pointNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.q_zero_point())



"""
WARNING: Module Qint32Node was generated using fallback option. May contain bugs
"""

class Qint32Node(Node):
    """None"""

    title = 'Qint32Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.qint32())



"""
WARNING: Module Qint8Node was generated using fallback option. May contain bugs
"""

class Qint8Node(Node):
    """None"""

    title = 'Qint8Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.qint8())



"""
WARNING: Module QrNode was generated using fallback option. May contain bugs
"""

class QrNode(Node):
    """
qr(input, some=True, *, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix or a batch of matrices :attr:`input`,
and returns a namedtuple (Q, R) of tensors such that :math:`\text{input} = Q R`
with :math:`Q` being an orthogonal matrix or batch of orthogonal matrices and
:math:`R` being an upper triangular matrix or batch of upper triangular matrices.

If :attr:`some` is ``True``, then this function returns the thin (reduced) QR factorization.
Otherwise, if :attr:`some` is ``False``, this function returns the complete QR factorization.

.. warning::

    :func:`torch.qr` is deprecated in favor of :func:`torch.linalg.qr`
    and will be removed in a future PyTorch release. The boolean parameter :attr:`some` has been
    replaced with a string parameter :attr:`mode`.

    ``Q, R = torch.qr(A)`` should be replaced with

    .. code:: python

        Q, R = torch.linalg.qr(A)

    ``Q, R = torch.qr(A, some=False)`` should be replaced with

    .. code:: python

        Q, R = torch.linalg.qr(A, mode="complete")

.. warning::
          If you plan to backpropagate through QR, note that the current backward implementation
          is only well-defined when the first :math:`\min(input.size(-1), input.size(-2))`
          columns of :attr:`input` are linearly independent.
          This behavior will propably change once QR supports pivoting.

.. note:: This function uses LAPACK for CPU inputs and MAGMA for CUDA inputs,
          and may produce different (valid) decompositions on different device types
          or different platforms.

Args:
    input (Tensor): the input tensor of size :math:`(*, m, n)` where `*` is zero or more
                batch dimensions consisting of matrices of dimension :math:`m \times n`.
    some (bool, optional): Set to ``True`` for reduced QR decomposition and ``False`` for
                complete QR decomposition. If `k = min(m, n)` then:

                  * ``some=True`` : returns `(Q, R)` with dimensions (m, k), (k, n) (default)

                  * ``'some=False'``: returns `(Q, R)` with dimensions (m, m), (m, n)

Keyword args:
    out (tuple, optional): tuple of `Q` and `R` tensors.
                The dimensions of `Q` and `R` are detailed in the description of :attr:`some` above.

Example::

    >>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> q, r = torch.qr(a)
    >>> q
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    >>> r
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    >>> torch.mm(q, r).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    >>> a = torch.randn(3, 4, 5)
    >>> q, r = torch.qr(a, some=False)
    >>> torch.allclose(torch.matmul(q, r), a)
    True
    >>> torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))
    True
"""

    title = 'QrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.qr())


class QschemeNode(Node):
    """None"""

    title = 'QschemeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.qscheme())



"""
WARNING: Module QuantileNode was generated using fallback option. May contain bugs
"""

class QuantileNode(Node):
    """
quantile(input, q, dim=None, keepdim=False, *, out=None) -> Tensor

Computes the q-th quantiles of each row of the :attr:`input` tensor
along the dimension :attr:`dim`.

To compute the quantile, we map q in [0, 1] to the range of indices [0, n] to find the location
of the quantile in the sorted input. If the quantile lies between two data points ``a < b`` with
indices ``i`` and ``j`` in the sorted order, result is computed using linear interpolation as follows:

``a + (b - a) * fraction``, where ``fraction`` is the fractional part of the computed quantile index.

If :attr:`q` is a 1D tensor, the first dimension of the output represents the quantiles and has size
equal to the size of :attr:`q`, the remaining dimensions are what remains from the reduction.

.. note::
    By default :attr:`dim` is ``None`` resulting in the :attr:`input` tensor being flattened before computation.

Args:
    input (Tensor): the input tensor.
    q (float or Tensor): a scalar or 1D tensor of values in the range [0, 1].
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(2, 3)
    >>> a
    tensor([[ 0.0795, -1.2117,  0.9765],
            [ 1.1707,  0.6706,  0.4884]])
    >>> q = torch.tensor([0.25, 0.5, 0.75])
    >>> torch.quantile(a, q, dim=1, keepdim=True)
    tensor([[[-0.5661],
            [ 0.5795]],

            [[ 0.0795],
            [ 0.6706]],

            [[ 0.5280],
            [ 0.9206]]])
    >>> torch.quantile(a, q, dim=1, keepdim=True).shape
    torch.Size([3, 2, 1])
    >>> a = torch.arange(4.)
    >>> a
    tensor([0., 1., 2., 3.])
"""

    title = 'QuantileNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantile())



"""
WARNING: Module QuantizationNode was generated using fallback option. May contain bugs
"""

class QuantizationNode(Node):
    """None"""

    title = 'QuantizationNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantization())



"""
WARNING: Module Quantize_per_channelNode was generated using fallback option. May contain bugs
"""

class Quantize_per_channelNode(Node):
    """
quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor

Converts a float tensor to a per-channel quantized tensor with given scales and zero points.

Arguments:
    input (Tensor): float tensor to quantize
    scales (Tensor): float 1D tensor of scales to use, size should match ``input.size(axis)``
    zero_points (int): integer 1D tensor of offset to use, size should match ``input.size(axis)``
    axis (int): dimension on which apply per-channel quantization
    dtype (:class:`torch.dtype`): the desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``

Returns:
    Tensor: A newly quantized tensor

Example::

    >>> x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8)
    tensor([[-1.,  0.],
            [ 1.,  2.]], size=(2, 2), dtype=torch.quint8,
           quantization_scheme=torch.per_channel_affine,
           scale=tensor([0.1000, 0.0100], dtype=torch.float64),
           zero_point=tensor([10,  0]), axis=0)
    >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8).int_repr()
    tensor([[  0,  10],
            [100, 200]], dtype=torch.uint8)
"""

    title = 'Quantize_per_channelNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantize_per_channel())



"""
WARNING: Module Quantize_per_tensorNode was generated using fallback option. May contain bugs
"""

class Quantize_per_tensorNode(Node):
    """
quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor

Converts a float tensor to a quantized tensor with given scale and zero point.

Arguments:
    input (Tensor): float tensor to quantize
    scale (float): scale to apply in quantization formula
    zero_point (int): offset in integer value that maps to float zero
    dtype (:class:`torch.dtype`): the desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``

Returns:
    Tensor: A newly quantized tensor

Example::

    >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)
    tensor([-1.,  0.,  1.,  2.], size=(4,), dtype=torch.quint8,
           quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=10)
    >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8).int_repr()
    tensor([ 0, 10, 20, 30], dtype=torch.uint8)
"""

    title = 'Quantize_per_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantize_per_tensor())



"""
WARNING: Module Quantized_batch_normNode was generated using fallback option. May contain bugs
"""

class Quantized_batch_normNode(Node):
    """None"""

    title = 'Quantized_batch_normNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_batch_norm())



"""
WARNING: Module Quantized_gruNode was generated using fallback option. May contain bugs
"""

class Quantized_gruNode(Node):
    """quantized_gru(*args, **kwargs) -> object

Automatically bound operator 'aten::quantized_gru' with schema(s):
  aten::quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
  aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
  aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
  aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)

"""

    title = 'Quantized_gruNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_gru(self.input(0), self.input(1)))



"""
WARNING: Module Quantized_gru_cellNode was generated using fallback option. May contain bugs
"""

class Quantized_gru_cellNode(Node):
    """None"""

    title = 'Quantized_gru_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_gru_cell())



"""
WARNING: Module Quantized_lstmNode was generated using fallback option. May contain bugs
"""

class Quantized_lstmNode(Node):
    """quantized_lstm(*args, **kwargs) -> object

Automatically bound operator 'aten::quantized_lstm' with schema(s):
  aten::quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
  aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
  aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
  aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)

"""

    title = 'Quantized_lstmNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_lstm(self.input(0), self.input(1)))



"""
WARNING: Module Quantized_lstm_cellNode was generated using fallback option. May contain bugs
"""

class Quantized_lstm_cellNode(Node):
    """None"""

    title = 'Quantized_lstm_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_lstm_cell())



"""
WARNING: Module Quantized_max_pool1dNode was generated using fallback option. May contain bugs
"""

class Quantized_max_pool1dNode(Node):
    """None"""

    title = 'Quantized_max_pool1dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_max_pool1d())



"""
WARNING: Module Quantized_max_pool2dNode was generated using fallback option. May contain bugs
"""

class Quantized_max_pool2dNode(Node):
    """None"""

    title = 'Quantized_max_pool2dNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_max_pool2d())



"""
WARNING: Module Quantized_rnn_relu_cellNode was generated using fallback option. May contain bugs
"""

class Quantized_rnn_relu_cellNode(Node):
    """None"""

    title = 'Quantized_rnn_relu_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_rnn_relu_cell())



"""
WARNING: Module Quantized_rnn_tanh_cellNode was generated using fallback option. May contain bugs
"""

class Quantized_rnn_tanh_cellNode(Node):
    """None"""

    title = 'Quantized_rnn_tanh_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quantized_rnn_tanh_cell())



"""
WARNING: Module QuasirandomNode was generated using fallback option. May contain bugs
"""

class QuasirandomNode(Node):
    """None"""

    title = 'QuasirandomNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quasirandom())



"""
WARNING: Module Quint4x2Node was generated using fallback option. May contain bugs
"""

class Quint4x2Node(Node):
    """None"""

    title = 'Quint4x2Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quint4x2())



"""
WARNING: Module Quint8Node was generated using fallback option. May contain bugs
"""

class Quint8Node(Node):
    """None"""

    title = 'Quint8Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.quint8())



"""
WARNING: Module Rad2degNode was generated using fallback option. May contain bugs
"""

class Rad2degNode(Node):
    """
rad2deg(input, *, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input`
converted from angles in radians to degrees.

Args:
    input (Tensor): the input tensor.

Keyword arguments:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
    >>> torch.rad2deg(a)
    tensor([[ 180.0233, -180.0233],
            [ 359.9894, -359.9894],
            [  89.9544,  -89.9544]])

"""

    title = 'Rad2degNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rad2deg())



"""
WARNING: Module Rad2deg_Node was generated using fallback option. May contain bugs
"""

class Rad2deg_Node(Node):
    """None"""

    title = 'Rad2deg_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rad2deg_())



"""
WARNING: Module RandNode was generated using fallback option. May contain bugs
"""

class RandNode(Node):
    """
rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a uniform distribution
on the interval :math:`[0, 1)`

The shape of the tensor is defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.rand(4)
    tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
    >>> torch.rand(2, 3)
    tensor([[ 0.8237,  0.5781,  0.6879],
            [ 0.3816,  0.7249,  0.0998]])
"""

    title = 'RandNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rand())



"""
WARNING: Module Rand_likeNode was generated using fallback option. May contain bugs
"""

class Rand_likeNode(Node):
    """
rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a uniform distribution on the interval :math:`[0, 1)`.
``torch.rand_like(input)`` is equivalent to
``torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

"""

    title = 'Rand_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rand_like())



"""
WARNING: Module RandintNode was generated using fallback option. May contain bugs
"""

class RandintNode(Node):
    """
randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random integers generated uniformly
between :attr:`low` (inclusive) and :attr:`high` (exclusive).

The shape of the tensor is defined by the variable argument :attr:`size`.

.. note::
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.
    size (tuple): a tuple defining the shape of the output tensor.

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.randint(3, 5, (3,))
    tensor([4, 3, 4])


    >>> torch.randint(10, (2, 2))
    tensor([[0, 2],
            [5, 5]])


    >>> torch.randint(3, 10, (2, 2))
    tensor([[4, 5],
            [6, 7]])


"""

    title = 'RandintNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.randint())



"""
WARNING: Module Randint_likeNode was generated using fallback option. May contain bugs
"""

class Randint_likeNode(Node):
    """
randint_like(input, low=0, high, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same shape as Tensor :attr:`input` filled with
random integers generated uniformly between :attr:`low` (inclusive) and
:attr:`high` (exclusive).

.. note:
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

"""

    title = 'Randint_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.randint_like())



"""
WARNING: Module RandnNode was generated using fallback option. May contain bugs
"""

class RandnNode(Node):
    """
randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a normal distribution
with mean `0` and variance `1` (also called the standard normal
distribution).

.. math::
    \text{out}_{i} \sim \mathcal{N}(0, 1)

The shape of the tensor is defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.randn(4)
    tensor([-2.1436,  0.9966,  2.3426, -0.6366])
    >>> torch.randn(2, 3)
    tensor([[ 1.5954,  2.8929, -1.0923],
            [ 1.1719, -0.4709, -0.1996]])
"""

    title = 'RandnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.randn())



"""
WARNING: Module Randn_likeNode was generated using fallback option. May contain bugs
"""

class Randn_likeNode(Node):
    """
randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a normal distribution with mean 0 and variance 1.
``torch.randn_like(input)`` is equivalent to
``torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

"""

    title = 'Randn_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.randn_like())



"""
WARNING: Module RandomNode was generated using fallback option. May contain bugs
"""

class RandomNode(Node):
    """None"""

    title = 'RandomNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.random())



"""
WARNING: Module RandpermNode was generated using fallback option. May contain bugs
"""

class RandpermNode(Node):
    """
randperm(n, *, generator=None, out=None, dtype=torch.int64,layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a random permutation of integers from ``0`` to ``n - 1``.

Args:
    n (int): the upper bound (exclusive)

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: ``torch.int64``.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.

Example::

    >>> torch.randperm(4)
    tensor([2, 1, 0, 3])
"""

    title = 'RandpermNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.randperm())



"""
WARNING: Module RangeNode was generated using fallback option. May contain bugs
"""

class RangeNode(Node):
    """
range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
the gap between two values in the tensor.

.. math::
    \text{out}_{i+1} = \text{out}_i + \text{step}.

.. warning::
    This function is deprecated and will be removed in a future release because its behavior is inconsistent with
    Python's range builtin. Instead, use :func:`torch.arange`, which produces values in [start, end).

Args:
    start (float): the starting value for the set of points. Default: ``0``.
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points. Default: ``1``.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). If `dtype` is not given, infer the data type from the other input
        arguments. If any of `start`, `end`, or `stop` are floating-point, the
        `dtype` is inferred to be the default dtype, see
        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
        be `torch.int64`.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.range(1, 4)
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.range(1, 4, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])
"""

    title = 'RangeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.range())



"""
WARNING: Module RavelNode was generated using fallback option. May contain bugs
"""

class RavelNode(Node):
    """
ravel(input) -> Tensor

Return a contiguous flattened tensor. A copy is made only if needed.

Args:
    input (Tensor): the input tensor.

Example::

    >>> t = torch.tensor([[[1, 2],
    ...                    [3, 4]],
    ...                   [[5, 6],
    ...                    [7, 8]]])
    >>> torch.ravel(t)
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
"""

    title = 'RavelNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.ravel())



"""
WARNING: Module RealNode was generated using fallback option. May contain bugs
"""

class RealNode(Node):
    """
real(input) -> Tensor

Returns a new tensor containing real values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

.. warning::
    :func:`real` is only supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.real
    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])

"""

    title = 'RealNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.real())



"""
WARNING: Module ReciprocalNode was generated using fallback option. May contain bugs
"""

class ReciprocalNode(Node):
    """
reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
"""

    title = 'ReciprocalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.reciprocal())



"""
WARNING: Module Reciprocal_Node was generated using fallback option. May contain bugs
"""

class Reciprocal_Node(Node):
    """None"""

    title = 'Reciprocal_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.reciprocal_())



"""
WARNING: Module ReluNode was generated using fallback option. May contain bugs
"""

class ReluNode(Node):
    """None"""

    title = 'ReluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.relu())



"""
WARNING: Module Relu_Node was generated using fallback option. May contain bugs
"""

class Relu_Node(Node):
    """
relu_(input) -> Tensor

In-place version of :func:`~relu`.
"""

    title = 'Relu_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.relu_())



"""
WARNING: Module RemainderNode was generated using fallback option. May contain bugs
"""

class RemainderNode(Node):
    """
remainder(input, other, *, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor :attr:`other`.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

.. note::
    Complex inputs are not supported. In some cases, it is not mathematically
    possible to satisfy the definition of a modulo operation with complex numbers.
    See :func:`torch.fmod` for how division by zero is handled.

Args:
    input (Tensor): the dividend
    other (Tensor or Scalar): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
    >>> torch.remainder(torch.tensor([1, 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

.. seealso::

        :func:`torch.fmod`, which computes the element-wise remainder of
        division equivalently to the C library function ``fmod()``.
"""

    title = 'RemainderNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.remainder())



"""
WARNING: Module RenormNode was generated using fallback option. May contain bugs
"""

class RenormNode(Node):
    """
renorm(input, p, dim, maxnorm, *, out=None) -> Tensor

Returns a tensor where each sub-tensor of :attr:`input` along dimension
:attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
than the value :attr:`maxnorm`

.. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

Args:
    input (Tensor): the input tensor.
    p (float): the power for the norm computation
    dim (int): the dimension to slice over to get the sub-tensors
    maxnorm (float): the maximum norm to keep each sub-tensor under

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    tensor([ 2.,  2.,  2.])
    >>> x[2].fill_(3)
    tensor([ 3.,  3.,  3.])
    >>> x
    tensor([[ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.]])
    >>> torch.renorm(x, 1, 0, 5)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.6667,  1.6667,  1.6667],
            [ 1.6667,  1.6667,  1.6667]])
"""

    title = 'RenormNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.renorm())



"""
WARNING: Module Repeat_interleaveNode was generated using fallback option. May contain bugs
"""

class Repeat_interleaveNode(Node):
    """
repeat_interleave(input, repeats, dim=None) -> Tensor

Repeat elements of a tensor.

.. warning::

    This is different from :meth:`torch.Tensor.repeat` but similar to ``numpy.repeat``.

Args:
    input (Tensor): the input tensor.
    repeats (Tensor or int): The number of repetitions for each element.
        repeats is broadcasted to fit the shape of the given axis.
    dim (int, optional): The dimension along which to repeat values.
        By default, use the flattened input array, and return a flat output
        array.

Returns:
    Tensor: Repeated tensor which has the same shape as input, except along the given axis.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat_interleave(2)
    tensor([1, 1, 2, 2, 3, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.repeat_interleave(y, 2)
    tensor([1, 1, 2, 2, 3, 3, 4, 4])
    >>> torch.repeat_interleave(y, 3, dim=1)
    tensor([[1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4]])
    >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
    tensor([[1, 2],
            [3, 4],
            [3, 4]])

.. function:: repeat_interleave(repeats) -> Tensor

If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
`tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
`1` appears `n2` times, `2` appears `n3` times, etc.
"""

    title = 'Repeat_interleaveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.repeat_interleave())



"""
WARNING: Module ReshapeNode was generated using fallback option. May contain bugs
"""

class ReshapeNode(Node):
    """
reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

See :meth:`torch.Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped
    shape (tuple of ints): the new shape

Example::

    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])
"""

    title = 'ReshapeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.reshape())



"""
WARNING: Module Resize_as_Node was generated using fallback option. May contain bugs
"""

class Resize_as_Node(Node):
    """None"""

    title = 'Resize_as_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.resize_as_())



"""
WARNING: Module Resize_as_sparse_Node was generated using fallback option. May contain bugs
"""

class Resize_as_sparse_Node(Node):
    """None"""

    title = 'Resize_as_sparse_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.resize_as_sparse_())



"""
WARNING: Module Result_typeNode was generated using fallback option. May contain bugs
"""

class Result_typeNode(Node):
    """
result_type(tensor1, tensor2) -> dtype

Returns the :class:`torch.dtype` that would result from performing an arithmetic
operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
for more information on the type promotion logic.

Args:
    tensor1 (Tensor or Number): an input tensor or number
    tensor2 (Tensor or Number): an input tensor or number

Example::

    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.int), 1.0)
    torch.float32
    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor(1))
    torch.uint8
"""

    title = 'Result_typeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.result_type())



"""
WARNING: Module Rnn_reluNode was generated using fallback option. May contain bugs
"""

class Rnn_reluNode(Node):
    """None"""

    title = 'Rnn_reluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rnn_relu())



"""
WARNING: Module Rnn_relu_cellNode was generated using fallback option. May contain bugs
"""

class Rnn_relu_cellNode(Node):
    """None"""

    title = 'Rnn_relu_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rnn_relu_cell())



"""
WARNING: Module Rnn_tanhNode was generated using fallback option. May contain bugs
"""

class Rnn_tanhNode(Node):
    """None"""

    title = 'Rnn_tanhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rnn_tanh())



"""
WARNING: Module Rnn_tanh_cellNode was generated using fallback option. May contain bugs
"""

class Rnn_tanh_cellNode(Node):
    """None"""

    title = 'Rnn_tanh_cellNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rnn_tanh_cell())



"""
WARNING: Module RollNode was generated using fallback option. May contain bugs
"""

class RollNode(Node):
    """
roll(input, shifts, dims=None) -> Tensor

Roll the tensor along the given dimension(s). Elements that are shifted beyond the
last position are re-introduced at the first position. If a dimension is not
specified, the tensor will be flattened before rolling and then restored
to the original shape.

Args:
    input (Tensor): the input tensor.
    shifts (int or tuple of ints): The number of places by which the elements
        of the tensor are shifted. If shifts is a tuple, dims must be a tuple of
        the same size, and each dimension will be rolled by the corresponding
        value
    dims (int or tuple of ints): Axis along which to roll

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    >>> x
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    >>> torch.roll(x, 1, 0)
    tensor([[7, 8],
            [1, 2],
            [3, 4],
            [5, 6]])
    >>> torch.roll(x, -1, 0)
    tensor([[3, 4],
            [5, 6],
            [7, 8],
            [1, 2]])
    >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
    tensor([[6, 5],
            [8, 7],
            [2, 1],
            [4, 3]])
"""

    title = 'RollNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.roll())



"""
WARNING: Module Rot90Node was generated using fallback option. May contain bugs
"""

class Rot90Node(Node):
    """
rot90(input, k, dims) -> Tensor

Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0.

Args:
    input (Tensor): the input tensor.
    k (int): number of times to rotate
    dims (a list or tuple): axis to rotate

Example::

    >>> x = torch.arange(4).view(2, 2)
    >>> x
    tensor([[0, 1],
            [2, 3]])
    >>> torch.rot90(x, 1, [0, 1])
    tensor([[1, 3],
            [0, 2]])

    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[0, 1],
             [2, 3]],

            [[4, 5],
             [6, 7]]])
    >>> torch.rot90(x, 1, [1, 2])
    tensor([[[1, 3],
             [0, 2]],

            [[5, 7],
             [4, 6]]])
"""

    title = 'Rot90Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rot90())



"""
WARNING: Module RoundNode was generated using fallback option. May contain bugs
"""

class RoundNode(Node):
    """
round(input, *, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input` rounded
to the closest integer.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
    >>> torch.round(a)
    tensor([ 1.,  1.,  1., -1.])
"""

    title = 'RoundNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.round())



"""
WARNING: Module Round_Node was generated using fallback option. May contain bugs
"""

class Round_Node(Node):
    """None"""

    title = 'Round_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.round_())



"""
WARNING: Module Row_stackNode was generated using fallback option. May contain bugs
"""

class Row_stackNode(Node):
    """
row_stack(tensors, *, out=None) -> Tensor

Alias of :func:`torch.vstack`.
"""

    title = 'Row_stackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.row_stack())



"""
WARNING: Module RreluNode was generated using fallback option. May contain bugs
"""

class RreluNode(Node):
    """None"""

    title = 'RreluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rrelu())



"""
WARNING: Module Rrelu_Node was generated using fallback option. May contain bugs
"""

class Rrelu_Node(Node):
    """
rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor

In-place version of :func:`~rrelu`.
"""

    title = 'Rrelu_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rrelu_())



"""
WARNING: Module RsqrtNode was generated using fallback option. May contain bugs
"""

class RsqrtNode(Node):
    """
rsqrt(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0370,  0.2970,  1.5420, -0.9105])
    >>> torch.rsqrt(a)
    tensor([    nan,  1.8351,  0.8053,     nan])
"""

    title = 'RsqrtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rsqrt())



"""
WARNING: Module Rsqrt_Node was generated using fallback option. May contain bugs
"""

class Rsqrt_Node(Node):
    """None"""

    title = 'Rsqrt_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rsqrt_())



"""
WARNING: Module RsubNode was generated using fallback option. May contain bugs
"""

class RsubNode(Node):
    """None"""

    title = 'RsubNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.rsub())



"""
WARNING: Module SaddmmNode was generated using fallback option. May contain bugs
"""

class SaddmmNode(Node):
    """None"""

    title = 'SaddmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.saddmm())


class SaveNode(Node):
    """save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, 'tensor.pt')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """

    title = 'SaveNode'
    init_inputs = [
        NodeInputBP('obj'),
NodeInputBP('f'),
NodeInputBP('pickle_module'),
NodeInputBP('pickle_protocol'),
NodeInputBP('_use_new_zipfile_serialization'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.save(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4)))



"""
WARNING: Module Scalar_tensorNode was generated using fallback option. May contain bugs
"""

class Scalar_tensorNode(Node):
    """None"""

    title = 'Scalar_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.scalar_tensor())



"""
WARNING: Module ScatterNode was generated using fallback option. May contain bugs
"""

class ScatterNode(Node):
    """
scatter(input, dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_`
"""

    title = 'ScatterNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.scatter())



"""
WARNING: Module Scatter_addNode was generated using fallback option. May contain bugs
"""

class Scatter_addNode(Node):
    """
scatter_add(input, dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_add_`
"""

    title = 'Scatter_addNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.scatter_add())



"""
WARNING: Module SearchsortedNode was generated using fallback option. May contain bugs
"""

class SearchsortedNode(Node):
    """
searchsorted(sorted_sequence, values, *, out_int32=False, right=False, out=None) -> Tensor

Find the indices from the *innermost* dimension of :attr:`sorted_sequence` such that, if the
corresponding values in :attr:`values` were inserted before the indices, the order of the
corresponding *innermost* dimension within :attr:`sorted_sequence` would be preserved.
Return a new tensor with the same size as :attr:`values`. If :attr:`right` is False (default),
then the left boundary of :attr:`sorted_sequence` is closed. More formally, the returned index
satisfies the following rules:

.. list-table::
   :widths: 12 10 78
   :header-rows: 1

   * - :attr:`sorted_sequence`
     - :attr:`right`
     - *returned index satisfies*
   * - 1-D
     - False
     - ``sorted_sequence[i-1] < values[m][n]...[l][x] <= sorted_sequence[i]``
   * - 1-D
     - True
     - ``sorted_sequence[i-1] <= values[m][n]...[l][x] < sorted_sequence[i]``
   * - N-D
     - False
     - ``sorted_sequence[m][n]...[l][i-1] < values[m][n]...[l][x] <= sorted_sequence[m][n]...[l][i]``
   * - N-D
     - True
     - ``sorted_sequence[m][n]...[l][i-1] <= values[m][n]...[l][x] < sorted_sequence[m][n]...[l][i]``

Args:
    sorted_sequence (Tensor): N-D or 1-D tensor, containing monotonically increasing sequence on the *innermost*
                              dimension.
    values (Tensor or Scalar): N-D tensor or a Scalar containing the search value(s).

Keyword args:
    out_int32 (bool, optional): indicate the output data type. torch.int32 if True, torch.int64 otherwise.
                                Default value is False, i.e. default output data type is torch.int64.
    right (bool, optional): if False, return the first suitable location that is found. If True, return the
                            last such index. If no suitable index found, return 0 for non-numerical value
                            (eg. nan, inf) or the size of *innermost* dimension within :attr:`sorted_sequence`
                            (one pass the last index of the *innermost* dimension). In other words, if False,
                            gets the lower bound index for each value in :attr:`values` on the corresponding
                            *innermost* dimension of the :attr:`sorted_sequence`. If True, gets the upper
                            bound index instead. Default value is False.
    out (Tensor, optional): the output tensor, must be the same size as :attr:`values` if provided.

.. note:: If your use case is always 1-D sorted sequence, :func:`torch.bucketize` is preferred,
          because it has fewer dimension checks resulting in slightly better performance.


Example::

    >>> sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    >>> sorted_sequence
    tensor([[ 1,  3,  5,  7,  9],
            [ 2,  4,  6,  8, 10]])
    >>> values = torch.tensor([[3, 6, 9], [3, 6, 9]])
    >>> values
    tensor([[3, 6, 9],
            [3, 6, 9]])
    >>> torch.searchsorted(sorted_sequence, values)
    tensor([[1, 3, 4],
            [1, 2, 4]])
    >>> torch.searchsorted(sorted_sequence, values, right=True)
    tensor([[2, 3, 5],
            [1, 3, 4]])

    >>> sorted_sequence_1d = torch.tensor([1, 3, 5, 7, 9])
    >>> sorted_sequence_1d
    tensor([1, 3, 5, 7, 9])
    >>> torch.searchsorted(sorted_sequence_1d, values)
    tensor([[1, 3, 4],
            [1, 3, 4]])
"""

    title = 'SearchsortedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.searchsorted())


class SeedNode(Node):
    """Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.
    """

    title = 'SeedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.seed())



"""
WARNING: Module Segment_reduceNode was generated using fallback option. May contain bugs
"""

class Segment_reduceNode(Node):
    """None"""

    title = 'Segment_reduceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.segment_reduce())



"""
WARNING: Module SelectNode was generated using fallback option. May contain bugs
"""

class SelectNode(Node):
    """None"""

    title = 'SelectNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.select())



"""
WARNING: Module SeluNode was generated using fallback option. May contain bugs
"""

class SeluNode(Node):
    """None"""

    title = 'SeluNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.selu())



"""
WARNING: Module Selu_Node was generated using fallback option. May contain bugs
"""

class Selu_Node(Node):
    """
selu_(input) -> Tensor

In-place version of :func:`~selu`.
"""

    title = 'Selu_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.selu_())



"""
WARNING: Module SerializationNode was generated using fallback option. May contain bugs
"""

class SerializationNode(Node):
    """None"""

    title = 'SerializationNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.serialization())



"""
WARNING: Module Set_anomaly_enabledNode was generated using fallback option. May contain bugs
"""

class Set_anomaly_enabledNode(Node):
    """None"""

    title = 'Set_anomaly_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_anomaly_enabled())



"""
WARNING: Module Set_autocast_enabledNode was generated using fallback option. May contain bugs
"""

class Set_autocast_enabledNode(Node):
    """None"""

    title = 'Set_autocast_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_autocast_enabled())


class Set_default_dtypeNode(Node):
    """Sets the default floating point dtype to :attr:`d`.
    This dtype is:

    1. The inferred dtype for python floats in :func:`torch.tensor`.
    2. Used to infer dtype for python complex numbers. The default complex dtype is set to
       ``torch.complex128`` if default floating point dtype is ``torch.float64``,
       otherwise it's set to ``torch.complex64``

    The default floating point dtype is initially ``torch.float32``.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default

    Example:
        >>> # initial default for floating point is torch.float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64
        >>> torch.set_default_dtype(torch.float64)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64
        >>> torch.tensor([1.2, 3j]).dtype   # a new complex tensor
        torch.complex128

    """

    title = 'Set_default_dtypeNode'
    init_inputs = [
        NodeInputBP('d'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_default_dtype(self.input(0)))


class Set_default_tensor_typeNode(Node):
    """Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64

    """

    title = 'Set_default_tensor_typeNode'
    init_inputs = [
        NodeInputBP('t'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_default_tensor_type(self.input(0)))


class Set_deterministicNode(Node):
    """This function is deprecated and will be removed in a future release.
    Please use :func:`torch.use_deterministic_algorithms` instead.
    """

    title = 'Set_deterministicNode'
    init_inputs = [
        NodeInputBP('d'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_deterministic(self.input(0)))



"""
WARNING: Module Set_flush_denormalNode was generated using fallback option. May contain bugs
"""

class Set_flush_denormalNode(Node):
    """
set_flush_denormal(mode) -> bool

Disables denormal floating numbers on CPU.

Returns ``True`` if your system supports flushing denormal numbers and it
successfully configures flush denormal mode.  :meth:`~torch.set_flush_denormal`
is only supported on x86 architectures supporting SSE3.

Args:
    mode (bool): Controls whether to enable flush denormal mode or not

Example::

    >>> torch.set_flush_denormal(True)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor([ 0.], dtype=torch.float64)
    >>> torch.set_flush_denormal(False)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor(9.88131e-324 *
           [ 1.0000], dtype=torch.float64)
"""

    title = 'Set_flush_denormalNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_flush_denormal())


class Set_grad_enabledNode(Node):
    """Context-manager that sets gradient calculation to on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    title = 'Set_grad_enabledNode'
    init_inputs = [
        NodeInputBP('self'),
NodeInputBP('mode'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_grad_enabled(self.input(0), self.input(1)))



"""
WARNING: Module Set_num_interop_threadsNode was generated using fallback option. May contain bugs
"""

class Set_num_interop_threadsNode(Node):
    """
set_num_interop_threads(int)

Sets the number of threads used for interop parallelism
(e.g. in JIT interpreter) on CPU.

.. warning::
    Can only be called once and before any inter-op parallel work
    is started (e.g. JIT execution).
"""

    title = 'Set_num_interop_threadsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_num_interop_threads())



"""
WARNING: Module Set_num_threadsNode was generated using fallback option. May contain bugs
"""

class Set_num_threadsNode(Node):
    """
set_num_threads(int)

Sets the number of threads used for intraop parallelism on CPU.

.. warning::
    To ensure that the correct number of threads is used, set_num_threads
    must be called before running eager, JIT or autograd code.
"""

    title = 'Set_num_threadsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_num_threads())


class Set_printoptionsNode(Node):
    """Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.
    """

    title = 'Set_printoptionsNode'
    init_inputs = [
        NodeInputBP('precision'),
NodeInputBP('threshold'),
NodeInputBP('edgeitems'),
NodeInputBP('linewidth'),
NodeInputBP('profile'),
NodeInputBP('sci_mode'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_printoptions(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4), self.input(5)))


class Set_rng_stateNode(Node):
    """Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """

    title = 'Set_rng_stateNode'
    init_inputs = [
        NodeInputBP('new_state'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_rng_state(self.input(0)))



"""
WARNING: Module Set_vitalNode was generated using fallback option. May contain bugs
"""

class Set_vitalNode(Node):
    """set_vital(arg0: str, arg1: str, arg2: str) -> bool
"""

    title = 'Set_vitalNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
NodeInputBP('c'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_vital(self.input(0), self.input(1), self.input(2)))


class Set_warn_alwaysNode(Node):
    """When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """

    title = 'Set_warn_alwaysNode'
    init_inputs = [
        NodeInputBP('b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.set_warn_always(self.input(0)))



"""
WARNING: Module SgnNode was generated using fallback option. May contain bugs
"""

class SgnNode(Node):
    """
sgn(input, *, out=None) -> Tensor

This function is an extension of torch.sign() to complex tensors.
It computes a new tensor whose elements have
the same angles as the corresponding elements of :attr:`input` and
absolute values (i.e. magnitudes) of one for complex tensors and
is equivalent to torch.sign() for non-complex tensors.

.. math::
    \text{out}_{i} = \begin{cases}
                    0 & |\text{{input}}_i| == 0 \\
                    \frac{{\text{{input}}_i}}{|{\text{{input}}_i}|} & \text{otherwise}
                    \end{cases}


Args:
    input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.tensor([3+4j, 7-24j, 0, 1+2j])
    >>> t.sgn()
    tensor([0.6000+0.8000j, 0.2800-0.9600j, 0.0000+0.0000j, 0.4472+0.8944j])
"""

    title = 'SgnNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sgn())



"""
WARNING: Module ShortNode was generated using fallback option. May contain bugs
"""

class ShortNode(Node):
    """None"""

    title = 'ShortNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.short())



"""
WARNING: Module SigmoidNode was generated using fallback option. May contain bugs
"""

class SigmoidNode(Node):
    """
sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`.
"""

    title = 'SigmoidNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sigmoid())



"""
WARNING: Module Sigmoid_Node was generated using fallback option. May contain bugs
"""

class Sigmoid_Node(Node):
    """None"""

    title = 'Sigmoid_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sigmoid_())



"""
WARNING: Module SignNode was generated using fallback option. May contain bugs
"""

class SignNode(Node):
    """
sign(input, *, out=None) -> Tensor

Returns a new tensor with the signs of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> torch.sign(a)
    tensor([ 1., -1.,  0.,  1.])
"""

    title = 'SignNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sign())



"""
WARNING: Module SignbitNode was generated using fallback option. May contain bugs
"""

class SignbitNode(Node):
    """
signbit(input, *, out=None) -> Tensor

Tests if each element of :attr:`input` has its sign bit set (is less than zero) or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> torch.signbit(a)
    tensor([ False, True,  False,  False])
"""

    title = 'SignbitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.signbit())



"""
WARNING: Module SinNode was generated using fallback option. May contain bugs
"""

class SinNode(Node):
    """
sin(input, *, out=None) -> Tensor

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
"""

    title = 'SinNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sin())



"""
WARNING: Module Sin_Node was generated using fallback option. May contain bugs
"""

class Sin_Node(Node):
    """None"""

    title = 'Sin_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sin_())



"""
WARNING: Module SincNode was generated using fallback option. May contain bugs
"""

class SincNode(Node):
    """
sinc(input, *, out=None) -> Tensor

Computes the normalized sinc of :attr:`input.`

.. math::
    \text{out}_{i} =
    \begin{cases}
      1, & \text{if}\ \text{input}_{i}=0 \\
      \sin(\pi \text{input}_{i}) / (\pi \text{input}_{i}), & \text{otherwise}
    \end{cases}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2252, -0.2948,  1.0267, -1.1566])
    >>> torch.sinc(a)
    tensor([ 0.9186,  0.8631, -0.0259, -0.1300])
"""

    title = 'SincNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sinc())



"""
WARNING: Module Sinc_Node was generated using fallback option. May contain bugs
"""

class Sinc_Node(Node):
    """None"""

    title = 'Sinc_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sinc_())



"""
WARNING: Module SinhNode was generated using fallback option. May contain bugs
"""

class SinhNode(Node):
    """
sinh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic sine of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \sinh(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
    >>> torch.sinh(a)
    tensor([ 0.5644, -0.9744, -0.1268,  1.0845])

.. note::
   When :attr:`input` is on the CPU, the implementation of torch.sinh may use
   the Sleef library, which rounds very large results to infinity or negative
   infinity. See `here <https://sleef.org/purec.xhtml>`_ for details.
"""

    title = 'SinhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sinh())



"""
WARNING: Module Sinh_Node was generated using fallback option. May contain bugs
"""

class Sinh_Node(Node):
    """None"""

    title = 'Sinh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sinh_())



"""
WARNING: Module SlogdetNode was generated using fallback option. May contain bugs
"""

class SlogdetNode(Node):
    """
slogdet(input) -> (Tensor, Tensor)

Alias for :func:`torch.linalg.slogdet`
"""

    title = 'SlogdetNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.slogdet())



"""
WARNING: Module SmmNode was generated using fallback option. May contain bugs
"""

class SmmNode(Node):
    """
smm(input, mat) -> Tensor

Performs a matrix multiplication of the sparse matrix :attr:`input`
with the dense matrix :attr:`mat`.

Args:
    input (Tensor): a sparse matrix to be matrix multiplied
    mat (Tensor): a dense matrix to be matrix multiplied
"""

    title = 'SmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.smm())



"""
WARNING: Module SoftmaxNode was generated using fallback option. May contain bugs
"""

class SoftmaxNode(Node):
    """None"""

    title = 'SoftmaxNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.softmax())



"""
WARNING: Module SolveNode was generated using fallback option. May contain bugs
"""

class SolveNode(Node):
    """
torch.solve(input, A, *, out=None) -> (Tensor, Tensor)

This function returns the solution to the system of linear
equations represented by :math:`AX = B` and the LU factorization of
A, in order as a namedtuple `solution, LU`.

`LU` contains `L` and `U` factors for LU factorization of `A`.

`torch.solve(B, A)` can take in 2D inputs `B, A` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `solution, LU`.

Supports real-valued and complex-valued inputs.

.. warning::

    :func:`torch.solve` is deprecated in favor of :func:`torch.linalg.solve`
    and will be removed in a future PyTorch release.
    :func:`torch.linalg.solve` has its arguments reversed and does not return the
    LU factorization of the input. To get the LU factorization see :func:`torch.lu`,
    which may be used with :func:`torch.lu_solve` and :func:`torch.lu_unpack`.

    ``X = torch.solve(B, A).solution`` should be replaced with

    .. code:: python

        X = torch.linalg.solve(A, B)

.. note::

    Irrespective of the original strides, the returned matrices
    `solution` and `LU` will be transposed, i.e. with strides like
    `B.contiguous().transpose(-1, -2).stride()` and
    `A.contiguous().transpose(-1, -2).stride()` respectively.

Args:
    input (Tensor): input matrix :math:`B` of size :math:`(*, m, k)` , where :math:`*`
                is zero or more batch dimensions.
    A (Tensor): input square matrix of size :math:`(*, m, m)`, where
                :math:`*` is zero or more batch dimensions.

Keyword args:
    out ((Tensor, Tensor), optional): optional output tuple.

Example::

    >>> A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
    ...                   [-6.05, -3.30,  5.36, -4.44,  1.08],
    ...                   [-0.45,  2.58, -2.70,  0.27,  9.04],
    ...                   [8.32,  2.71,  4.35,  -7.17,  2.14],
    ...                   [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    >>> B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
    ...                   [-1.56,  4.00, -8.67,  1.75,  2.86],
    ...                   [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, torch.mm(A, X))
    tensor(1.00000e-06 *
           7.0977)

    >>> # Batched solver example
    >>> A = torch.randn(2, 3, 1, 4, 4)
    >>> B = torch.randn(2, 3, 1, 4, 6)
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, A.matmul(X))
    tensor(1.00000e-06 *
       3.6386)

"""

    title = 'SolveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.solve())



"""
WARNING: Module SortNode was generated using fallback option. May contain bugs
"""

class SortNode(Node):
    """
sort(input, dim=-1, descending=False, stable=False, *, out=None) -> (Tensor, LongTensor)

Sorts the elements of the :attr:`input` tensor along a given dimension
in ascending order by value.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`descending` is ``True`` then the elements are sorted in descending
order by value.

If :attr:`stable` is ``True`` then the sorting routine becomes stable, preserving
the order of equivalent elements.

A namedtuple of (values, indices) is returned, where the `values` are the
sorted values and `indices` are the indices of the elements in the original
`input` tensor.

.. warning:: `stable=True` only works on the CPU for now.

Args:
    input (Tensor): the input tensor.
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)
    stable (bool, optional): makes the sorting routine stable, which guarantees that the order
       of equivalent elements is preserved.

Keyword args:
    out (tuple, optional): the output tuple of (`Tensor`, `LongTensor`) that can
        be optionally given to be used as output buffers

Example::

    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted
    tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
            [-0.5793,  0.0061,  0.6058,  0.9497],
            [-0.5071,  0.3343,  0.9553,  1.0960]])
    >>> indices
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    >>> indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
    >>> x = torch.tensor([0, 1] * 9)
    >>> x.sort()
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 2, 16,  4,  6, 14,  8,  0, 10, 12,  9, 17, 15, 13, 11,  7,  5,  3,  1]))
    >>> x.sort(stable=True)
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16,  1,  3,  5,  7,  9, 11, 13, 15, 17]))
"""

    title = 'SortNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sort())



"""
WARNING: Module SparseNode was generated using fallback option. May contain bugs
"""

class SparseNode(Node):
    """None"""

    title = 'SparseNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sparse())



"""
WARNING: Module Sparse_cooNode was generated using fallback option. May contain bugs
"""

class Sparse_cooNode(Node):
    """None"""

    title = 'Sparse_cooNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sparse_coo())



"""
WARNING: Module Sparse_coo_tensorNode was generated using fallback option. May contain bugs
"""

class Sparse_coo_tensorNode(Node):
    """
sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False) -> Tensor

Constructs a :ref:`sparse tensor in COO(rdinate) format
<sparse-coo-docs>` with specified values at the given
:attr:`indices`.

.. note::

   This function returns an :ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.

Args:
    indices (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types. Will be cast to a :class:`torch.LongTensor`
        internally. The indices are the coordinates of the non-zero values in the matrix, and thus
        should be two-dimensional where the first dimension is the number of tensor dimensions and
        the second dimension is the number of non-zero values.
    values (array_like): Initial values for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    size (list, tuple, or :class:`torch.Size`, optional): Size of the sparse tensor. If not
        provided the size will be inferred as the minimum size big enough to hold all non-zero
        elements.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`values`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.


Example::

    >>> i = torch.tensor([[0, 1, 1],
    ...                   [2, 0, 2]])
    >>> v = torch.tensor([3, 4, 5], dtype=torch.float32)
    >>> torch.sparse_coo_tensor(i, v, [2, 4])
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 4), nnz=3, layout=torch.sparse_coo)

    >>> torch.sparse_coo_tensor(i, v)  # Shape inference
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)

    >>> torch.sparse_coo_tensor(i, v, [2, 4],
    ...                         dtype=torch.float64,
    ...                         device=torch.device('cuda:0'))
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           device='cuda:0', size=(2, 4), nnz=3, dtype=torch.float64,
           layout=torch.sparse_coo)

    # Create an empty sparse tensor with the following invariants:
    #   1. sparse_dim + dense_dim = len(SparseTensor.shape)
    #   2. SparseTensor._indices().shape = (sparse_dim, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
    #
    # For instance, to create an empty sparse tensor with nnz = 0, dense_dim = 0 and
    # sparse_dim = 1 (hence indices is a 2D tensor of shape = (1, 0))
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0,)),
           size=(1,), nnz=0, layout=torch.sparse_coo)

    # and to create an empty sparse tensor with nnz = 0, dense_dim = 1 and
    # sparse_dim = 1
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0, 2)),
           size=(1, 2), nnz=0, layout=torch.sparse_coo)

.. _torch.sparse: https://pytorch.org/docs/stable/sparse.html
"""

    title = 'Sparse_coo_tensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sparse_coo_tensor())



"""
WARNING: Module Sparse_csrNode was generated using fallback option. May contain bugs
"""

class Sparse_csrNode(Node):
    """None"""

    title = 'Sparse_csrNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sparse_csr())



"""
WARNING: Module SpecialNode was generated using fallback option. May contain bugs
"""

class SpecialNode(Node):
    """None"""

    title = 'SpecialNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.special())


class SplitNode(Node):
    """Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Args:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
            list of sizes for each chunk
        dim (int): dimension along which to split the tensor.

    Example::

        >>> a = torch.arange(10).reshape(5,2)
        >>> a
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]])
        >>> torch.split(a, 2)
        (tensor([[0, 1],
                 [2, 3]]),
         tensor([[4, 5],
                 [6, 7]]),
         tensor([[8, 9]]))
        >>> torch.split(a, [1,4])
        (tensor([[0, 1]]),
         tensor([[2, 3],
                 [4, 5],
                 [6, 7],
                 [8, 9]]))
    """

    title = 'SplitNode'
    init_inputs = [
        NodeInputBP('tensor'),
NodeInputBP('split_size_or_sections'),
NodeInputBP('dim'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.split(self.input(0), self.input(1), self.input(2)))



"""
WARNING: Module Split_with_sizesNode was generated using fallback option. May contain bugs
"""

class Split_with_sizesNode(Node):
    """None"""

    title = 'Split_with_sizesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.split_with_sizes())



"""
WARNING: Module SpmmNode was generated using fallback option. May contain bugs
"""

class SpmmNode(Node):
    """None"""

    title = 'SpmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.spmm())



"""
WARNING: Module SqrtNode was generated using fallback option. May contain bugs
"""

class SqrtNode(Node):
    """
sqrt(input, *, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.sqrt(a)
    tensor([    nan,  1.0112,  0.2883,  0.6933])
"""

    title = 'SqrtNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sqrt())



"""
WARNING: Module Sqrt_Node was generated using fallback option. May contain bugs
"""

class Sqrt_Node(Node):
    """None"""

    title = 'Sqrt_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sqrt_())



"""
WARNING: Module SquareNode was generated using fallback option. May contain bugs
"""

class SquareNode(Node):
    """
square(input, *, out=None) -> Tensor

Returns a new tensor with the square of the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.square(a)
    tensor([ 4.3077,  1.0457,  0.0069,  0.2310])
"""

    title = 'SquareNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.square())



"""
WARNING: Module Square_Node was generated using fallback option. May contain bugs
"""

class Square_Node(Node):
    """None"""

    title = 'Square_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.square_())



"""
WARNING: Module SqueezeNode was generated using fallback option. May contain bugs
"""

class SqueezeNode(Node):
    """
squeeze(input, dim=None, *, out=None) -> Tensor

Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

For example, if `input` is of shape:
:math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
will be of shape: :math:`(A \times B \times C \times D)`.

When :attr:`dim` is given, a squeeze operation is done only in the given
dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
will squeeze the tensor to the shape :math:`(A \times B)`.

.. note:: The returned tensor shares the storage with the input tensor,
          so changing the contents of one will change the contents of the other.

.. warning:: If the tensor has a batch dimension of size 1, then `squeeze(input)`
          will also remove the batch dimension, which can lead to unexpected
          errors.

Args:
    input (Tensor): the input tensor.
    dim (int, optional): if given, the input will be squeezed only in
           this dimension

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.zeros(2, 1, 2, 1, 2)
    >>> x.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x)
    >>> y.size()
    torch.Size([2, 2, 2])
    >>> y = torch.squeeze(x, 0)
    >>> y.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x, 1)
    >>> y.size()
    torch.Size([2, 2, 1, 2])
"""

    title = 'SqueezeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.squeeze())



"""
WARNING: Module SspaddmmNode was generated using fallback option. May contain bugs
"""

class SspaddmmNode(Node):
    """
sspaddmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

Matrix multiplies a sparse tensor :attr:`mat1` with a dense tensor
:attr:`mat2`, then adds the sparse tensor :attr:`input` to the result.

Note: This function is equivalent to :func:`torch.addmm`, except
:attr:`input` and :attr:`mat1` are sparse.

Args:
    input (Tensor): a sparse matrix to be added
    mat1 (Tensor): a sparse matrix to be matrix multiplied
    mat2 (Tensor): a dense matrix to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.
"""

    title = 'SspaddmmNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sspaddmm())



"""
WARNING: Module StackNode was generated using fallback option. May contain bugs
"""

class StackNode(Node):
    """
stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive)

Keyword args:
    out (Tensor, optional): the output tensor.
"""

    title = 'StackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.stack())



"""
WARNING: Module StdNode was generated using fallback option. May contain bugs
"""

class StdNode(Node):
    """
std(input, dim, unbiased, keepdim=False, *, out=None) -> Tensor

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.

Keyword args:
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
    out (Tensor, optional): the output tensor.


.. function:: std(input, unbiased) -> Tensor
   :noindex:

Calculates the standard deviation of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).

Example::

    >>> a = torch.tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.std(a, unbiased=False)
    tensor(0.4188)
"""

    title = 'StdNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.std())



"""
WARNING: Module Std_meanNode was generated using fallback option. May contain bugs
"""

class Std_meanNode(Node):
    """
std_mean(input, dim, unbiased, keepdim=False, *, out=None) -> (Tensor, Tensor)

If :attr:`unbiased` is ``True``, Bessel's correction will be used to calculate
the standard deviation. Otherwise, the sample deviation is calculated, without
any correction.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.

Keyword args:
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
    out (Tensor, optional): the output tensor.

Returns:
    A tuple (std, mean) containing the standard deviation and mean.

.. function:: std_mean(input, unbiased) -> (Tensor, Tensor)
   :noindex:

Calculates the standard deviation and mean of all elements in the :attr:`input`
tensor.

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).

Returns:
    A tuple (std, mean) containing the standard deviation and mean.

Example::

    >>> a = torch.tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.std_mean(a, unbiased=False)
    (tensor(0.4188), tensor(-0.8509))
"""

    title = 'Std_meanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.std_mean())


class StftNode(Node):
    """Short-time Fourier transform (STFT).

    .. warning::
        From version 1.8.0, :attr:`return_complex` must always be given
        explicitly for real inputs and `return_complex=False` has been
        deprecated. Strongly prefer `return_complex=True` as in a future
        pytorch release, this function will only return complex tensors.

        Note that :func:`torch.view_as_real` can be used to recover a real
        tensor with an extra last dimension for real and imaginary components.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This giving frequency components of the signal as they change over
    time. The interface of this function is modeled after the librosa_ stft function.

    .. _librosa: https://librosa.org/doc/latest/generated/librosa.stft.html

    Ignoring the optional batch dimension, this method computes the following
    expression:

    .. math::
        X[\omega, m] = \sum_{k = 0}^{\text{win\_length-1}}%
                            \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
                            \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\_length}}\right),

    where :math:`m` is the index of the sliding window, and :math:`\omega` is
    the frequency :math:`0 \leq \omega < \text{n\_fft}` for ``onesided=False``,
    or :math:`0 \leq \omega < \lfloor \text{n\_fft} / 2 \rfloor + 1` for ``onesided=True``.

    * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.

    * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
      ``floor(n_fft / 4)``.

    * If :attr:`win_length` is ``None`` (default), it is treated as equal to
      :attr:`n_fft`.

    * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
      :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
      treated as if having :math:`1` everywhere in the window. If
      :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
      both sides to length :attr:`n_fft` before being applied.

    * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
      both sides so that the :math:`t`-th frame is centered at time
      :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
      begins at time  :math:`t \times \text{hop\_length}`.

    * :attr:`pad_mode` determines the padding method used on :attr:`input` when
      :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
      all available options. Default is ``"reflect"``.

    * If :attr:`onesided` is ``True`` (default for real input), only values for
      :math:`\omega` in :math:`\left[0, 1, 2, \dots, \left\lfloor
      \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]` are returned because
      the real-to-complex Fourier transform satisfies the conjugate symmetry,
      i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.
      Note if the input or window tensors are complex, then :attr:`onesided`
      output is not possible.

    * If :attr:`normalized` is ``True`` (default is ``False``), the function
      returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.

    * If :attr:`return_complex` is ``True`` (default if input is complex), the
      return is a ``input.dim() + 1`` dimensional complex tensor. If ``False``,
      the output is a ``input.dim() + 2`` dimensional real tensor where the last
      dimension represents the real and imaginary components.

    Returns either a complex tensor of size :math:`(* \times N \times T)` if
    :attr:`return_complex` is true, or a real tensor of size :math:`(* \times N
    \times T \times 2)`. Where :math:`*` is the optional batch size of
    :attr:`input`, :math:`N` is the number of frequencies where STFT is applied
    and :math:`T` is the total number of frames used.

    .. warning::
      This function changed signature at version 0.4.1. Calling with the
      previous signature may cause error or return incorrect result.

    Args:
        input (Tensor): the input tensor
        n_fft (int): size of Fourier transform
        hop_length (int, optional): the distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`)
        window (Tensor, optional): the optional window function.
            Default: ``None`` (treated as window of all :math:`1` s)
        center (bool, optional): whether to pad :attr:`input` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.

    Returns:
        Tensor: A tensor containing the STFT result with shape described above

    """

    title = 'StftNode'
    init_inputs = [
        NodeInputBP('input'),
NodeInputBP('n_fft'),
NodeInputBP('hop_length'),
NodeInputBP('win_length'),
NodeInputBP('window'),
NodeInputBP('center'),
NodeInputBP('pad_mode'),
NodeInputBP('normalized'),
NodeInputBP('onesided'),
NodeInputBP('return_complex'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.stft(self.input(0), self.input(1), self.input(2), self.input(3), self.input(4), self.input(5), self.input(6), self.input(7), self.input(8), self.input(9)))



"""
WARNING: Module StorageNode was generated using fallback option. May contain bugs
"""

class StorageNode(Node):
    """None"""

    title = 'StorageNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.storage())



"""
WARNING: Module StridedNode was generated using fallback option. May contain bugs
"""

class StridedNode(Node):
    """None"""

    title = 'StridedNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.strided())



"""
WARNING: Module SubNode was generated using fallback option. May contain bugs
"""

class SubNode(Node):
    """
sub(input, other, *, alpha=1, out=None) -> Tensor

Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Scalar): the tensor or scalar to subtract from :attr:`input`

Keyword args:
    alpha (Scalar): the scalar multiplier for :attr:`other`
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor((1, 2))
    >>> b = torch.tensor((0, 1))
    >>> torch.sub(a, b, alpha=2)
    tensor([1, 0])
"""

    title = 'SubNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sub())



"""
WARNING: Module SubtractNode was generated using fallback option. May contain bugs
"""

class SubtractNode(Node):
    """
subtract(input, other, *, alpha=1, out=None) -> Tensor

Alias for :func:`torch.sub`.
"""

    title = 'SubtractNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.subtract())



"""
WARNING: Module SumNode was generated using fallback option. May contain bugs
"""

class SumNode(Node):
    """
sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
"""

    title = 'SumNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sum())



"""
WARNING: Module SvdNode was generated using fallback option. May contain bugs
"""

class SvdNode(Node):
    """
svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)

Computes the singular value decomposition of either a matrix or batch of
matrices :attr:`input`. The singular value decomposition is represented as a
namedtuple `(U, S, V)`, such that :attr:`input` `= U diag(S) Vᴴ`.
where `Vᴴ` is the transpose of `V` for real inputs,
and the conjugate transpose of `V` for complex inputs.
If :attr:`input` is a batch of matrices, then `U`, `S`, and `V` are also
batched with the same batch dimensions as :attr:`input`.

If :attr:`some` is `True` (default), the method returns the reduced singular
value decomposition. In this case, if the last two dimensions of :attr:`input` are
`m` and `n`, then the returned `U` and `V` matrices will contain only
`min(n, m)` orthonormal columns.

If :attr:`compute_uv` is `False`, the returned `U` and `V` will be
zero-filled matrices of shape `(m, m)` and `(n, n)`
respectively, and the same device as :attr:`input`. The argument :attr:`some`
has no effect when :attr:`compute_uv` is `False`.

Supports :attr:`input` of float, double, cfloat and cdouble data types.
The dtypes of `U` and `V` are the same as :attr:`input`'s. `S` will
always be real-valued, even if :attr:`input` is complex.

.. warning::

    :func:`torch.svd` is deprecated in favor of :func:`torch.linalg.svd`
    and will be removed in a future PyTorch release.

    ``U, S, V = torch.svd(A, some=some, compute_uv=True)`` (default) should be replaced with

    .. code:: python

        U, S, Vh = torch.linalg.svd(A, full_matrices=not some)
        V = Vh.transpose(-2, -1).conj()

    ``_, S, _ = torch.svd(A, some=some, compute_uv=False)`` should be replaced with

    .. code:: python

        S = torch.svdvals(A)

.. note:: Differences with :func:`torch.linalg.svd`:

             * :attr:`some` is the opposite of
               :func:`torch.linalg.svd`'s :attr:`full_matrices`. Note that
               default value for both is `True`, so the default behavior is
               effectively the opposite.
             * :func:`torch.svd` returns `V`, whereas :func:`torch.linalg.svd` returns
               `Vh`, that is, `Vᴴ`.
             * If :attr:`compute_uv` is `False`, :func:`torch.svd` returns zero-filled
               tensors for `U` and `Vh`, whereas :func:`torch.linalg.svd` returns
               empty tensors.

.. note:: The singular values are returned in descending order. If :attr:`input` is a batch of matrices,
          then the singular values of each matrix in the batch are returned in descending order.

.. note:: The `S` tensor can only be used to compute gradients if :attr:`compute_uv` is `True`.

.. note:: When :attr:`some` is `False`, the gradients on `U[..., :, min(m, n):]`
          and `V[..., :, min(m, n):]` will be ignored in the backward pass, as those vectors
          can be arbitrary bases of the corresponding subspaces.

.. note:: The implementation of :func:`torch.linalg.svd` on CPU uses LAPACK's routine `?gesdd`
          (a divide-and-conquer algorithm) instead of `?gesvd` for speed. Analogously,
          on GPU, it uses cuSOLVER's routines `gesvdj` and `gesvdjBatched` on CUDA 10.1.243
          and later, and MAGMA's routine `gesdd` on earlier versions of CUDA.

.. note:: The returned `U` will not be contiguous. The matrix (or batch of matrices) will
          be represented as a column-major matrix (i.e. Fortran-contiguous).

.. warning:: The gradients with respect to `U` and `V` will only be finite when the input does not
             have zero nor repeated singular values.

.. warning:: If the distance between any two singular values is close to zero, the gradients with respect to
             `U` and `V` will be numerically unstable, as they depends on
             :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`. The same happens when the matrix
             has small singular values, as these gradients also depend on `S⁻¹`.

.. warning:: For complex-valued :attr:`input` the singular value decomposition is not unique,
             as `U` and `V` may be multiplied by an arbitrary phase factor :math:`e^{i \phi}` on every column.
             The same happens when :attr:`input` has repeated singular values, where one may multiply
             the columns of the spanning subspace in `U` and `V` by a rotation matrix
             and `the resulting vectors will span the same subspace`_.
             Different platforms, like NumPy, or inputs on different device types,
             may produce different `U` and `V` tensors.

Args:
    input (Tensor): the input tensor of size `(*, m, n)` where `*` is zero or more
                    batch dimensions consisting of `(m, n)` matrices.
    some (bool, optional): controls whether to compute the reduced or full decomposition, and
                           consequently, the shape of returned `U` and `V`. Default: `True`.
    compute_uv (bool, optional): controls whether to compute `U` and `V`. Default: `True`.

Keyword args:
    out (tuple, optional): the output tuple of tensors

Example::

    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[ 0.2364, -0.7752,  0.6372],
            [ 1.7201,  0.7394, -0.0504],
            [-0.3371, -1.0584,  0.5296],
            [ 0.3550, -0.4022,  1.5569],
            [ 0.2445, -0.0158,  1.1414]])
    >>> u, s, v = torch.svd(a)
    >>> u
    tensor([[ 0.4027,  0.0287,  0.5434],
            [-0.1946,  0.8833,  0.3679],
            [ 0.4296, -0.2890,  0.5261],
            [ 0.6604,  0.2717, -0.2618],
            [ 0.4234,  0.2481, -0.4733]])
    >>> s
    tensor([2.3289, 2.0315, 0.7806])
    >>> v
    tensor([[-0.0199,  0.8766,  0.4809],
            [-0.5080,  0.4054, -0.7600],
            [ 0.8611,  0.2594, -0.4373]])
    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    tensor(8.6531e-07)
    >>> a_big = torch.randn(7, 5, 3)
    >>> u, s, v = torch.svd(a_big)
    >>> torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))
    tensor(2.6503e-06)

.. _the resulting vectors will span the same subspace:
       (https://en.wikipedia.org/wiki/Singular_value_decomposition#Singular_values,_singular_vectors,_and_their_relation_to_the_SVD)
"""

    title = 'SvdNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.svd())


class Svd_lowrankNode(Node):
    """Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U diag(S) V^T`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    .. note:: The input is assumed to be a low-rank matrix.

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """

    title = 'Svd_lowrankNode'
    init_inputs = [
        NodeInputBP('A'),
NodeInputBP('q'),
NodeInputBP('niter'),
NodeInputBP('M'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.svd_lowrank(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module SwapaxesNode was generated using fallback option. May contain bugs
"""

class SwapaxesNode(Node):
    """
swapaxes(input, axis0, axis1) -> Tensor

Alias for :func:`torch.transpose`.

This function is equivalent to NumPy's swapaxes function.

Examples::

    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    tensor([[[0, 1],
            [2, 3]],

            [[4, 5],
            [6, 7]]])
    >>> torch.swapaxes(x, 0, 1)
    tensor([[[0, 1],
            [4, 5]],

            [[2, 3],
            [6, 7]]])
    >>> torch.swapaxes(x, 0, 2)
    tensor([[[0, 4],
            [2, 6]],

            [[1, 5],
            [3, 7]]])
"""

    title = 'SwapaxesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.swapaxes())



"""
WARNING: Module SwapdimsNode was generated using fallback option. May contain bugs
"""

class SwapdimsNode(Node):
    """
swapdims(input, dim0, dim1) -> Tensor

Alias for :func:`torch.transpose`.

This function is equivalent to NumPy's swapaxes function.

Examples::

    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    tensor([[[0, 1],
            [2, 3]],

            [[4, 5],
            [6, 7]]])
    >>> torch.swapdims(x, 0, 1)
    tensor([[[0, 1],
            [4, 5]],

            [[2, 3],
            [6, 7]]])
    >>> torch.swapdims(x, 0, 2)
    tensor([[[0, 4],
            [2, 6]],

            [[1, 5],
            [3, 7]]])
"""

    title = 'SwapdimsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.swapdims())



"""
WARNING: Module SymeigNode was generated using fallback option. May contain bugs
"""

class SymeigNode(Node):
    """
symeig(input, eigenvectors=False, upper=True, *, out=None) -> (Tensor, Tensor)

This function returns eigenvalues and eigenvectors
of a real symmetric or complex Hermitian matrix :attr:`input` or a batch thereof,
represented by a namedtuple (eigenvalues, eigenvectors).

This function calculates all eigenvalues (and vectors) of :attr:`input`
such that :math:`\text{input} = V \text{diag}(e) V^T`.

The boolean argument :attr:`eigenvectors` defines computation of
both eigenvectors and eigenvalues or eigenvalues only.

If it is ``False``, only eigenvalues are computed. If it is ``True``,
both eigenvalues and eigenvectors are computed.

Since the input matrix :attr:`input` is supposed to be symmetric or Hermitian,
only the upper triangular portion is used by default.

If :attr:`upper` is ``False``, then lower triangular portion is used.

.. warning::

    :func:`torch.symeig` is deprecated in favor of :func:`torch.linalg.eigh`
    and will be removed in a future PyTorch release. The default behavior has changed
    from using the upper triangular portion of the matrix by default to using the
    lower triangular portion.

    ``L, _ = torch.symeig(A, upper=upper)`` should be replaced with

    .. code :: python

        UPLO = "U" if upper else "L"
        L = torch.linalg.eigvalsh(A, UPLO=UPLO)

    ``L, V = torch.symeig(A, eigenvectors=True, upper=upper)`` should be replaced with

    .. code :: python

        UPLO = "U" if upper else "L"
        L, V = torch.linalg.eigh(A, UPLO=UPLO)

.. note:: The eigenvalues are returned in ascending order. If :attr:`input` is a batch of matrices,
          then the eigenvalues of each matrix in the batch is returned in ascending order.

.. note:: Irrespective of the original strides, the returned matrix `V` will
          be transposed, i.e. with strides `V.contiguous().transpose(-1, -2).stride()`.

.. warning:: Extra care needs to be taken when backward through outputs. Such
             operation is only stable when all eigenvalues are distinct and becomes
             less stable the smaller :math:`\min_{i \neq j} |\lambda_i - \lambda_j|` is.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                    batch dimensions consisting of symmetric or Hermitian matrices.
    eigenvectors(bool, optional): controls whether eigenvectors have to be computed
    upper(boolean, optional): controls whether to consider upper-triangular or lower-triangular region

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, Tensor)

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(*, m)`. The eigenvalues in ascending order.
        - **eigenvectors** (*Tensor*): Shape :math:`(*, m, m)`.
          If ``eigenvectors=False``, it's an empty tensor.
          Otherwise, this tensor contains the orthonormal eigenvectors of the ``input``.

Examples::


    >>> a = torch.randn(5, 5)
    >>> a = a + a.t()  # To make a symmetric
    >>> a
    tensor([[-5.7827,  4.4559, -0.2344, -1.7123, -1.8330],
            [ 4.4559,  1.4250, -2.8636, -3.2100, -0.1798],
            [-0.2344, -2.8636,  1.7112, -5.5785,  7.1988],
            [-1.7123, -3.2100, -5.5785, -2.6227,  3.1036],
            [-1.8330, -0.1798,  7.1988,  3.1036, -5.1453]])
    >>> e, v = torch.symeig(a, eigenvectors=True)
    >>> e
    tensor([-13.7012,  -7.7497,  -2.3163,   5.2477,   8.1050])
    >>> v
    tensor([[ 0.1643,  0.9034, -0.0291,  0.3508,  0.1817],
            [-0.2417, -0.3071, -0.5081,  0.6534,  0.4026],
            [-0.5176,  0.1223, -0.0220,  0.3295, -0.7798],
            [-0.4850,  0.2695, -0.5773, -0.5840,  0.1337],
            [ 0.6415, -0.0447, -0.6381, -0.0193, -0.4230]])
    >>> a_big = torch.randn(5, 2, 2)
    >>> a_big = a_big + a_big.transpose(-2, -1)  # To make a_big symmetric
    >>> e, v = a_big.symeig(eigenvectors=True)
    >>> torch.allclose(torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1))), a_big)
    True
"""

    title = 'SymeigNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.symeig())



"""
WARNING: Module SysNode was generated using fallback option. May contain bugs
"""

class SysNode(Node):
    """This module provides access to some objects used or maintained by the
interpreter and to functions that interact strongly with the interpreter.

Dynamic objects:

argv -- command line arguments; argv[0] is the script pathname if known
path -- module search path; path[0] is the script directory, else ''
modules -- dictionary of loaded modules

displayhook -- called to show results in an interactive session
excepthook -- called to handle any uncaught exception other than SystemExit
  To customize printing in an interactive session or to install a custom
  top-level exception handler, assign other functions to replace these.

stdin -- standard input file object; used by input()
stdout -- standard output file object; used by print()
stderr -- standard error object; used for error messages
  By assigning other file objects (or objects that behave like files)
  to these, it is possible to redirect all of the interpreter's I/O.

last_type -- type of last uncaught exception
last_value -- value of last uncaught exception
last_traceback -- traceback of last uncaught exception
  These three are only available in an interactive session after a
  traceback has been printed.

Static objects:

builtin_module_names -- tuple of module names built into this interpreter
copyright -- copyright notice pertaining to this interpreter
exec_prefix -- prefix used to find the machine-specific Python library
executable -- absolute path of the executable binary of the Python interpreter
float_info -- a named tuple with information about the float implementation.
float_repr_style -- string indicating the style of repr() output for floats
hash_info -- a named tuple with information about the hash algorithm.
hexversion -- version information encoded as a single integer
implementation -- Python implementation information.
int_info -- a named tuple with information about the int implementation.
maxsize -- the largest supported length of containers.
maxunicode -- the value of the largest Unicode code point
platform -- platform identifier
prefix -- prefix used to find the Python library
thread_info -- a named tuple with information about the thread implementation.
version -- the version of this interpreter as a string
version_info -- version information as a named tuple
__stdin__ -- the original stdin; don't touch!
__stdout__ -- the original stdout; don't touch!
__stderr__ -- the original stderr; don't touch!
__displayhook__ -- the original displayhook; don't touch!
__excepthook__ -- the original excepthook; don't touch!

Functions:

displayhook() -- print an object to the screen, and save it in builtins._
excepthook() -- print an exception and its traceback to sys.stderr
exc_info() -- return thread-safe information about the current exception
exit() -- exit the interpreter by raising SystemExit
getdlopenflags() -- returns flags to be used for dlopen() calls
getprofile() -- get the global profiling function
getrefcount() -- return the reference count for an object (plus one :-)
getrecursionlimit() -- return the max recursion depth for the interpreter
getsizeof() -- return the size of an object in bytes
gettrace() -- get the global debug tracing function
setdlopenflags() -- set the flags to be used for dlopen() calls
setprofile() -- set the global profiling function
setrecursionlimit() -- set the max recursion depth for the interpreter
settrace() -- set the global debug tracing function
"""

    title = 'SysNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.sys(self.input(0)))



"""
WARNING: Module TNode was generated using fallback option. May contain bugs
"""

class TNode(Node):
    """
t(input) -> Tensor

Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
and 1.

0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
is equivalent to ``transpose(input, 0, 1)``.

Args:
    input (Tensor): the input tensor.

Example::

    >>> x = torch.randn(())
    >>> x
    tensor(0.1995)
    >>> torch.t(x)
    tensor(0.1995)
    >>> x = torch.randn(3)
    >>> x
    tensor([ 2.4320, -0.4608,  0.7702])
    >>> torch.t(x)
    tensor([ 2.4320, -0.4608,  0.7702])
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.4875,  0.9158, -0.5872],
            [ 0.3938, -0.6929,  0.6932]])
    >>> torch.t(x)
    tensor([[ 0.4875,  0.3938],
            [ 0.9158, -0.6929],
            [-0.5872,  0.6932]])
"""

    title = 'TNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.t())



"""
WARNING: Module TakeNode was generated using fallback option. May contain bugs
"""

class TakeNode(Node):
    """
take(input, index) -> Tensor

Returns a new tensor with the elements of :attr:`input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The result
takes the same shape as the indices.

Args:
    input (Tensor): the input tensor.
    index (LongTensor): the indices into tensor

Example::

    >>> src = torch.tensor([[4, 3, 5],
    ...                     [6, 7, 8]])
    >>> torch.take(src, torch.tensor([0, 2, 5]))
    tensor([ 4,  5,  8])
"""

    title = 'TakeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.take())



"""
WARNING: Module Take_along_dimNode was generated using fallback option. May contain bugs
"""

class Take_along_dimNode(Node):
    """
take_along_dim(input, indices, dim, *, out=None) -> Tensor

Selects values from :attr:`input` at the 1-dimensional indices from :attr:`indices` along the given :attr:`dim`.

Functions that return indices along a dimension, like :func:`torch.argmax` and :func:`torch.argsort`,
are designed to work with this function. See the examples below.

.. note::
    This function is similar to NumPy's `take_along_axis`.
    See also :func:`torch.gather`.

Args:
    input (Tensor): the input tensor.
    indices (tensor): the indices into :attr:`input`. Must have long dtype.
    dim (int): dimension to select along.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> t = torch.tensor([[10, 30, 20], [60, 40, 50]])
    >>> max_idx = torch.argmax(t)
    >>> torch.take_along_dim(t, max_idx)
    tensor([60])
    >>> sorted_idx = torch.argsort(t, dim=1)
    >>> torch.take_along_dim(t, sorted_idx, dim=1)
    tensor([[10, 20, 30],
            [40, 50, 60]])
"""

    title = 'Take_along_dimNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.take_along_dim())



"""
WARNING: Module TanNode was generated using fallback option. May contain bugs
"""

class TanNode(Node):
    """
tan(input, *, out=None) -> Tensor

Returns a new tensor with the tangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.2027, -1.7687,  0.4412, -1.3856])
    >>> torch.tan(a)
    tensor([-2.5930,  4.9859,  0.4722, -5.3366])
"""

    title = 'TanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tan())



"""
WARNING: Module Tan_Node was generated using fallback option. May contain bugs
"""

class Tan_Node(Node):
    """None"""

    title = 'Tan_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tan_())



"""
WARNING: Module TanhNode was generated using fallback option. May contain bugs
"""

class TanhNode(Node):
    """
tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
"""

    title = 'TanhNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tanh())



"""
WARNING: Module Tanh_Node was generated using fallback option. May contain bugs
"""

class Tanh_Node(Node):
    """None"""

    title = 'Tanh_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tanh_())



"""
WARNING: Module TensorNode was generated using fallback option. May contain bugs
"""

class TensorNode(Node):
    """
tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with :attr:`data`.

.. warning::

    :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a NumPy ``ndarray`` and want to avoid a copy, use
    :func:`torch.as_tensor`.

.. warning::

    When data is a tensor `x`, :func:`torch.tensor` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``torch.tensor(x)`` is equivalent to ``x.clone().detach()``
    and ``torch.tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
    The equivalents using ``clone()`` and ``detach()`` are recommended.

Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.


Example::

    >>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[ 0.1000,  1.2000],
            [ 2.2000,  3.1000],
            [ 4.9000,  5.2000]])

    >>> torch.tensor([0, 1])  # Type inference on data
    tensor([ 0,  1])

    >>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
    ...              dtype=torch.float64,
    ...              device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
    tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

    >>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
    tensor(3.1416)

    >>> torch.tensor([])  # Create an empty tensor (of size (0,))
    tensor([])
"""

    title = 'TensorNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tensor())



"""
WARNING: Module Tensor_splitNode was generated using fallback option. May contain bugs
"""

class Tensor_splitNode(Node):
    """
tensor_split(input, indices_or_sections, dim=0) -> List of Tensors

Splits a tensor into multiple sub-tensors, all of which are views of :attr:`input`,
along dimension :attr:`dim` according to the indices or number of sections specified
by :attr:`indices_or_sections`. This function is based on NumPy's
:func:`numpy.array_split`.

Args:
    input (Tensor): the tensor to split
    indices_or_sections (Tensor, int or list or tuple of ints):
        If :attr:`indices_or_sections` is an integer ``n`` or a zero dimensional long tensor
        with value ``n``, :attr:`input` is split into ``n`` sections along dimension :attr:`dim`.
        If :attr:`input` is divisible by ``n`` along dimension :attr:`dim`, each
        section will be of equal size, :code:`input.size(dim) / n`. If :attr:`input`
        is not divisible by ``n``, the sizes of the first :code:`int(input.size(dim) % n)`
        sections will have size :code:`int(input.size(dim) / n) + 1`, and the rest will
        have size :code:`int(input.size(dim) / n)`.

        If :attr:`indices_or_sections` is a list or tuple of ints, or a one-dimensional long
        tensor, then :attr:`input` is split along dimension :attr:`dim` at each of the indices
        in the list, tuple or tensor. For instance, :code:`indices_or_sections=[2, 3]` and :code:`dim=0`
        would result in the tensors :code:`input[:2]`, :code:`input[2:3]`, and :code:`input[3:]`.

        If indices_or_sections is a tensor, it must be a zero-dimensional or one-dimensional
        long tensor on the CPU.

    dim (int, optional): dimension along which to split the tensor. Default: ``0``

Example::

    >>> x = torch.arange(8)
    >>> torch.tensor_split(x, 3)
    (tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7]))

    >>> x = torch.arange(7)
    >>> torch.tensor_split(x, 3)
    (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    >>> torch.tensor_split(x, (1, 6))
    (tensor([0]), tensor([1, 2, 3, 4, 5]), tensor([6]))

    >>> x = torch.arange(14).reshape(2, 7)
    >>> x
    tensor([[ 0,  1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12, 13]])
    >>> torch.tensor_split(x, 3, dim=1)
    (tensor([[0, 1, 2],
            [7, 8, 9]]),
     tensor([[ 3,  4],
            [10, 11]]),
     tensor([[ 5,  6],
            [12, 13]]))
    >>> torch.tensor_split(x, (1, 6), dim=1)
    (tensor([[0],
            [7]]),
     tensor([[ 1,  2,  3,  4,  5],
            [ 8,  9, 10, 11, 12]]),
     tensor([[ 6],
            [13]]))
"""

    title = 'Tensor_splitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tensor_split())


class TensordotNode(Node):
    """Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`~torch.tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :func:`~torch.tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

        >>> a = torch.randn(3, 5, 4, 6)
        >>> b = torch.randn(6, 4, 5, 3)
        >>> torch.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
        tensor([[  7.7193,  -2.4867, -10.3204],
                [  1.5513, -14.4737,  -6.5113],
                [ -0.2850,   4.2573,  -3.5997]])
    """

    title = 'TensordotNode'
    init_inputs = [
        NodeInputBP('a'),
NodeInputBP('b'),
NodeInputBP('dims'),
NodeInputBP('out'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tensordot(self.input(0), self.input(1), self.input(2), self.input(3)))



"""
WARNING: Module TestingNode was generated using fallback option. May contain bugs
"""

class TestingNode(Node):
    """None"""

    title = 'TestingNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.testing())



"""
WARNING: Module TextwrapNode was generated using fallback option. May contain bugs
"""

class TextwrapNode(Node):
    """Text wrapping and filling.
"""

    title = 'TextwrapNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.textwrap(self.input(0)))



"""
WARNING: Module ThresholdNode was generated using fallback option. May contain bugs
"""

class ThresholdNode(Node):
    """None"""

    title = 'ThresholdNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.threshold())



"""
WARNING: Module Threshold_Node was generated using fallback option. May contain bugs
"""

class Threshold_Node(Node):
    """
threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`.
"""

    title = 'Threshold_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.threshold_())



"""
WARNING: Module TileNode was generated using fallback option. May contain bugs
"""

class TileNode(Node):
    """
tile(input, reps) -> Tensor

Constructs a tensor by repeating the elements of :attr:`input`.
The :attr:`reps` argument specifies the number of repetitions
in each dimension.

If :attr:`reps` specifies fewer dimensions than :attr:`input` has, then
ones are prepended to :attr:`reps` until all dimensions are specified.
For example, if :attr:`input` has shape (8, 6, 4, 2) and :attr:`reps`
is (2, 2), then :attr:`reps` is treated as (1, 1, 2, 2).

Analogously, if :attr:`input` has fewer dimensions than :attr:`reps`
specifies, then :attr:`input` is treated as if it were unsqueezed at
dimension zero until it has as many dimensions as :attr:`reps` specifies.
For example, if :attr:`input` has shape (4, 2) and :attr:`reps`
is (3, 3, 2, 2), then :attr:`input` is treated as if it had the
shape (1, 1, 4, 2).

.. note::

    This function is similar to NumPy's tile function.

Args:
    input (Tensor): the tensor whose elements to repeat.
    reps (tuple): the number of repetitions per dimension.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> x.tile((2,))
    tensor([1, 2, 3, 1, 2, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.tile(y, (2, 2))
    tensor([[1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]])
"""

    title = 'TileNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tile())



"""
WARNING: Module TopkNode was generated using fallback option. May contain bugs
"""

class TopkNode(Node):
    """
topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
of the elements in the original `input` tensor.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

Args:
    input (Tensor): the input tensor.
    k (int): the k in "top-k"
    dim (int, optional): the dimension to sort along
    largest (bool, optional): controls whether to return largest or
           smallest elements
    sorted (bool, optional): controls whether to return the elements
           in sorted order

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, LongTensor) that can be
        optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.topk(x, 3)
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
"""

    title = 'TopkNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.topk())



"""
WARNING: Module TorchNode was generated using fallback option. May contain bugs
"""

class TorchNode(Node):
    """
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serializing of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

    title = 'TorchNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.torch())



"""
WARNING: Module TraceNode was generated using fallback option. May contain bugs
"""

class TraceNode(Node):
    """
trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.]])
    >>> torch.trace(x)
    tensor(15.)
"""

    title = 'TraceNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.trace())



"""
WARNING: Module TransposeNode was generated using fallback option. May contain bugs
"""

class TransposeNode(Node):
    """
transpose(input, dim0, dim1) -> Tensor

Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

The resulting :attr:`out` tensor shares its underlying storage with the
:attr:`input` tensor, so changing the content of one would change the content
of the other.

Args:
    input (Tensor): the input tensor.
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
"""

    title = 'TransposeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.transpose())



"""
WARNING: Module TrapzNode was generated using fallback option. May contain bugs
"""

class TrapzNode(Node):
    """
trapz(y, x, *, dim=-1) -> Tensor

Estimate :math:`\int y\,dx` along `dim`, using the trapezoid rule.

Arguments:
    y (Tensor): The values of the function to integrate
    x (Tensor): The points at which the function `y` is sampled.
        If `x` is not in ascending order, intervals on which it is decreasing
        contribute negatively to the estimated integral (i.e., the convention
        :math:`\int_a^b f = -\int_b^a f` is followed).
    dim (int): The dimension along which to integrate.
        By default, use the last dimension.

Returns:
    A Tensor with the same shape as the input, except with `dim` removed.
    Each element of the returned tensor represents the estimated integral
    :math:`\int y\,dx` along `dim`.

Example::

    >>> y = torch.randn((2, 3))
    >>> y
    tensor([[-2.1156,  0.6857, -0.2700],
            [-1.2145,  0.5540,  2.0431]])
    >>> x = torch.tensor([[1, 3, 4], [1, 2, 3]])
    >>> torch.trapz(y, x)
    tensor([-1.2220,  0.9683])

.. function:: trapz(y, *, dx=1, dim=-1) -> Tensor

As above, but the sample points are spaced uniformly at a distance of `dx`.

Arguments:
    y (Tensor): The values of the function to integrate

Keyword args:
    dx (float): The distance between points at which `y` is sampled.
    dim (int): The dimension along which to integrate.
        By default, use the last dimension.

Returns:
    A Tensor with the same shape as the input, except with `dim` removed.
    Each element of the returned tensor represents the estimated integral
    :math:`\int y\,dx` along `dim`.
"""

    title = 'TrapzNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.trapz())



"""
WARNING: Module Triangular_solveNode was generated using fallback option. May contain bugs
"""

class Triangular_solveNode(Node):
    """
triangular_solve(b, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

Solves a system of equations with a triangular coefficient matrix :math:`A`
and multiple right-hand sides :math:`b`.

In particular, solves :math:`AX = b` and assumes :math:`A` is upper-triangular
with the default keyword arguments.

`torch.triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `X`

Supports input of float, double, cfloat and cdouble data types.

Args:
    b (Tensor): multiple right-hand sides of size :math:`(*, m, k)` where
                :math:`*` is zero of more batch dimensions
    A (Tensor): the input triangular coefficient matrix of size :math:`(*, m, m)`
                where :math:`*` is zero or more batch dimensions
    upper (bool, optional): whether to solve the upper-triangular system
        of equations (default) or the lower-triangular system of equations. Default: ``True``.
    transpose (bool, optional): whether :math:`A` should be transposed before
        being sent into the solver. Default: ``False``.
    unitriangular (bool, optional): whether :math:`A` is unit triangular.
        If True, the diagonal elements of :math:`A` are assumed to be
        1 and not referenced from :math:`A`. Default: ``False``.

Returns:
    A namedtuple `(solution, cloned_coefficient)` where `cloned_coefficient`
    is a clone of :math:`A` and `solution` is the solution :math:`X` to :math:`AX = b`
    (or whatever variant of the system of equations, depending on the keyword arguments.)

Examples::

    >>> A = torch.randn(2, 2).triu()
    >>> A
    tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]])
    >>> b = torch.randn(2, 3)
    >>> b
    tensor([[-0.0210,  2.3513, -1.5492],
            [ 1.5429,  0.7403, -1.0243]])
    >>> torch.triangular_solve(b, A)
    torch.return_types.triangular_solve(
    solution=tensor([[ 1.7841,  2.9046, -2.5405],
            [ 1.9320,  0.9270, -1.2826]]),
    cloned_coefficient=tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]]))
"""

    title = 'Triangular_solveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.triangular_solve())



"""
WARNING: Module TrilNode was generated using fallback option. May contain bugs
"""

class TrilNode(Node):
    """
tril(input, diagonal=0, *, out=None) -> Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.

Args:
    input (Tensor): the input tensor.
    diagonal (int, optional): the diagonal to consider

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0813, -0.8619,  0.7105],
            [ 0.0935,  0.1380,  2.2112],
            [-0.3409, -0.9828,  0.0289]])
    >>> torch.tril(a)
    tensor([[-1.0813,  0.0000,  0.0000],
            [ 0.0935,  0.1380,  0.0000],
            [-0.3409, -0.9828,  0.0289]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
            [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
    >>> torch.tril(b, diagonal=1)
    tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
    >>> torch.tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
"""

    title = 'TrilNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tril())



"""
WARNING: Module Tril_indicesNode was generated using fallback option. May contain bugs
"""

class Tril_indicesNode(Node):
    """
tril_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the lower triangular part of a :attr:`row`-by-
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

.. note::
    When running on CUDA, ``row * col`` must be less than :math:`2^{59}` to
    prevent overflow during calculation.

Args:
    row (``int``): number of rows in the 2-D matrix.
    col (``int``): number of columns in the 2-D matrix.
    offset (``int``): diagonal offset from the main diagonal.
        Default: if not provided, 0.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, ``torch.long``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    layout (:class:`torch.layout`, optional): currently only support ``torch.strided``.

Example::

    >>> a = torch.tril_indices(3, 3)
    >>> a
    tensor([[0, 1, 1, 2, 2, 2],
            [0, 0, 1, 0, 1, 2]])

    >>> a = torch.tril_indices(4, 3, -1)
    >>> a
    tensor([[1, 2, 2, 3, 3, 3],
            [0, 0, 1, 0, 1, 2]])

    >>> a = torch.tril_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
"""

    title = 'Tril_indicesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.tril_indices())



"""
WARNING: Module Triplet_margin_lossNode was generated using fallback option. May contain bugs
"""

class Triplet_margin_lossNode(Node):
    """None"""

    title = 'Triplet_margin_lossNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.triplet_margin_loss())



"""
WARNING: Module TriuNode was generated using fallback option. May contain bugs
"""

class TriuNode(Node):
    """
triu(input, diagonal=0, *, out=None) -> Tensor

Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.

Args:
    input (Tensor): the input tensor.
    diagonal (int, optional): the diagonal to consider

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.0000, -1.0680,  0.6602],
            [ 0.0000,  0.0000, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.0000, -0.5211, -0.4573]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=1)
    tensor([[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=-1)
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000, -1.3337, -1.6556,  0.4798,  0.2830]])
"""

    title = 'TriuNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.triu())



"""
WARNING: Module Triu_indicesNode was generated using fallback option. May contain bugs
"""

class Triu_indicesNode(Node):
    """
triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the upper triangular part of a :attr:`row` by
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

.. note::
    When running on CUDA, ``row * col`` must be less than :math:`2^{59}` to
    prevent overflow during calculation.

Args:
    row (``int``): number of rows in the 2-D matrix.
    col (``int``): number of columns in the 2-D matrix.
    offset (``int``): diagonal offset from the main diagonal.
        Default: if not provided, 0.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, ``torch.long``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    layout (:class:`torch.layout`, optional): currently only support ``torch.strided``.

Example::

    >>> a = torch.triu_indices(3, 3)
    >>> a
    tensor([[0, 0, 0, 1, 1, 2],
            [0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, -1)
    >>> a
    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
            [0, 1, 2, 0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1],
            [1, 2, 2]])
"""

    title = 'Triu_indicesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.triu_indices())



"""
WARNING: Module True_divideNode was generated using fallback option. May contain bugs
"""

class True_divideNode(Node):
    """
true_divide(dividend, divisor, *, out) -> Tensor

Alias for :func:`torch.div` with ``rounding_mode=None``.
"""

    title = 'True_divideNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.true_divide())



"""
WARNING: Module TruncNode was generated using fallback option. May contain bugs
"""

class TruncNode(Node):
    """
trunc(input, *, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
    >>> torch.trunc(a)
    tensor([ 3.,  0., -0., -0.])
"""

    title = 'TruncNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.trunc())



"""
WARNING: Module Trunc_Node was generated using fallback option. May contain bugs
"""

class Trunc_Node(Node):
    """None"""

    title = 'Trunc_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.trunc_())


class TypenameNode(Node):
    """None"""

    title = 'TypenameNode'
    init_inputs = [
        NodeInputBP('o'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.typename(self.input(0)))



"""
WARNING: Module TypesNode was generated using fallback option. May contain bugs
"""

class TypesNode(Node):
    """None"""

    title = 'TypesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.types())



"""
WARNING: Module Uint8Node was generated using fallback option. May contain bugs
"""

class Uint8Node(Node):
    """None"""

    title = 'Uint8Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.uint8())



"""
WARNING: Module UnbindNode was generated using fallback option. May contain bugs
"""

class UnbindNode(Node):
    """
unbind(input, dim=0) -> seq

Removes a tensor dimension.

Returns a tuple of all slices along a given dimension, already without it.

Arguments:
    input (Tensor): the tensor to unbind
    dim (int): dimension to remove

Example::

    >>> torch.unbind(torch.tensor([[1, 2, 3],
    >>>                            [4, 5, 6],
    >>>                            [7, 8, 9]]))
    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
"""

    title = 'UnbindNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unbind())



"""
WARNING: Module Unify_type_listNode was generated using fallback option. May contain bugs
"""

class Unify_type_listNode(Node):
    """unify_type_list(arg0: List[c10::Type]) -> c10::Type
"""

    title = 'Unify_type_listNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unify_type_list(self.input(0)))


class UniqueNode(Node):
    """Returns the unique elements of the input tensor.

    .. note:: This function is different from :func:`torch.unique_consecutive` in the sense that
        this function also eliminates non-consecutive duplicate values.

    .. note:: Currently in the CUDA implementation and the CPU implementation when dim is specified,
        `torch.unique` always sort the tensor at the beginning regardless of the `sort` argument.
        Sorting could be slow, so if your input tensor is already sorted, it is recommended to use
        :func:`torch.unique_consecutive` which avoids the sorting.

    Args:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.

    Example::

        >>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
        >>> output
        tensor([ 2,  3,  1])

        >>> output, inverse_indices = torch.unique(
        ...     torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([ 0,  2,  1,  2])

        >>> output, inverse_indices = torch.unique(
        ...     torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([[ 0,  2],
                [ 1,  2]])

    """

    title = 'UniqueNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unique())


class Unique_consecutiveNode(Node):
    """Eliminates all but the first element from every consecutive group of equivalent elements.

    .. note:: This function is different from :func:`torch.unique` in the sense that this function
        only eliminates consecutive duplicate values. This semantics is similar to `std::unique`
        in C++.

    Args:
        input (Tensor): the input tensor
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.

    Example::

        >>> x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
        >>> output = torch.unique_consecutive(x)
        >>> output
        tensor([1, 2, 3, 1, 2])

        >>> output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> inverse_indices
        tensor([0, 0, 1, 1, 2, 3, 3, 4])

        >>> output, counts = torch.unique_consecutive(x, return_counts=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> counts
        tensor([2, 2, 1, 2, 1])
    """

    title = 'Unique_consecutiveNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unique_consecutive())



"""
WARNING: Module Unsafe_chunkNode was generated using fallback option. May contain bugs
"""

class Unsafe_chunkNode(Node):
    """
unsafe_chunk(input, chunks, dim=0) -> List of Tensors

Works like :func:`torch.chunk` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warning::
    This function is safe to use as long as only the input, or only the outputs
    are modified inplace after calling this function. It is user's
    responsibility to ensure that is the case. If both the input and one or more
    of the outputs are modified inplace, gradients computed by autograd will be
    silently incorrect.
"""

    title = 'Unsafe_chunkNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unsafe_chunk())



"""
WARNING: Module Unsafe_splitNode was generated using fallback option. May contain bugs
"""

class Unsafe_splitNode(Node):
    """
unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors

Works like :func:`torch.split` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warning::
    This function is safe to use as long as only the input, or only the outputs
    are modified inplace after calling this function. It is user's
    responsibility to ensure that is the case. If both the input and one or more
    of the outputs are modified inplace, gradients computed by autograd will be
    silently incorrect.
"""

    title = 'Unsafe_splitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unsafe_split())



"""
WARNING: Module Unsafe_split_with_sizesNode was generated using fallback option. May contain bugs
"""

class Unsafe_split_with_sizesNode(Node):
    """None"""

    title = 'Unsafe_split_with_sizesNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unsafe_split_with_sizes())



"""
WARNING: Module UnsqueezeNode was generated using fallback option. May contain bugs
"""

class UnsqueezeNode(Node):
    """
unsqueeze(input, dim) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
applied at :attr:`dim` = ``dim + input.dim() + 1``.

Args:
    input (Tensor): the input tensor.
    dim (int): the index at which to insert the singleton dimension

Example::

    >>> x = torch.tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])
    >>> torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])
"""

    title = 'UnsqueezeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.unsqueeze())


class Use_deterministic_algorithmsNode(Node):
    """ Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.gather` when ``input`` dimension is one and called
          on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor
        * :func:`torch.kthvalue` with called on a CUDA tensor
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.gather` when ``input`` dimension is larger than one
          and called on a CUDA tensor that requires grad
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor

    A handful of CUDA operations are nondeterministic if the CUDA version is
    10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set. See the CUDA documentation for more
    details: `<https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
    If one of these environment variable configurations is not set, a :class:`RuntimeError`
    will be raised from these operations when called with CUDA tensors:

        * :func:`torch.mm`
        * :func:`torch.mv`
        * :func:`torch.bmm`

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.

    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Example::

        >>> torch.use_deterministic_algorithms(True)

        # Forward mode nondeterministic error
        >>> torch.randn(10).index_copy(0, torch.tensor([0]), torch.randn(1))
        ...
        RuntimeError: index_copy does not have a deterministic implementation...

        # Backward mode nondeterministic error
        >>> torch.randn(10, requires_grad=True, device='cuda').index_select(0, torch.tensor([0], device='cuda')).backward()
        ...
        RuntimeError: index_add_cuda_ does not have a deterministic implementation...
    """

    title = 'Use_deterministic_algorithmsNode'
    init_inputs = [
        NodeInputBP('mode'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.use_deterministic_algorithms(self.input(0)))



"""
WARNING: Module UtilsNode was generated using fallback option. May contain bugs
"""

class UtilsNode(Node):
    """None"""

    title = 'UtilsNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.utils())



"""
WARNING: Module VanderNode was generated using fallback option. May contain bugs
"""

class VanderNode(Node):
    """
vander(x, N=None, increasing=False) -> Tensor

Generates a Vandermonde matrix.

The columns of the output matrix are elementwise powers of the input vector :math:`x^{(N-1)}, x^{(N-2)}, ..., x^0`.
If increasing is True, the order of the columns is reversed :math:`x^0, x^1, ..., x^{(N-1)}`. Such a
matrix with a geometric progression in each row is named for Alexandre-Theophile Vandermonde.

Arguments:
    x (Tensor): 1-D input tensor.
    N (int, optional): Number of columns in the output. If N is not specified,
        a square array is returned :math:`(N = len(x))`.
    increasing (bool, optional): Order of the powers of the columns. If True,
        the powers increase from left to right, if False (the default) they are reversed.

Returns:
    Tensor: Vandermonde matrix. If increasing is False, the first column is :math:`x^{(N-1)}`,
    the second :math:`x^{(N-2)}` and so forth. If increasing is True, the columns
    are :math:`x^0, x^1, ..., x^{(N-1)}`.

Example::

    >>> x = torch.tensor([1, 2, 3, 5])
    >>> torch.vander(x)
    tensor([[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]])
    >>> torch.vander(x, N=3)
    tensor([[ 1,  1,  1],
            [ 4,  2,  1],
            [ 9,  3,  1],
            [25,  5,  1]])
    >>> torch.vander(x, N=3, increasing=True)
    tensor([[ 1,  1,  1],
            [ 1,  2,  4],
            [ 1,  3,  9],
            [ 1,  5, 25]])

"""

    title = 'VanderNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.vander())



"""
WARNING: Module VarNode was generated using fallback option. May contain bugs
"""

class VarNode(Node):
    """
var(input, dim, unbiased, keepdim=False, *, out=None) -> Tensor

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample variance is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.

Keyword args:
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
    out (Tensor, optional): the output tensor.

.. function:: var(input, unbiased) -> Tensor
   :noindex:

Calculates the variance of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).

Example::

    >>> a = torch.tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.var(a, unbiased=False)
    tensor(0.1754)
"""

    title = 'VarNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.var())



"""
WARNING: Module Var_meanNode was generated using fallback option. May contain bugs
"""

class Var_meanNode(Node):
    """
var_mean(input, dim, unbiased, keepdim=False, *, out=None) -> (Tensor, Tensor)

If :attr:`unbiased` is ``True``, Bessel's correction will be used to calculate
the variance. Otherwise, the sample variance is calculated, without any
correction.

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.

Keyword args:
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
    out (Tensor, optional): the output tensor.

Returns:
    A tuple (var, mean) containing the variance and mean.

.. function:: var_mean(input, unbiased) -> (Tensor, Tensor)
   :noindex:

Calculates the variance and mean of all elements in the :attr:`input`
tensor.

If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    input (Tensor): the input tensor.
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).

Returns:
    A tuple (var, mean) containing the variance and mean.

Example::

    >>> a = torch.tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.var_mean(a, unbiased=False)
    (tensor(0.1754), tensor(-0.8509))
"""

    title = 'Var_meanNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.var_mean())



"""
WARNING: Module VdotNode was generated using fallback option. May contain bugs
"""

class VdotNode(Node):
    """
vdot(input, other, *, out=None) -> Tensor

Computes the dot product of two 1D tensors. The vdot(a, b) function handles complex numbers
differently than dot(a, b). If the first argument is complex, the complex conjugate of the
first argument is used for the calculation of the dot product.

.. note::

    Unlike NumPy's vdot, torch.vdot intentionally only supports computing the dot product
    of two 1D tensors with the same number of elements.

Args:
    input (Tensor): first tensor in the dot product, must be 1D. Its conjugate is used if it's complex.
    other (Tensor): second tensor in the dot product, must be 1D.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.vdot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)
    >>> a = torch.tensor((1 +2j, 3 - 1j))
    >>> b = torch.tensor((2 +1j, 4 - 0j))
    >>> torch.vdot(a, b)
    tensor([16.+1.j])
    >>> torch.vdot(b, a)
    tensor([16.-1.j])
"""

    title = 'VdotNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.vdot())



"""
WARNING: Module VersionNode was generated using fallback option. May contain bugs
"""

class VersionNode(Node):
    """None"""

    title = 'VersionNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.version())



"""
WARNING: Module View_as_complexNode was generated using fallback option. May contain bugs
"""

class View_as_complexNode(Node):
    """
view_as_complex(input) -> Tensor

Returns a view of :attr:`input` as a complex tensor. For an input complex
tensor of :attr:`size` :math:`m1, m2, \dots, mi, 2`, this function returns a
new complex tensor of :attr:`size` :math:`m1, m2, \dots, mi` where the last
dimension of the input tensor is expected to represent the real and imaginary
components of complex numbers.

.. warning::
    :func:`view_as_complex` is only supported for tensors with
    :class:`torch.dtype` ``torch.float64`` and ``torch.float32``.  The input is
    expected to have the last dimension of :attr:`size` 2. In addition, the
    tensor must have a `stride` of 1 for its last dimension. The strides of all
    other dimensions must be even numbers.

Args:
    input (Tensor): the input tensor.

Example::

    >>> x=torch.randn(4, 2)
    >>> x
    tensor([[ 1.6116, -0.5772],
            [-1.4606, -0.9120],
            [ 0.0786, -1.7497],
            [-0.6561, -1.6623]])
    >>> torch.view_as_complex(x)
    tensor([(1.6116-0.5772j), (-1.4606-0.9120j), (0.0786-1.7497j), (-0.6561-1.6623j)])
"""

    title = 'View_as_complexNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.view_as_complex())



"""
WARNING: Module View_as_realNode was generated using fallback option. May contain bugs
"""

class View_as_realNode(Node):
    """
view_as_real(input) -> Tensor

Returns a view of :attr:`input` as a real tensor. For an input complex tensor of
:attr:`size` :math:`m1, m2, \dots, mi`, this function returns a new
real tensor of size :math:`m1, m2, \dots, mi, 2`, where the last dimension of size 2
represents the real and imaginary components of complex numbers.

.. warning::
    :func:`view_as_real` is only supported for tensors with ``complex dtypes``.

Args:
    input (Tensor): the input tensor.

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.4737-0.3839j), (-0.2098-0.6699j), (0.3470-0.9451j), (-0.5174-1.3136j)])
    >>> torch.view_as_real(x)
    tensor([[ 0.4737, -0.3839],
            [-0.2098, -0.6699],
            [ 0.3470, -0.9451],
            [-0.5174, -1.3136]])
"""

    title = 'View_as_realNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.view_as_real())



"""
WARNING: Module Vitals_enabledNode was generated using fallback option. May contain bugs
"""

class Vitals_enabledNode(Node):
    """vitals_enabled() -> bool
"""

    title = 'Vitals_enabledNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.vitals_enabled())



"""
WARNING: Module VsplitNode was generated using fallback option. May contain bugs
"""

class VsplitNode(Node):
    """
vsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with two or more dimensions, into multiple tensors
vertically according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0)
(the split dimension is 0), except that if :attr:`indices_or_sections` is an integer
it must evenly divide the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.vsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (Tensor, int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::
    >>> t = torch.arange(16.0).reshape(4,4)
    >>> t
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])
    >>> torch.vsplit(t, 2)
    (tensor([[0., 1., 2., 3.],
             [4., 5., 6., 7.]]),
     tensor([[ 8.,  9., 10., 11.],
             [12., 13., 14., 15.]]))
    >>> torch.vsplit(t, [3, 6])
    (tensor([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]]),
     tensor([[12., 13., 14., 15.]]),
     tensor([], size=(0, 4)))

"""

    title = 'VsplitNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.vsplit())



"""
WARNING: Module VstackNode was generated using fallback option. May contain bugs
"""

class VstackNode(Node):
    """
vstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence vertically (row wise).

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by :func:`torch.atleast_2d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.vstack((a,b))
    tensor([[1, 2, 3],
            [4, 5, 6]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.vstack((a,b))
    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])


"""

    title = 'VstackNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.vstack())



"""
WARNING: Module WaitNode was generated using fallback option. May contain bugs
"""

class WaitNode(Node):
    """wait(arg0: torch._C.Future) -> object
"""

    title = 'WaitNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.wait(self.input(0)))



"""
WARNING: Module WarningsNode was generated using fallback option. May contain bugs
"""

class WarningsNode(Node):
    """Python part of the warnings subsystem."""

    title = 'WarningsNode'
    init_inputs = [
        NodeInputBP('a'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.warnings(self.input(0)))



"""
WARNING: Module WhereNode was generated using fallback option. May contain bugs
"""

class WhereNode(Node):
    """
where(condition, x, y) -> Tensor

Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{x}_i & \text{if } \text{condition}_i \\
        \text{y}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

.. note::
    Currently valid scalar and tensor combination are
    1. Scalar of floating dtype and torch.double
    2. Scalar of integral dtype and torch.long
    3. Scalar of complex dtype and torch.complex128

Arguments:
    condition (BoolTensor): When True (nonzero), yield x, otherwise yield y
    x (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    y (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
    >>> torch.where(x > 0, x, 0.)
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)

.. function:: where(condition) -> tuple of LongTensor

``torch.where(condition)`` is identical to
``torch.nonzero(condition, as_tuple=True)``.

.. note::
    See also :func:`torch.nonzero`.
"""

    title = 'WhereNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.where())



"""
WARNING: Module XlogyNode was generated using fallback option. May contain bugs
"""

class XlogyNode(Node):
    """
xlogy(input, other, *, out=None) -> Tensor

Computes ``input * log(other)`` with the following cases.

.. math::
    \text{out}_{i} = \begin{cases}
        \text{NaN} & \text{if } \text{other}_{i} = \text{NaN} \\
        0 & \text{if } \text{input}_{i} = 0.0 \\
        \text{input}_{i} * \log{(\text{other}_{i})} & \text{otherwise}
    \end{cases}

Similar to SciPy's `scipy.special.xlogy`.



Args:
    input (Number or Tensor) : Multiplier
    other (Number or Tensor) : Argument

.. note:: At least one of :attr:`input` or :attr:`other` must be a tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.zeros(5,)
    >>> y = torch.tensor([-1, 0, 1, float('inf'), float('nan')])
    >>> torch.xlogy(x, y)
    tensor([0., 0., 0., 0., nan])
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> torch.xlogy(x, y)
    tensor([1.0986, 1.3863, 0.0000])
    >>> torch.xlogy(x, 4)
    tensor([1.3863, 2.7726, 4.1589])
    >>> torch.xlogy(2, y)
    tensor([2.1972, 1.3863, 0.0000])
"""

    title = 'XlogyNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.xlogy())



"""
WARNING: Module Xlogy_Node was generated using fallback option. May contain bugs
"""

class Xlogy_Node(Node):
    """None"""

    title = 'Xlogy_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.xlogy_())



"""
WARNING: Module Zero_Node was generated using fallback option. May contain bugs
"""

class Zero_Node(Node):
    """None"""

    title = 'Zero_Node'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.zero_())



"""
WARNING: Module ZerosNode was generated using fallback option. May contain bugs
"""

class ZerosNode(Node):
    """
zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `0`, with the shape defined
by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.zeros(2, 3)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> torch.zeros(5)
    tensor([ 0.,  0.,  0.,  0.,  0.])
"""

    title = 'ZerosNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.zeros())



"""
WARNING: Module Zeros_likeNode was generated using fallback option. May contain bugs
"""

class Zeros_likeNode(Node):
    """
zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`. ``torch.zeros_like(input)`` is equivalent to
``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.zeros_like(input, out=output)`` is equivalent to
    ``torch.zeros(input.size(), out=output)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
"""

    title = 'Zeros_likeNode'
    init_inputs = [
        
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#001199'

    def update_event(self, inp=-1):
        self.set_output_val(0, torch.zeros_like())

torch_nodes = [
    AvgNode, 
    AggregationtypeNode, 
    AliasdbNode, 
    AnytypeNode, 
    ArgumentNode, 
    ArgumentspecNode, 
    Bfloat16storageNode, 
    Bfloat16tensorNode, 
    BenchmarkconfigNode, 
    BenchmarkexecutionstatsNode, 
    BlockNode, 
    BoolstorageNode, 
    BooltensorNode, 
    BooltypeNode, 
    BufferdictNode, 
    BytestorageNode, 
    BytetensorNode, 
    Conv_bn_fusionNode, 
    CallstackNode, 
    CapsuleNode, 
    CharstorageNode, 
    ChartensorNode, 
    ClasstypeNode, 
    CodeNode, 
    CompilationunitNode, 
    CompleteargumentspecNode, 
    ComplexdoublestorageNode, 
    ComplexfloatstorageNode, 
    ComplextypeNode, 
    ConcretemoduletypeNode, 
    ConcretemoduletypebuilderNode, 
    DeepcopymemotableNode, 
    DeviceobjtypeNode, 
    DicttypeNode, 
    DisabletorchfunctionNode, 
    DoublestorageNode, 
    DoubletensorNode, 
    EnumtypeNode, 
    ErrorreportNode, 
    ExecutionplanNode, 
    Fuse_add_reluNode, 
    FatalerrorNode, 
    FilecheckNode, 
    FloatstorageNode, 
    FloattensorNode, 
    FloattypeNode, 
    FunctionschemaNode, 
    FutureNode, 
    FuturetypeNode, 
    GeneratorNode, 
    GradientNode, 
    GraphNode, 
    GraphexecutorstateNode, 
    Hoist_conv_packed_paramsNode, 
    HalfstorageNode, 
    HalfstoragebaseNode, 
    HalftensorNode, 
    Insert_fold_prepack_opsNode, 
    IodescriptorNode, 
    InferredtypeNode, 
    IntstorageNode, 
    InttensorNode, 
    InttypeNode, 
    InterfacetypeNode, 
    JitexceptionNode, 
    ListtypeNode, 
    LitescriptmoduleNode, 
    LockingloggerNode, 
    LoggerbaseNode, 
    LongstorageNode, 
    LongtensorNode, 
    MobileoptimizertypeNode, 
    ModuledictNode, 
    NodeNode, 
    NonetypeNode, 
    NooploggerNode, 
    NumbertypeNode, 
    OptionaltypeNode, 
    ParameterdictNode, 
    PyobjecttypeNode, 
    PytorchfilereaderNode, 
    PytorchfilewriterNode, 
    Qint32storageNode, 
    Qint32storagebaseNode, 
    Qint8storageNode, 
    Qint8storagebaseNode, 
    Quint4x2storageNode, 
    Quint8storageNode, 
    Remove_dropoutNode, 
    RreftypeNode, 
    SumNode, 
    ScriptclassNode, 
    ScriptclassfunctionNode, 
    ScriptfunctionNode, 
    ScriptmethodNode, 
    ScriptmoduleNode, 
    ScriptmoduleserializerNode, 
    ScriptobjectNode, 
    SetNode, 
    ShortstorageNode, 
    ShorttensorNode, 
    SizeNode, 
    StaticmoduleNode, 
    StorageNode, 
    StoragecontextNode, 
    StreamNode, 
    StreamobjtypeNode, 
    StringtypeNode, 
    Type_checkingNode, 
    TensorNode, 
    TensortypeNode, 
    ThroughputbenchmarkNode, 
    TracingstateNode, 
    TupletypeNode, 
    TypeNode, 
    Use_global_depsNode, 
    Use_rtld_global_with_libtorchNode, 
    UseNode, 
    ValueNode, 
    _cNode, 
    _storagebaseNode, 
    _vfNode, 
    _adaptive_avg_pool2dNode, 
    _adaptive_avg_pool3dNode, 
    _add_batch_dimNode, 
    _add_reluNode, 
    _add_relu_Node, 
    _aminmaxNode, 
    _amp_foreach_non_finite_check_and_unscale_Node, 
    _amp_update_scale_Node, 
    _assertNode, 
    _assert_asyncNode, 
    _autograd_functionsNode, 
    _baddbmm_mkl_Node, 
    _batch_norm_impl_indexNode, 
    _bmmNode, 
    _cast_byteNode, 
    _cast_charNode, 
    _cast_doubleNode, 
    _cast_floatNode, 
    _cast_halfNode, 
    _cast_intNode, 
    _cast_longNode, 
    _cast_shortNode, 
    _catNode, 
    _choose_qparams_per_tensorNode, 
    _classesNode, 
    _coalesceNode, 
    _compute_linear_combinationNode, 
    _conjNode, 
    _convolutionNode, 
    _convolution_modeNode, 
    _convolution_nogroupNode, 
    _copy_fromNode, 
    _ctc_lossNode, 
    _cudnn_ctc_lossNode, 
    _cudnn_init_dropout_stateNode, 
    _cudnn_rnnNode, 
    _cudnn_rnn_flatten_weightNode, 
    _cufft_clear_plan_cacheNode, 
    _cufft_get_plan_cache_max_sizeNode, 
    _cufft_get_plan_cache_sizeNode, 
    _cufft_set_plan_cache_max_sizeNode, 
    _cummax_helperNode, 
    _cummin_helperNode, 
    _debug_has_internal_overlapNode, 
    _dim_arangeNode, 
    _dirichlet_gradNode, 
    _embedding_bagNode, 
    _embedding_bag_forward_onlyNode, 
    _empty_affine_quantizedNode, 
    _empty_per_channel_affine_quantizedNode, 
    _euclidean_distNode, 
    _fake_quantize_learnable_per_channel_affineNode, 
    _fake_quantize_learnable_per_tensor_affineNode, 
    _fft_c2cNode, 
    _fft_c2rNode, 
    _fft_r2cNode, 
    _foreach_absNode, 
    _foreach_abs_Node, 
    _foreach_acosNode, 
    _foreach_acos_Node, 
    _foreach_addNode, 
    _foreach_add_Node, 
    _foreach_addcdivNode, 
    _foreach_addcdiv_Node, 
    _foreach_addcmulNode, 
    _foreach_addcmul_Node, 
    _foreach_asinNode, 
    _foreach_asin_Node, 
    _foreach_atanNode, 
    _foreach_atan_Node, 
    _foreach_ceilNode, 
    _foreach_ceil_Node, 
    _foreach_cosNode, 
    _foreach_cos_Node, 
    _foreach_coshNode, 
    _foreach_cosh_Node, 
    _foreach_divNode, 
    _foreach_div_Node, 
    _foreach_erfNode, 
    _foreach_erf_Node, 
    _foreach_erfcNode, 
    _foreach_erfc_Node, 
    _foreach_expNode, 
    _foreach_exp_Node, 
    _foreach_expm1Node, 
    _foreach_expm1_Node, 
    _foreach_floorNode, 
    _foreach_floor_Node, 
    _foreach_fracNode, 
    _foreach_frac_Node, 
    _foreach_lgammaNode, 
    _foreach_lgamma_Node, 
    _foreach_logNode, 
    _foreach_log10Node, 
    _foreach_log10_Node, 
    _foreach_log1pNode, 
    _foreach_log1p_Node, 
    _foreach_log2Node, 
    _foreach_log2_Node, 
    _foreach_log_Node, 
    _foreach_maximumNode, 
    _foreach_minimumNode, 
    _foreach_mulNode, 
    _foreach_mul_Node, 
    _foreach_negNode, 
    _foreach_neg_Node, 
    _foreach_reciprocalNode, 
    _foreach_reciprocal_Node, 
    _foreach_roundNode, 
    _foreach_round_Node, 
    _foreach_sigmoidNode, 
    _foreach_sigmoid_Node, 
    _foreach_sinNode, 
    _foreach_sin_Node, 
    _foreach_sinhNode, 
    _foreach_sinh_Node, 
    _foreach_sqrtNode, 
    _foreach_sqrt_Node, 
    _foreach_subNode, 
    _foreach_sub_Node, 
    _foreach_tanNode, 
    _foreach_tan_Node, 
    _foreach_tanhNode, 
    _foreach_tanh_Node, 
    _foreach_truncNode, 
    _foreach_trunc_Node, 
    _foreach_zero_Node, 
    _fused_dropoutNode, 
    _grid_sampler_2d_cpu_fallbackNode, 
    _has_compatible_shallow_copy_typeNode, 
    _import_dotted_nameNode, 
    _index_copy_Node, 
    _index_put_impl_Node, 
    _initextensionNode, 
    _jit_internalNode, 
    _linalg_inv_out_helper_Node, 
    _linalg_qr_helperNode, 
    _linalg_solve_out_helper_Node, 
    _linalg_utilsNode, 
    _load_global_depsNode, 
    _lobpcgNode, 
    _log_softmaxNode, 
    _log_softmax_backward_dataNode, 
    _logcumsumexpNode, 
    _lowrankNode, 
    _lu_with_infoNode, 
    _make_dualNode, 
    _make_per_channel_quantized_tensorNode, 
    _make_per_tensor_quantized_tensorNode, 
    _masked_scaleNode, 
    _mkldnnNode, 
    _mkldnn_reshapeNode, 
    _mkldnn_transposeNode, 
    _mkldnn_transpose_Node, 
    _namedtensor_internalsNode, 
    _nnpack_availableNode, 
    _nnpack_spatial_convolutionNode, 
    _opsNode, 
    _pack_padded_sequenceNode, 
    _pad_packed_sequenceNode, 
    _remove_batch_dimNode, 
    _reshape_from_tensorNode, 
    _rowwise_pruneNode, 
    _s_whereNode, 
    _sample_dirichletNode, 
    _saturate_weight_to_fp16Node, 
    _shape_as_tensorNode, 
    _sixNode, 
    _sobol_engine_drawNode, 
    _sobol_engine_ff_Node, 
    _sobol_engine_initialize_state_Node, 
    _sobol_engine_scramble_Node, 
    _softmaxNode, 
    _softmax_backward_dataNode, 
    _sparse_addmmNode, 
    _sparse_coo_tensor_unsafeNode, 
    _sparse_csr_tensorNode, 
    _sparse_log_softmaxNode, 
    _sparse_log_softmax_backward_dataNode, 
    _sparse_mask_helperNode, 
    _sparse_mmNode, 
    _sparse_softmaxNode, 
    _sparse_softmax_backward_dataNode, 
    _sparse_sparse_matmulNode, 
    _sparse_sumNode, 
    _stackNode, 
    _standard_gammaNode, 
    _standard_gamma_gradNode, 
    _storage_classesNode, 
    _string_classesNode, 
    _tensorNode, 
    _tensor_classesNode, 
    _tensor_strNode, 
    _test_serialization_subcmulNode, 
    _trilinearNode, 
    _uniqueNode, 
    _unique2Node, 
    _unpack_dualNode, 
    _use_cudnn_ctc_lossNode, 
    _use_cudnn_rnn_flatten_weightNode, 
    _utilsNode, 
    _utils_internalNode, 
    _validate_sparse_coo_tensor_argsNode, 
    _vmap_internalsNode, 
    _weight_normNode, 
    _weight_norm_cuda_interfaceNode, 
    AbsNode, 
    Abs_Node, 
    AbsoluteNode, 
    AcosNode, 
    Acos_Node, 
    AcoshNode, 
    Acosh_Node, 
    Adaptive_avg_pool1dNode, 
    Adaptive_max_pool1dNode, 
    AddNode, 
    AddbmmNode, 
    AddcdivNode, 
    AddcmulNode, 
    AddmmNode, 
    AddmvNode, 
    Addmv_Node, 
    AddrNode, 
    Affine_grid_generatorNode, 
    Align_tensorsNode, 
    AllNode, 
    AllcloseNode, 
    Alpha_dropoutNode, 
    Alpha_dropout_Node, 
    AmaxNode, 
    AminNode, 
    AngleNode, 
    AnyNode, 
    ArangeNode, 
    ArccosNode, 
    Arccos_Node, 
    ArccoshNode, 
    Arccosh_Node, 
    ArcsinNode, 
    Arcsin_Node, 
    ArcsinhNode, 
    Arcsinh_Node, 
    ArctanNode, 
    Arctan_Node, 
    ArctanhNode, 
    Arctanh_Node, 
    Are_deterministic_algorithms_enabledNode, 
    ArgmaxNode, 
    ArgminNode, 
    ArgsortNode, 
    As_stridedNode, 
    As_strided_Node, 
    As_tensorNode, 
    AsinNode, 
    Asin_Node, 
    AsinhNode, 
    Asinh_Node, 
    AtanNode, 
    Atan2Node, 
    Atan_Node, 
    AtanhNode, 
    Atanh_Node, 
    Atleast_1dNode, 
    Atleast_2dNode, 
    Atleast_3dNode, 
    AttrNode, 
    Autocast_decrement_nestingNode, 
    Autocast_increment_nestingNode, 
    AutogradNode, 
    Avg_pool1dNode, 
    BackendsNode, 
    BaddbmmNode, 
    Bartlett_windowNode, 
    Batch_normNode, 
    Batch_norm_backward_elemtNode, 
    Batch_norm_backward_reduceNode, 
    Batch_norm_elemtNode, 
    Batch_norm_gather_statsNode, 
    Batch_norm_gather_stats_with_countsNode, 
    Batch_norm_statsNode, 
    Batch_norm_update_statsNode, 
    BernoulliNode, 
    Bfloat16Node, 
    BilinearNode, 
    Binary_cross_entropy_with_logitsNode, 
    BincountNode, 
    BinomialNode, 
    Bitwise_andNode, 
    Bitwise_notNode, 
    Bitwise_orNode, 
    Bitwise_xorNode, 
    Blackman_windowNode, 
    Block_diagNode, 
    BmmNode, 
    BoolNode, 
    Broadcast_shapesNode, 
    Broadcast_tensorsNode, 
    Broadcast_toNode, 
    BucketizeNode, 
    Can_castNode, 
    CandidateNode, 
    Cartesian_prodNode, 
    CatNode, 
    CdistNode, 
    CdoubleNode, 
    CeilNode, 
    Ceil_Node, 
    CeluNode, 
    Celu_Node, 
    CfloatNode, 
    Chain_matmulNode, 
    Channel_shuffleNode, 
    Channels_lastNode, 
    Channels_last_3dNode, 
    CholeskyNode, 
    Cholesky_inverseNode, 
    Cholesky_solveNode, 
    Choose_qparams_optimizedNode, 
    ChunkNode, 
    ClampNode, 
    Clamp_Node, 
    Clamp_maxNode, 
    Clamp_max_Node, 
    Clamp_minNode, 
    Clamp_min_Node, 
    ClassesNode, 
    Clear_autocast_cacheNode, 
    ClipNode, 
    Clip_Node, 
    CloneNode, 
    Column_stackNode, 
    CombinationsNode, 
    Compiled_with_cxx11_abiNode, 
    ComplexNode, 
    Complex128Node, 
    Complex32Node, 
    Complex64Node, 
    ConjNode, 
    Constant_pad_ndNode, 
    Contiguous_formatNode, 
    Conv1dNode, 
    Conv2dNode, 
    Conv3dNode, 
    Conv_tbcNode, 
    Conv_transpose1dNode, 
    Conv_transpose2dNode, 
    Conv_transpose3dNode, 
    ConvolutionNode, 
    CopysignNode, 
    CosNode, 
    Cos_Node, 
    CoshNode, 
    Cosh_Node, 
    Cosine_embedding_lossNode, 
    Cosine_similarityNode, 
    Count_nonzeroNode, 
    CppNode, 
    CrossNode, 
    Ctc_lossNode, 
    CtypesNode, 
    CudaNode, 
    Cudnn_affine_grid_generatorNode, 
    Cudnn_batch_normNode, 
    Cudnn_convolutionNode, 
    Cudnn_convolution_add_reluNode, 
    Cudnn_convolution_reluNode, 
    Cudnn_convolution_transposeNode, 
    Cudnn_grid_samplerNode, 
    Cudnn_is_acceptableNode, 
    CummaxNode, 
    CumminNode, 
    CumprodNode, 
    CumsumNode, 
    Default_generatorNode, 
    Deg2radNode, 
    Deg2rad_Node, 
    DequantizeNode, 
    DetNode, 
    DetachNode, 
    Detach_Node, 
    DeviceNode, 
    DiagNode, 
    Diag_embedNode, 
    DiagflatNode, 
    DiagonalNode, 
    DiffNode, 
    DigammaNode, 
    DistNode, 
    DistributedNode, 
    DistributionsNode, 
    DivNode, 
    DivideNode, 
    DotNode, 
    DoubleNode, 
    DropoutNode, 
    Dropout_Node, 
    DsmmNode, 
    DsplitNode, 
    DstackNode, 
    DtypeNode, 
    EigNode, 
    EinsumNode, 
    EmbeddingNode, 
    Embedding_bagNode, 
    Embedding_renorm_Node, 
    EmptyNode, 
    Empty_likeNode, 
    Empty_quantizedNode, 
    Empty_stridedNode, 
    Enable_gradNode, 
    EqNode, 
    EqualNode, 
    ErfNode, 
    Erf_Node, 
    ErfcNode, 
    Erfc_Node, 
    ErfinvNode, 
    ExpNode, 
    Exp2Node, 
    Exp2_Node, 
    Exp_Node, 
    Expm1Node, 
    Expm1_Node, 
    EyeNode, 
    Fake_quantize_per_channel_affineNode, 
    Fake_quantize_per_tensor_affineNode, 
    Fbgemm_linear_fp16_weightNode, 
    Fbgemm_linear_fp16_weight_fp32_activationNode, 
    Fbgemm_linear_int8_weightNode, 
    Fbgemm_linear_int8_weight_fp32_activationNode, 
    Fbgemm_linear_quantize_weightNode, 
    Fbgemm_pack_gemm_matrix_fp16Node, 
    Fbgemm_pack_quantized_matrixNode, 
    Feature_alpha_dropoutNode, 
    Feature_alpha_dropout_Node, 
    Feature_dropoutNode, 
    Feature_dropout_Node, 
    FftNode, 
    Fill_Node, 
    FinfoNode, 
    FixNode, 
    Fix_Node, 
    FlattenNode, 
    FlipNode, 
    FliplrNode, 
    FlipudNode, 
    FloatNode, 
    Float16Node, 
    Float32Node, 
    Float64Node, 
    Float_powerNode, 
    FloorNode, 
    Floor_Node, 
    Floor_divideNode, 
    FmaxNode, 
    FminNode, 
    FmodNode, 
    ForkNode, 
    FracNode, 
    Frac_Node, 
    FrexpNode, 
    Frobenius_normNode, 
    From_fileNode, 
    From_numpyNode, 
    FullNode, 
    Full_likeNode, 
    FunctionalNode, 
    FuturesNode, 
    GatherNode, 
    GcdNode, 
    Gcd_Node, 
    GeNode, 
    GeqrfNode, 
    GerNode, 
    Get_default_dtypeNode, 
    Get_deviceNode, 
    Get_file_pathNode, 
    Get_num_interop_threadsNode, 
    Get_num_threadsNode, 
    Get_rng_stateNode, 
    GradientNode, 
    GreaterNode, 
    Greater_equalNode, 
    Grid_samplerNode, 
    Grid_sampler_2dNode, 
    Grid_sampler_3dNode, 
    Group_normNode, 
    GruNode, 
    Gru_cellNode, 
    GtNode, 
    HalfNode, 
    Hamming_windowNode, 
    Hann_windowNode, 
    HardshrinkNode, 
    Has_cudaNode, 
    Has_cudnnNode, 
    Has_lapackNode, 
    Has_mklNode, 
    Has_mkldnnNode, 
    Has_mlcNode, 
    Has_openmpNode, 
    HeavisideNode, 
    Hinge_embedding_lossNode, 
    HistcNode, 
    HsmmNode, 
    HsplitNode, 
    HspmmNode, 
    HstackNode, 
    HubNode, 
    HypotNode, 
    I0Node, 
    I0_Node, 
    IgammaNode, 
    IgammacNode, 
    IinfoNode, 
    ImagNode, 
    Import_ir_moduleNode, 
    Import_ir_module_from_bufferNode, 
    Index_addNode, 
    Index_copyNode, 
    Index_fillNode, 
    Index_putNode, 
    Index_put_Node, 
    Index_selectNode, 
    Inference_modeNode, 
    Init_num_threadsNode, 
    Initial_seedNode, 
    InnerNode, 
    Instance_normNode, 
    IntNode, 
    Int16Node, 
    Int32Node, 
    Int64Node, 
    Int8Node, 
    Int_reprNode, 
    InverseNode, 
    Is_anomaly_enabledNode, 
    Is_autocast_enabledNode, 
    Is_complexNode, 
    Is_deterministicNode, 
    Is_distributedNode, 
    Is_floating_pointNode, 
    Is_grad_enabledNode, 
    Is_inference_mode_enabledNode, 
    Is_nonzeroNode, 
    Is_same_sizeNode, 
    Is_signedNode, 
    Is_storageNode, 
    Is_tensorNode, 
    Is_vulkan_availableNode, 
    Is_warn_always_enabledNode, 
    IscloseNode, 
    IsfiniteNode, 
    IsinfNode, 
    IsnanNode, 
    IsneginfNode, 
    IsposinfNode, 
    IsrealNode, 
    IstftNode, 
    JitNode, 
    Kaiser_windowNode, 
    Kl_divNode, 
    KronNode, 
    KthvalueNode, 
    Layer_normNode, 
    LayoutNode, 
    LcmNode, 
    Lcm_Node, 
    LdexpNode, 
    Ldexp_Node, 
    LeNode, 
    Legacy_contiguous_formatNode, 
    LerpNode, 
    LessNode, 
    Less_equalNode, 
    LgammaNode, 
    LinalgNode, 
    LinspaceNode, 
    LoadNode, 
    LobpcgNode, 
    LogNode, 
    Log10Node, 
    Log10_Node, 
    Log1pNode, 
    Log1p_Node, 
    Log2Node, 
    Log2_Node, 
    Log_Node, 
    Log_softmaxNode, 
    LogaddexpNode, 
    Logaddexp2Node, 
    LogcumsumexpNode, 
    LogdetNode, 
    Logical_andNode, 
    Logical_notNode, 
    Logical_orNode, 
    Logical_xorNode, 
    LogitNode, 
    Logit_Node, 
    LogspaceNode, 
    LogsumexpNode, 
    LongNode, 
    LstmNode, 
    Lstm_cellNode, 
    LstsqNode, 
    LtNode, 
    LuNode, 
    Lu_solveNode, 
    Lu_unpackNode, 
    Manual_seedNode, 
    Margin_ranking_lossNode, 
    Masked_fillNode, 
    Masked_scatterNode, 
    Masked_selectNode, 
    MatmulNode, 
    Matrix_expNode, 
    Matrix_powerNode, 
    Matrix_rankNode, 
    MaxNode, 
    Max_pool1dNode, 
    Max_pool1d_with_indicesNode, 
    Max_pool2dNode, 
    Max_pool3dNode, 
    MaximumNode, 
    MeanNode, 
    MedianNode, 
    Memory_formatNode, 
    Merge_type_from_type_commentNode, 
    MeshgridNode, 
    MinNode, 
    MinimumNode, 
    Miopen_batch_normNode, 
    Miopen_convolutionNode, 
    Miopen_convolution_transposeNode, 
    Miopen_depthwise_convolutionNode, 
    Miopen_rnnNode, 
    Mkldnn_adaptive_avg_pool2dNode, 
    Mkldnn_convolutionNode, 
    Mkldnn_convolution_backward_weightsNode, 
    Mkldnn_linear_backward_weightsNode, 
    Mkldnn_max_pool2dNode, 
    Mkldnn_max_pool3dNode, 
    MmNode, 
    ModeNode, 
    MoveaxisNode, 
    MovedimNode, 
    MsortNode, 
    MulNode, 
    MultinomialNode, 
    MultiplyNode, 
    MultiprocessingNode, 
    MvNode, 
    MvlgammaNode, 
    NameNode, 
    Nan_to_numNode, 
    Nan_to_num_Node, 
    NanmedianNode, 
    NanquantileNode, 
    NansumNode, 
    NarrowNode, 
    Narrow_copyNode, 
    Native_batch_normNode, 
    Native_group_normNode, 
    Native_layer_normNode, 
    Native_normNode, 
    NeNode, 
    NegNode, 
    Neg_Node, 
    NegativeNode, 
    Negative_Node, 
    NextafterNode, 
    NnNode, 
    No_gradNode, 
    NonzeroNode, 
    NormNode, 
    Norm_except_dimNode, 
    NormalNode, 
    Not_equalNode, 
    Nuclear_normNode, 
    NumelNode, 
    OnesNode, 
    Ones_likeNode, 
    OnnxNode, 
    OpsNode, 
    OptimNode, 
    OrgqrNode, 
    OrmqrNode, 
    OsNode, 
    OuterNode, 
    OverridesNode, 
    PackageNode, 
    Pairwise_distanceNode, 
    Parse_irNode, 
    Parse_schemaNode, 
    Parse_type_commentNode, 
    Pca_lowrankNode, 
    PdistNode, 
    Per_channel_affineNode, 
    Per_channel_affine_float_qparamsNode, 
    Per_channel_symmetricNode, 
    Per_tensor_affineNode, 
    Per_tensor_symmetricNode, 
    PermuteNode, 
    PinverseNode, 
    Pixel_shuffleNode, 
    Pixel_unshuffleNode, 
    PlatformNode, 
    PoissonNode, 
    Poisson_nll_lossNode, 
    PolarNode, 
    PolygammaNode, 
    PositiveNode, 
    PowNode, 
    PreluNode, 
    Prepare_multiprocessing_environmentNode, 
    Preserve_formatNode, 
    ProdNode, 
    ProfilerNode, 
    Promote_typesNode, 
    PutNode, 
    Q_per_channel_axisNode, 
    Q_per_channel_scalesNode, 
    Q_per_channel_zero_pointsNode, 
    Q_scaleNode, 
    Q_zero_pointNode, 
    Qint32Node, 
    Qint8Node, 
    QrNode, 
    QschemeNode, 
    QuantileNode, 
    QuantizationNode, 
    Quantize_per_channelNode, 
    Quantize_per_tensorNode, 
    Quantized_batch_normNode, 
    Quantized_gruNode, 
    Quantized_gru_cellNode, 
    Quantized_lstmNode, 
    Quantized_lstm_cellNode, 
    Quantized_max_pool1dNode, 
    Quantized_max_pool2dNode, 
    Quantized_rnn_relu_cellNode, 
    Quantized_rnn_tanh_cellNode, 
    QuasirandomNode, 
    Quint4x2Node, 
    Quint8Node, 
    Rad2degNode, 
    Rad2deg_Node, 
    RandNode, 
    Rand_likeNode, 
    RandintNode, 
    Randint_likeNode, 
    RandnNode, 
    Randn_likeNode, 
    RandomNode, 
    RandpermNode, 
    RangeNode, 
    RavelNode, 
    RealNode, 
    ReciprocalNode, 
    Reciprocal_Node, 
    ReluNode, 
    Relu_Node, 
    RemainderNode, 
    RenormNode, 
    Repeat_interleaveNode, 
    ReshapeNode, 
    Resize_as_Node, 
    Resize_as_sparse_Node, 
    Result_typeNode, 
    Rnn_reluNode, 
    Rnn_relu_cellNode, 
    Rnn_tanhNode, 
    Rnn_tanh_cellNode, 
    RollNode, 
    Rot90Node, 
    RoundNode, 
    Round_Node, 
    Row_stackNode, 
    RreluNode, 
    Rrelu_Node, 
    RsqrtNode, 
    Rsqrt_Node, 
    RsubNode, 
    SaddmmNode, 
    SaveNode, 
    Scalar_tensorNode, 
    ScatterNode, 
    Scatter_addNode, 
    SearchsortedNode, 
    SeedNode, 
    Segment_reduceNode, 
    SelectNode, 
    SeluNode, 
    Selu_Node, 
    SerializationNode, 
    Set_anomaly_enabledNode, 
    Set_autocast_enabledNode, 
    Set_default_dtypeNode, 
    Set_default_tensor_typeNode, 
    Set_deterministicNode, 
    Set_flush_denormalNode, 
    Set_grad_enabledNode, 
    Set_num_interop_threadsNode, 
    Set_num_threadsNode, 
    Set_printoptionsNode, 
    Set_rng_stateNode, 
    Set_vitalNode, 
    Set_warn_alwaysNode, 
    SgnNode, 
    ShortNode, 
    SigmoidNode, 
    Sigmoid_Node, 
    SignNode, 
    SignbitNode, 
    SinNode, 
    Sin_Node, 
    SincNode, 
    Sinc_Node, 
    SinhNode, 
    Sinh_Node, 
    SlogdetNode, 
    SmmNode, 
    SoftmaxNode, 
    SolveNode, 
    SortNode, 
    SparseNode, 
    Sparse_cooNode, 
    Sparse_coo_tensorNode, 
    Sparse_csrNode, 
    SpecialNode, 
    SplitNode, 
    Split_with_sizesNode, 
    SpmmNode, 
    SqrtNode, 
    Sqrt_Node, 
    SquareNode, 
    Square_Node, 
    SqueezeNode, 
    SspaddmmNode, 
    StackNode, 
    StdNode, 
    Std_meanNode, 
    StftNode, 
    StorageNode, 
    StridedNode, 
    SubNode, 
    SubtractNode, 
    SumNode, 
    SvdNode, 
    Svd_lowrankNode, 
    SwapaxesNode, 
    SwapdimsNode, 
    SymeigNode, 
    SysNode, 
    TNode, 
    TakeNode, 
    Take_along_dimNode, 
    TanNode, 
    Tan_Node, 
    TanhNode, 
    Tanh_Node, 
    TensorNode, 
    Tensor_splitNode, 
    TensordotNode, 
    TestingNode, 
    TextwrapNode, 
    ThresholdNode, 
    Threshold_Node, 
    TileNode, 
    TopkNode, 
    TorchNode, 
    TraceNode, 
    TransposeNode, 
    TrapzNode, 
    Triangular_solveNode, 
    TrilNode, 
    Tril_indicesNode, 
    Triplet_margin_lossNode, 
    TriuNode, 
    Triu_indicesNode, 
    True_divideNode, 
    TruncNode, 
    Trunc_Node, 
    TypenameNode, 
    TypesNode, 
    Uint8Node, 
    UnbindNode, 
    Unify_type_listNode, 
    UniqueNode, 
    Unique_consecutiveNode, 
    Unsafe_chunkNode, 
    Unsafe_splitNode, 
    Unsafe_split_with_sizesNode, 
    UnsqueezeNode, 
    Use_deterministic_algorithmsNode, 
    UtilsNode, 
    VanderNode, 
    VarNode, 
    Var_meanNode, 
    VdotNode, 
    VersionNode, 
    View_as_complexNode, 
    View_as_realNode, 
    Vitals_enabledNode, 
    VsplitNode, 
    VstackNode, 
    WaitNode, 
    WarningsNode, 
    WhereNode, 
    XlogyNode, 
    Xlogy_Node, 
    Zero_Node, 
    ZerosNode, 
    Zeros_likeNode	
]
export_nodes(*torch_nodes)
    