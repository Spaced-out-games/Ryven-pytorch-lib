from ryven.NENV import *
from math import *

class AcosNode(Node):
    """Return the arc cosine (measured in radians) of x."""

    title = 'AcosNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x')
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, acos(self.input(0)))
class AcoshNode(Node):
    """Return the inverse hyperbolic cosine of x."""

    title = 'AcoshNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, acosh(self.input(0)))
class AsinNode(Node):
    """Return the arc sine (measured in radians) of x."""

    title = 'AsinNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, asin(self.input(0)))
class AsinhNode(Node):
    """Return the inverse hyperbolic sine of x."""

    title = 'AsinhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, asinh(self.input(0)))
class AtanNode(Node):
    """Return the arc tangent (measured in radians) of x."""

    title = 'AtanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, atan(self.input(0)))
class Atan2Node(Node):
    """Return the arc tangent (measured in radians) of y/x.

Unlike atan(y/x), the signs of both x and y are considered."""

    title = 'Atan2Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, atan2(self.input(0), self.input(1)))
class AtanhNode(Node):
    """Return the inverse hyperbolic tangent of x."""

    title = 'AtanhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, atanh(self.input(0)))
class CeilNode(Node):
    """Return the ceiling of x as an Integral.

This is the smallest integer >= x."""

    title = 'CeilNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, ceil(self.input(0)))
class CopysignNode(Node):
    """Return a float with the magnitude (absolute value) of x but the sign of y.

On platforms that support signed zeros, copysign(1.0, -0.0)
returns -1.0.
"""

    title = 'CopysignNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, copysign(self.input(0), self.input(1)))
class CosNode(Node):
    """Return the cosine of x (measured in radians)."""

    title = 'CosNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, cos(self.input(0)))
class CoshNode(Node):
    """Return the hyperbolic cosine of x."""

    title = 'CoshNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, cosh(self.input(0)))
class DegreesNode(Node):
    """Convert angle x from radians to degrees."""
    title = 'DegreesNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    tags = ["deg"]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, degrees(self.input(0)))
class ENode(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'ENode'
    init_inputs = [
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, e)
class ErfNode(Node):
    """Error function at x."""

    title = 'ErfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, erf(self.input(0)))
class ErfcNode(Node):
    """Complementary error function at x."""

    title = 'ErfcNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, erfc(self.input(0)))
class ExpNode(Node):
    """Return e raised to the power of x."""

    title = 'ExpNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, exp(self.input(0)))
class Expm1Node(Node):
    """Return exp(x)-1.

This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x."""

    title = 'Expm1Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, expm1(self.input(0)))
class FabsNode(Node):
    """Return the absolute value of the float x."""

    title = 'FabsNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, fabs(self.input(0)))
class FactorialNode(Node):
    """Find x!.

Raise a ValueError if x is negative or non-integral."""

    title = 'FactorialNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, factorial(self.input(0)))
class FloorNode(Node):
    """Return the floor of x as an Integral.

This is the largest integer <= x."""

    title = 'FloorNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, floor(self.input(0)))
class FmodNode(Node):
    """Return fmod(x, y), according to platform C.

x % y may differ."""

    title = 'FmodNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, fmod(self.input(0), self.input(1)))
class FrexpNode(Node):
    """Return the mantissa and exponent of x, as pair (m, e).

m is a float and e is an int, such that x = m * 2.**e.
If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0."""

    title = 'FrexpNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, frexp(self.input(0)))
class FsumNode(Node):
    """Return an accurate floating point sum of values in the iterable seq.

Assumes IEEE-754 floating point arithmetic."""

    title = 'FsumNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'seq'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, fsum(self.input(0)))
class GammaNode(Node):
    """Gamma function at x."""

    title = 'GammaNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, gamma(self.input(0)))
class GcdNode(Node):
    """greatest common divisor of x and y"""

    title = 'GcdNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, gcd(self.input(0), self.input(1)))
class HypotNode(Node):
    """Return the Euclidean distance, sqrt(x*x + y*y)."""

    title = 'HypotNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, hypot(self.input(0), self.input(1)))

"""
WARNING: Module InfNode was generated using fallback option. May contain bugs
"""
class InfNode(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'InfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, inf(self.input(0), self.input(1)))

#rel_tol abs_tol
class IscloseNode(Node):
    """Determine whether two floating point numbers are close in value.

  rel_tol
    maximum difference for being considered "close", relative to the
    magnitude of the input values
  abs_tol
    maximum difference for being considered "close", regardless of the
    magnitude of the input values

Return True if a is close in value to b, and False otherwise.

For the values to be considered close, the difference between them
must be smaller than at least one of the tolerances.

-inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
is, NaN is not close to anything, even itself.  inf and -inf are
only close to themselves."""

    title = 'IscloseNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
        NodeInputBP(dtype=dtypes.Data(default=1), label = '')
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isclose(self.input(0), self.input(1)))
class IsfiniteNode(Node):
    """Return True if x is neither an infinity nor a NaN, and False otherwise."""

    title = 'IsfiniteNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isfinite(self.input(0)))
class IsinfNode(Node):
    """Return True if x is a positive or negative infinity, and False otherwise."""

    title = 'IsinfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isinf(self.input(0)))
class IsnanNode(Node):
    """Return True if x is a NaN (not a number), and False otherwise."""

    title = 'IsnanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isnan(self.input(0)))
class LdexpNode(Node):
    """Return x * (2**i).

This is essentially the inverse of frexp()."""

    title = 'LdexpNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'i'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, ldexp(self.input(0), self.input(1)))
class LgammaNode(Node):
    """Natural logarithm of absolute value of Gamma function at x."""

    title = 'LgammaNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, lgamma(self.input(0)))

"""
WARNING: Module LogNode was generated using fallback option. May contain bugs
"""

class LogNode(Node):
    """log(x, [base=math.e])
Return the logarithm of x to the given base.

If the base not specified, returns the natural logarithm (base e) of x."""

    title = 'LogNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log(self.input(0), self.input(1)))
class Log10Node(Node):
    """Return the base 10 logarithm of x."""

    title = 'Log10Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log10(self.input(0)))
class Log1pNode(Node):
    """Return the natural logarithm of 1+x (base e).

The result is computed in a way which is accurate for x near zero."""

    title = 'Log1pNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log1p(self.input(0)))
class Log2Node(Node):
    """Return the base 2 logarithm of x."""

    title = 'Log2Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log2(self.input(0)))
class ModfNode(Node):
    """Return the fractional and integer parts of x.

Both results carry the sign of x and are floats."""

    title = 'ModfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, modf(self.input(0)))

"""
WARNING: Module NanNode was generated using fallback option. May contain bugs
"""

class NanNode(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'NanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, nan(self.input(0), self.input(1)))

"""
WARNING: Module PiNode was generated using fallback option. May contain bugs
"""

class PiNode(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'PiNode'
    init_inputs = [
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'
    def update_event(self, inp=-1):
        self.set_output_val(0, pi)
class PowNode(Node):
    """Return x**y (x to the power of y)."""

    title = 'PowNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, pow(self.input(0), self.input(1)))
class RadiansNode(Node):
    """Convert angle x from degrees to radians."""

    title = 'RadiansNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, radians(self.input(0)))
class RemainderNode(Node):
    """Difference between x and the closest integer multiple of y.

Return x - n*y where n*y is the closest integer multiple of y.
In the case where x is exactly halfway between two multiples of
y, the nearest even value of n is used. The result is always exact."""

    title = 'RemainderNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'y'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, remainder(self.input(0), self.input(1)))
class SinNode(Node):
    """Return the sine of x (measured in radians)."""

    title = 'SinNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sin(self.input(0)))
class SinhNode(Node):
    """Return the hyperbolic sine of x."""

    title = 'SinhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sinh(self.input(0)))
class SqrtNode(Node):
    """Return the square root of x."""

    title = 'SqrtNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sqrt(self.input(0)))
class TanNode(Node):
    """Return the tangent of x (measured in radians)."""

    title = 'TanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, tan(self.input(0)))
class TanhNode(Node):
    """Return the hyperbolic tangent of x."""

    title = 'TanhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, tanh(self.input(0)))

"""
WARNING: Module TauNode was generated using fallback option. May contain bugs
"""

class TauNode(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'TauNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, tau(self.input(0), self.input(1)))
class TruncNode(Node):
    """Truncates the Real x to the nearest Integral toward 0.

Uses the __trunc__ magic method."""

    title = 'Truncate'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, trunc(self.input(0)))
math_nodes = [
    AcosNode, AcoshNode,AsinNode,AtanNode,Atan2Node,AtanhNode,CeilNode,CopysignNode,CosNode,CoshNode,
    ENode,ErfNode,ErfcNode,ExpNode,Expm1Node,FabsNode,FactorialNode,FloorNode,FmodNode,FrexpNode,FsumNode,GammaNode,GcdNode,HypotNode,
    InfNode,IscloseNode,IsfiniteNode,IsinfNode,IsnanNode,LdexpNode,LgammaNode,
    LogNode,Log10Node,Log1pNode,Log2Node,ModfNode,
    NanNode,
    PiNode,PowNode,RadiansNode,RemainderNode,SinNode,SinhNode,SqrtNode,TanNode,TanhNode,
    TauNode,TruncNode
]
export_nodes(
    *math_nodes
)
