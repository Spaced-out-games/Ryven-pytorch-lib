from ryven import *

from math import *

class MathAcosNode_Node(Node):
    """Return the arc cosine (measured in radians) of x."""

    title = 'MathAcosNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x')
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, acos(self.input(0)))
class MathAcoshNode_Node(Node):
    """Return the inverse hyperbolic cosine of x."""

    title = 'MathAcoshNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, acosh(self.input(0)))
class MathAsinNode_Node(Node):
    """Return the arc sine (measured in radians) of x."""

    title = 'MathAsinNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, asin(self.input(0)))
class MathAsinhNode_Node(Node):
    """Return the inverse hyperbolic sine of x."""

    title = 'MathAsinhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, asinh(self.input(0)))
class MathAtanNode_Node(Node):
    """Return the arc tangent (measured in radians) of x."""

    title = 'MathAtanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, atan(self.input(0)))
class MathAtan2Node_Node(Node):
    """Return the arc tangent (measured in radians) of y/x.

Unlike atan(y/x), the signs of both x and y are considered."""

    title = 'MathAtan2Node'
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
class MathAtanhNode_Node(Node):
    """Return the inverse hyperbolic tangent of x."""

    title = 'MathAtanhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, atanh(self.input(0)))
class MathCeilNode_Node(Node):
    """Return the ceiling of x as an Integral.

This is the smallest integer >= x."""

    title = 'MathCeilNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, ceil(self.input(0)))
class MathCopysignNode_Node(Node):
    """Return a float with the magnitude (absolute value) of x but the sign of y.

On platforms that support signed zeros, copysign(1.0, -0.0)
returns -1.0.
"""

    title = 'MathCopysignNode'
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
class MathCosNode_Node(Node):
    """Return the cosine of x (measured in radians)."""

    title = 'MathCosNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, cos(self.input(0)))
class MathCoshNode_Node(Node):
    """Return the hyperbolic cosine of x."""

    title = 'MathCoshNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, cosh(self.input(0)))
class MathDegreesNode_Node(Node):
    """Convert angle x from radians to degrees."""

    title = 'MathDegreesNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, degrees(self.input(0)))

"""
WARNING: Module MathENode was generated using fallback option. May contain bugs
"""

class MathENode_Node(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'MathENode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, e(self.input(0), self.input(1)))
class MathErfNode_Node(Node):
    """Error function at x."""

    title = 'MathErfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, erf(self.input(0)))
class MathErfcNode_Node(Node):
    """Complementary error function at x."""

    title = 'MathErfcNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, erfc(self.input(0)))
class MathExpNode_Node(Node):
    """Return e raised to the power of x."""

    title = 'MathExpNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, exp(self.input(0)))
class MathExpm1Node_Node(Node):
    """Return exp(x)-1.

This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x."""

    title = 'MathExpm1Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, expm1(self.input(0)))
class MathFabsNode_Node(Node):
    """Return the absolute value of the float x."""

    title = 'MathFabsNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, fabs(self.input(0)))
class MathFactorialNode_Node(Node):
    """Find x!.

Raise a ValueError if x is negative or non-integral."""

    title = 'MathFactorialNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, factorial(self.input(0)))
class MathFloorNode_Node(Node):
    """Return the floor of x as an Integral.

This is the largest integer <= x."""

    title = 'MathFloorNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, floor(self.input(0)))
class MathFmodNode_Node(Node):
    """Return fmod(x, y), according to platform C.

x % y may differ."""

    title = 'MathFmodNode'
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
class MathFrexpNode_Node(Node):
    """Return the mantissa and exponent of x, as pair (m, e).

m is a float and e is an int, such that x = m * 2.**e.
If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0."""

    title = 'MathFrexpNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, frexp(self.input(0)))
class MathFsumNode_Node(Node):
    """Return an accurate floating point sum of values in the iterable seq.

Assumes IEEE-754 floating point arithmetic."""

    title = 'MathFsumNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'seq'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, fsum(self.input(0)))
class MathGammaNode_Node(Node):
    """Gamma function at x."""

    title = 'MathGammaNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, gamma(self.input(0)))
class MathGcdNode_Node(Node):
    """greatest common divisor of x and y"""

    title = 'MathGcdNode'
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
class MathHypotNode_Node(Node):
    """Return the Euclidean distance, sqrt(x*x + y*y)."""

    title = 'MathHypotNode'
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
WARNING: Module MathInfNode was generated using fallback option. May contain bugs
"""
class MathInfNode_Node(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'MathInfNode'
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
class MathIscloseNode_Node(Node):
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

    title = 'MathIscloseNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isclose(self.input(0), self.input(1)))
class MathIsfiniteNode_Node(Node):
    """Return True if x is neither an infinity nor a NaN, and False otherwise."""

    title = 'MathIsfiniteNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isfinite(self.input(0)))
class MathIsinfNode_Node(Node):
    """Return True if x is a positive or negative infinity, and False otherwise."""

    title = 'MathIsinfNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isinf(self.input(0)))
class MathIsnanNode_Node(Node):
    """Return True if x is a NaN (not a number), and False otherwise."""

    title = 'MathIsnanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, isnan(self.input(0)))
class MathLdexpNode_Node(Node):
    """Return x * (2**i).

This is essentially the inverse of frexp()."""

    title = 'MathLdexpNode'
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
class MathLgammaNode_Node(Node):
    """Natural logarithm of absolute value of Gamma function at x."""

    title = 'MathLgammaNode'
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
WARNING: Module MathLogNode was generated using fallback option. May contain bugs
"""

class MathLogNode_Node(Node):
    """log(x, [base=math.e])
Return the logarithm of x to the given base.

If the base not specified, returns the natural logarithm (base e) of x."""

    title = 'MathLogNode'
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
class MathLog10Node_Node(Node):
    """Return the base 10 logarithm of x."""

    title = 'MathLog10Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log10(self.input(0)))
class MathLog1pNode_Node(Node):
    """Return the natural logarithm of 1+x (base e).

The result is computed in a way which is accurate for x near zero."""

    title = 'MathLog1pNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log1p(self.input(0)))
class MathLog2Node_Node(Node):
    """Return the base 2 logarithm of x."""

    title = 'MathLog2Node'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, log2(self.input(0)))
class MathModfNode_Node(Node):
    """Return the fractional and integer parts of x.

Both results carry the sign of x and are floats."""

    title = 'MathModfNode'
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
WARNING: Module MathNanNode was generated using fallback option. May contain bugs
"""

class MathNanNode_Node(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'MathNanNode'
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
WARNING: Module MathPiNode was generated using fallback option. May contain bugs
"""

class MathPiNode_Node(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'MathPiNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'a'),
NodeInputBP(dtype=dtypes.Data(default=1), label = 'b'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, pi(self.input(0), self.input(1)))
class MathPowNode_Node(Node):
    """Return x**y (x to the power of y)."""

    title = 'MathPowNode'
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
class MathRadiansNode_Node(Node):
    """Convert angle x from degrees to radians."""

    title = 'MathRadiansNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, radians(self.input(0)))
class MathRemainderNode_Node(Node):
    """Difference between x and the closest integer multiple of y.

Return x - n*y where n*y is the closest integer multiple of y.
In the case where x is exactly halfway between two multiples of
y, the nearest even value of n is used. The result is always exact."""

    title = 'MathRemainderNode'
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
class MathSinNode_Node(Node):
    """Return the sine of x (measured in radians)."""

    title = 'MathSinNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sin(self.input(0)))
class MathSinhNode_Node(Node):
    """Return the hyperbolic sine of x."""

    title = 'MathSinhNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sinh(self.input(0)))
class MathSqrtNode_Node(Node):
    """Return the square root of x."""

    title = 'MathSqrtNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, sqrt(self.input(0)))
class MathTanNode_Node(Node):
    """Return the tangent of x (measured in radians)."""

    title = 'MathTanNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, tan(self.input(0)))
class MathTanhNode_Node(Node):
    """Return the hyperbolic tangent of x."""

    title = 'MathTanhNode'
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
WARNING: Module MathTauNode was generated using fallback option. May contain bugs
"""

class MathTauNode_Node(Node):
    """Convert a string or number to a floating point number, if possible."""

    title = 'MathTauNode'
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
class MathTruncNode_Node(Node):
    """Truncates the Real x to the nearest Integral toward 0.

Uses the __trunc__ magic method."""

    title = 'MathTruncNode'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Data(default=1), label = 'x'),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aaaaaa'

    def update_event(self, inp=-1):
        self.set_output_val(0, trunc(self.input(0)))
#acos acosh asin atan atan2 atanh ceil copysign cos cosh
#E erf erfc exp expm1 fabs factorial floor fmod frexp fsum gamma gcd hypot
#inf isclose isfinite isinf isnan ldexp lgamma
#log log10 log1p log2 modf
#nan
#pi pow radians remainder sin sinh sqrt tan tanh
#tau trunc
math_nodes = [
    MathAcosNode_Node, MathAcoshNode_Node,MathAsinNode_Node,MathAtanNode_Node,MathAtan2Node_Node,MathAtanhNode_Node,MathCeilNode_Node,MathCopysignNode_Node,MathCosNode_Node,MathCoshNode_Node,
    MathENode_Node,MathErfNode_Node,MathErfcNode_Node,MathExpNode_Node,MathExpm1Node_Node,MathFabsNode_Node,MathFactorialNode_Node,MathFloorNode_Node,MathFmodNode_Node,MathFrexpNode_Node,MathFsumNode_Node,MathGammaNode_Node,MathGcdNode_Node,MathHypotNode_Node,
    MathInfNode_Node,MathIscloseNode_Node,MathIsfiniteNode_Node,MathIsinfNode_Node,MathIsnanNode_Node,MathLdexpNode_Node,MathLgammaNode_Node,
    MathLogNode_Node,MathLog10Node_Node,MathLog1pNode_Node,MathLog2Node_Node,MathModfNode_Node,
    MathNanNode_Node,
    MathPiNode_Node,MathPowNode_Node,MathRadiansNode_Node,MathRemainderNode_Node,MathSinNode_Node,MathSinhNode_Node,MathSqrtNode_Node,MathTanNode_Node,MathTanhNode_Node,
    MathTauNode_Node,MathTruncNode_Node
]
export_nodes(
    *math_nodes
)