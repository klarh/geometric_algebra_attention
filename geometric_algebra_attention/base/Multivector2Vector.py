
class Multivector2Vector:
    """Convert a multivector representation into a 3D vector representation.

    This class simply strips out the non-vector components of the result.

    """

    @classmethod
    def _evaluate(cls, inputs):
        return inputs[..., 1:4]
