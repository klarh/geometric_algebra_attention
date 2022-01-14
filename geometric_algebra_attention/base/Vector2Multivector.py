
class Vector2Multivector:
    """Convert a 3D vector representation into a multivector representation.

    Pads each input vector on the left with 1 zero (the scalar
    component) and the right with 4 zeros (the bivector components and
    trivector component).

    """

    @classmethod
    def _evaluate(cls, inputs):
        scalar = trivec = cls.math.zeros_like(inputs[..., :1])
        bivec = cls.math.zeros_like(inputs)
        result = cls.math.concat([scalar, inputs, bivec, trivec], axis=-1)
        return result
