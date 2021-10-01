
from .. import base
from .Vector2VectorAttention import Vector2VectorAttention

class LabeledVectorAttention(base.LabeledVectorAttention, Vector2VectorAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type(self) == LabeledVectorAttention:
            self.init()
