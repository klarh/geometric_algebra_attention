
class Namespace:
    def __init__(self, **kwargs):
        for (name, val) in kwargs.items():
            setattr(self, name, val)
