from pyrr import Vector3


class Light:
    """
    Implements lighting.
    """
    def __init__(self, position: Vector3, color: Vector3) -> None:
        """
        Constructor.
        :param position: Light position.
        :param color: Light color.
        """
        self.position = position
        self.color = color
        # Just some random values for lights
        self.Ia = 0.2 * self.color
        self.Id = 0.9 * self.color
        self.Is = 0.5 * self.color
