import math
from numbers import Number
import numpy as np

class Quaternion(object):

    def __init__(self, x, y, z, w):
        self._quaternion = np.array([x, y, z, w])

    @staticmethod
    def from_imaginary_and_real(imaginary, real):
        return Quaternion(*imaginary, w=real)

    @property
    def real(self):
        return self._quaternion[3]

    @property
    def imaginary(self):
        return self._quaternion[:3]

    @property
    def x(self):
        return self._quaternion[0]

    @property
    def y(self):
        return self._quaternion[1]

    @property
    def z(self):
        return self._quaternion[2]

    @property
    def w(self):
        return self._quaternion[3]

    def __repr__(self):
        return f"Quaternion({self.x}i + {self.y}j + {self.z}k + {self.w})"

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Quaternion(*self._quaternion)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                *(self._quaternion + other._quaternion)
            )

        raise TypeError(
                f"{type(other)} isn't an instance of Quaternion"
            )

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            real = self.w * other.w - np.inner(
                self.imaginary, other.imaginary
            )
            imaginary = np.cross(
                self.imaginary, other.imaginary
            ) + self.w * other.imaginary + other.w * self.imaginary
            return Quaternion.from_imaginary_and_real(
                imaginary,
                real
            )
        if isinstance(other, Number):
            return Quaternion(*(other * self._quaternion))
        raise TypeError(
            f"Multiplication {type(other)} with Quaternion isn't supported"
        )

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Quaternion(*(other * self._quaternion))
        raise TypeError(
            f"RMUL of Quaternion and {type(other)} unsupported"
        )

    def __neg__(self):
        return Quaternion(*(-self._quaternion))

    def conjugate(self):
        return Quaternion.from_imaginary_and_real(
           -self.imaginary, self.real
        )

    def norm(self):
        return np.linalg.norm(self._quaternion)

    def normalized(self):
        return Quaternion(*(self._quaternion / self.norm()))

    @staticmethod
    def from_axis_angle(axis, angle):
        if angle == 0:
            return Quaternion(0, 0, 0, 1)
        w = math.cos(angle / 2)
        normalized_axis = axis / np.linalg.norm(axis)
        imaginary = math.sin(angle / 2) * normalized_axis
        return Quaternion(*imaginary, w)

    def to_axis_angle(self):
        q_normed = self.normalized()
        angle = 2 * math.acos(self.w)
        if math.isclose(math.fabs(self.w), 1):  # identity
            p = np.array([1, 0, 0])  # any unit vec
        else:
            p = self.imaginary / np.linalg.norm(self.imaginary)
        return (p, angle)

    @staticmethod
    def slerp(q1, q2, t, t_max):
        cos_0 = np.inner(q1._quaternion, q2._quaternion)
        if cos_0 < 0:
            q1 = -q1
            cos_0 = -cos_0
        if cos_0 > 0.95:
            return q1
        tfactor = t / t_max
        phi_0 = math.acos(cos_0)
        q1_factor = math.sin(phi_0 * (1 - tfactor)) / math.sin(phi_0)
        q2_factor = math.sin(phi_0 * tfactor) / math.sin(phi_0)
        return q1_factor * q1 + q2_factor * q2

