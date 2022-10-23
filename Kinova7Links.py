from Transformations import *
from functools import reduce
from torch import Tensor
from typing import List
import torch as th


class Kinova7Links:
    def __init__(self, config=Tensor([0 for _ in range(7)])):
        self.config = config
        self.joints = None

    def forward_kinematics(self, config: Tensor = None, decompose=True):
        config = self.config if config is None else config
        assert config.shape == th.Size([7]), f"Kinova Manipulator has only 7 joints, not {config.shape}"
        q1, q2, q3, q4, q5, q6, q7 = config

        T01: Tensor = TTz(0.333) @ TRz(q1)
        T12: Tensor = TRy(q2)
        T23: Tensor = TTz(0.316) @ TRz(q3)
        T34: Tensor = TTx(0.0825) @ TRy(-q4)
        T45: Tensor = TTx(-0.0825) @ TTz(0.384) @ TRz(q5)
        T56: Tensor = TRy(-q6)
        T67: Tensor = TTz(0.088) @ TRx(th.pi) @ TTz(0.107) @ TRz(q7)

        links: List[Tensor] = [T01, T12, T23, T34, T45, T56, T67]
        self.joints = [reduce(th.matmul, links[:i]) for i in range(1, 7)]

        FK: Tensor = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67

        if not decompose:
            return th.round(FK, decimals=4)

        R, P = T2RP(FK)
        euler_angles = R2Euler(R)

        return th.round(P, decimals=4), th.round(euler_angles, decimals=4)

    def inverse_kinematics(self):
        pass

    def jacobian(self):
        pass

    def forward_dynamics(self):
        pass

    def inverse_dynamics(self):
        pass


if __name__ == '__main__':
    manipulator = Kinova7Links()
    print(manipulator.forward_kinematics())
    print(manipulator.joints)
