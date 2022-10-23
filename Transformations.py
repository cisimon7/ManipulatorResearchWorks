import torch as th
from torch import Tensor


# Elementary Transformations Sequence


def TTx(x: Tensor | float):
    if type(x) == Tensor:
        assert x.numel() == 1
    else:
        x = Tensor([x])
    return Tensor([
        [1, 0, 0, x],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def TTy(y: Tensor | float):
    if type(y) == Tensor:
        assert y.numel() == 1
    else:
        y = Tensor([y])
    return Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def TTz(z: Tensor | float):
    if type(z) == Tensor:
        assert z.numel() == 1
    else:
        z = Tensor([z])
    return Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def TRx(x: Tensor | float):
    if type(x) == Tensor:
        assert x.numel() == 1
    else:
        x = Tensor([x])
    return Tensor([
        [1, 0, 0, 0],
        [0, th.cos(x), -th.sin(x), 0],
        [0, th.sin(x), th.cos(x), 0],
        [0, 0, 0, 1]
    ])


def TRy(y: Tensor | float):
    if type(y) == Tensor:
        assert y.numel() == 1
    else:
        y = Tensor([y])
    return Tensor([
        [th.cos(y), 0, th.sin(y), 0],
        [0, 1, 0, 0],
        [-th.sin(y), 0, th.cos(y), 0],
        [0, 0, 0, 1]
    ])


def TRz(z: Tensor | float):
    if type(z) == Tensor:
        assert z.numel() == 1
    else:
        z = Tensor([z])
    return Tensor([
        [th.cos(z), -th.sin(z), 0, 0],
        [th.sin(z), th.cos(z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def dTRx(x: Tensor | float):
    if type(x) == Tensor:
        assert x.numel() == 1
    else:
        x = Tensor([x])
    return Tensor([
        [0, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]) @ TRx(x)


def dTRy(y: Tensor | float):
    if type(y) == Tensor:
        assert y.numel() == 1
    else:
        y = Tensor([y])
    return Tensor([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 0]
    ]) @ TRy(y)


def dTRz(z: Tensor | float):
    if type(z) == Tensor:
        assert z.numel() == 1
    else:
        z = Tensor([z])
    return Tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]) @ TRz(z)


def dTTx(x: Tensor | float):
    if type(x) == Tensor:
        assert x.numel() == 1
    else:
        x = Tensor([x])
    return Tensor([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])


def dTTy(y: Tensor | float):
    if type(y) == Tensor:
        assert y.numel() == 1
    else:
        y = Tensor([y])
    return Tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])


def dTTz(z: Tensor | float):
    if type(z) == Tensor:
        assert z.numel() == 1
    else:
        z = Tensor([z])
    return Tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])


def T2RP(T: Tensor):
    """
    Transformation split to Rotation matrix and translation vector
    :param T: Transformation matrix
    :return: Tuple of Rotation matrix and Translation vector
    """
    assert T.shape == th.Size([4, 4]), f"Matrix of shape {T.shape} is not a transformation matrix"
    T = T[:3, :]
    return T[:, :3], T[:, -1].reshape(3,-1)


def R2Euler(R: Tensor):
    """
    Transforms a rotation matrix to its Euler vector equivalent
    :param R: Rotation matrix
    :return: Euler vector equivalent
    """
    assert R.shape == th.Size([3, 3]), f"Matrix of shape {R.shape} is not a rotation matrix"
    diagonal = th.diagonal(R)
    is_diagonal = th.allclose(R, th.diag(diagonal))
    if is_diagonal:
        if diagonal == Tensor([1, 1, 1]):
            return th.transpose(Tensor([[0, 0, 0]]), 0, 1)
        else:
            return (th.pi / 2) * th.transpose(Tensor([[R[1, 1] + 1, R[2, 2] + 1, R[3, 3] + 1]]), 0, 1)

    l = th.transpose(Tensor([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]]), 0, 1)
    l_norm = th.norm(l)
    return (th.atan2(l_norm, R[0, 0] + R[1, 1] + R[2, 2]) / l_norm) * l
