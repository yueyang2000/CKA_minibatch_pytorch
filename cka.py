from typing import Sequence
from torch import Tensor
import torch
from torch.nn import Module


def centering(k: Tensor, inplace: bool = True) -> Tensor:
    if not inplace:
        k = torch.clone(k)
    means = k.mean(dim=0)
    means -= means.mean() / 2
    k -= means.view(-1, 1)
    k -= means.view(1, -1)
    return k


# def centering(k: Tensor) -> Tensor:
#     m = k.shape[0]
#     h = torch.eye(m) - torch.ones(m, m) / m
#     return torch.matmul(h, torch.matmul(k, h))


def linear_hsic(k: Tensor, l: Tensor, unbiased: bool = True) -> Tensor:
    assert k.shape[0] == l.shape[0], 'Input must have the same size'
    m = k.shape[0]
    if unbiased:
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        kl = torch.matmul(k, l)
        score = torch.trace(kl) + k.sum() * l.sum() / ((m - 1) * (m - 2)) - 2 * kl.sum() / (m - 2)
        return score / (m * (m - 3))
    else:
        k, l = centering(k), centering(l)
        return (k * l).sum() / ((m - 1) ** 2)


def cka_score(x1: Tensor, x2: Tensor, gram: bool = False) -> Tensor:
    assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
    if not gram:
        x1 = torch.matmul(x1, x1.transpose(0, 1))
        x2 = torch.matmul(x2, x2.transpose(0, 1))
    cross_score = linear_hsic(x1, x2)
    self_score1 = linear_hsic(x1, x1)
    self_score2 = linear_hsic(x2, x2)
    return cross_score / torch.sqrt(self_score1 * self_score2)


class CKA_Minibatch(Module):
    """
    Minibatch Centered Kernel Alignment
    Reference: https://arxiv.org/pdf/2010.15327
    """

    def __init__(self):
        super().__init__()
        self.total = 0
        self.cross_hsic, self.self_hsic1, self.self_hsic2 = [], [], []

    def reset(self):
        self.total = 0
        self.cross_hsic, self.self_hsic1, self.self_hsic2 = [], [], []

    def update(self, x1: Tensor, x2: Tensor, gram: bool = False) -> None:
        """
            gram: if true, the method takes gram matrix as input
        """
        assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
        self.total += 1
        if not gram:
            x1 = torch.matmul(x1, x1.transpose(0, 1))
            x2 = torch.matmul(x2, x2.transpose(0, 1))
        self.cross_hsic.append(linear_hsic(x1, x2))
        self.self_hsic1.append(linear_hsic(x1, x1))
        self.self_hsic2.append(linear_hsic(x2, x2))

    def compute(self) -> Tensor:
        assert self.total > 0, 'Please call method update(x1, x2) first!'
        cross_score = sum(self.cross_hsic) / self.total
        self_score1 = sum(self.self_hsic1) / self.total
        self_score2 = sum(self.self_hsic2) / self.total
        return cross_score / torch.sqrt(self_score1 * self_score2)


class CKA_Minibatch_Grid(Module):
    '''
    Compute CKA for a 2D grid of features
    '''

    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.cka_loggers = [[CKA_Minibatch() for _ in range(dim2)] for _ in range(dim1)]
        self.dim1 = dim1
        self.dim2 = dim2

    def reset(self):
        for i in range(self.dim1):
            for j in range(self.dim2):
                self.cka_loggers[i][j].reset()

    def update(self, x1: Sequence[Tensor], x2: Sequence[Tensor], gram: bool = False) -> None:
        assert len(x1) == self.dim1, 'Grid dim0 mismatch'
        assert len(x2) == self.dim2, 'Grid dim1 mismatch'
        if not gram:
            x1 = [torch.matmul(x, x.transpose(0, 1)) for x in x1]
            x2 = [torch.matmul(x, x.transpose(0, 1)) for x in x2]
        for i in range(self.dim1):
            for j in range(self.dim2):
                self.cka_loggers[i][j].update(x1[i], x2[j], gram=True)

    def compute(self) -> Tensor:
        result = torch.zeros(self.dim1, self.dim2)
        for i in range(self.dim1):
            for j in range(self.dim2):
                result[i, j] = self.cka_loggers[i][j].compute()
        return result
