# coding: utf-8

from random import shuffle


def verctor_add(a, b):
    """
    two vector add
    a: list
    b: list
    return: list
    """
    return [a[i]+b[i] for i in range(len(a))]


def dot_prod(a, b):
    """
    two vector dot product
    a: list
    b: list
    return: float
    """
    return sum([a[i]*b[i] for i in range(len(a))])


def scale_vector(v, s):
    """
    vector scales
    v: list
    s: int
    return: list
    """
    return map(lambda x: x*s, v)


def chunks(l, n):
    """
    split list into n fold
    l: list
    n: int
    return: list[tuple]
    """
    shuffle(l)
    if not len(l) % n:
        size = len(l)//n
    else:
        size = len(l)//n + 1
    for i in xrange(0, len(l), size):
        yield (l[:i-1]+l[i+size:], l[i:i+size])
