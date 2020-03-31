import numpy
import math

"""W = Image dimensions, F = Filter size, P = Padding, S = Stride"""


def inputlayerdim(w_out, f, p, s):
    w_in = ((w_out - 1) * s) - (2 * p) + f
    return w_in


def magic():
    for w_out in range(1, 20):
        w_1 = inputlayerdim(w_out, 3, 0, 1)
        w_2 = inputlayerdim(w_1, 4, 0, 2)
        w_3 = inputlayerdim(w_2, 8, 0, 4)
        print(w_3)

if __name__ == '__main__':
    magic()