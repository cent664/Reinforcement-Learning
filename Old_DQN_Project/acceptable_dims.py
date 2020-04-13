import numpy
import math

"""W = Image dimensions, F = Filter size, P = Padding, S = Stride"""


def inputlayerdim(w_out, f, p, s):
    w_in = ((w_out - 1) * s) - (2 * p) + f
    return w_in


def magic():
    for w_out in range(1, 20):
        w_1 = inputlayerdim(w_out, 3, 0, 1)
        w_2 = inputlayerdim(w_1, 4, 0, 1)
        w_3 = inputlayerdim(w_2, 8, 0, 1)
        print(w_3)


def outputlayerdim(w_in, f, p, s):
    w_out = 1 + (w_in - f + (2 * p))/s
    return w_out

if __name__ == '__main__':
    # magic()
    w_out = outputlayerdim(16, 8, 0, 1)
    print(w_out)
    w_out = outputlayerdim(w_out, 4, 0, 1)
    print(w_out)
    w_out = outputlayerdim(w_out, 3, 0, 1)
    print(w_out)