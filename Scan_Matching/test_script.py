# from build_NDT import *
# from newton_optim import *
import numpy as np


def myfunc(pt, cos_phi, sin_phi):
    return np.array([[1,   0,   -pt[0]*sin_phi - pt[1]*cos_phi],
                     [0,   1,   pt[0]*cos_phi - pt[1]*sin_phi]])
def main():
    phi = np.pi/4
    curr = np.array([[1, 2],
                     [2, 3],
                     [5, 7]])
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    vfunc = np.vectorize(myfunc)
    print(vfunc(curr, sin_phi, cos_phi))
   


if __name__ == "__main__":
    main()