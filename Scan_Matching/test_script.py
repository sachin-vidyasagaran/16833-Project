# from build_NDT import *
# from newton_optim import *
import numpy as np
import csv

def main():
    with open("NDTtransforms.csv", "w", newline="") as f:
        writer = csv.writer(f)
        a = np.arange(50).reshape((5,10))
        for i in range(1,5):
            writer.writerow(a[i,:])
   


if __name__ == "__main__":
    main()