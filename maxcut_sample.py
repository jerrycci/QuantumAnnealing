import time
import numpy as np
import pickle
from DA4 import DA
from glob import glob
from tqdm import tqdm


def bqp_test(path="../data/g05/", file="g05", iters=1000, spin=False):
    data_list = [tag for tag in glob("{}/{}*".format(path, file))]
    solution_file = open("{}/solution.txt".format(path), "r").readlines()
    solution = {}
    for s in solution_file:
        f, v = s.split(" ")
        solution[f] = -int(v)

    output_file = open("{}hist{}.txt".format(file, iters), "w")
    data_list = np.sort(data_list)
    run=100
    for data in tqdm(data_list):
        sv = solution[data.split("/")[-1]]
        distribution = np.zeros(10)
        t = 0
        for n in tqdm(range(run)):
            f = open(data, "r").readlines()

            bin_size = f[0].split(" ")[0]
            Q = np.zeros([int(bin_size) + 1, int(bin_size) + 1])
            init_bin = np.zeros([int(bin_size) + 1])
            init_bin[-1] = 1
            for ele in f[1:]:
                i, j, v = ele.split()
                if (int(i) == int(j)) :
                    print('No edge connected at the same Node',int(i),int(j))
                else :
                    Q[int(i) - 1, int(j) - 1] += int(v)
                    Q[int(j) - 1, int(i) - 1] += int(v)
                    Q[int(i) - 1, int(i) - 1] += -int(v)
                    Q[int(j) - 1, int(j) - 1] += -int(v)

            #### annealing ####
            init_e = init_bin @ Q
            da1 = DA(Q, init_bin, maxStep=iters,
                     betaStart=0.01, betaStop=40, kernel_dim=(32 * 2,), spin=spin, energy = init_e)
            da1.run()
            # print(da1.binary)
            # print(f'time spent: {da1.time}')

            bin1 = np.expand_dims(da1.binary, axis=1)
            output = np.matmul(np.matmul(bin1.T, Q), bin1)[0][0]
            print("solution", sv, output)

            if (sv - output) >= 0:
                l = 0
                print('find optimal solution', sv, output)
            else:
                l = int(np.ceil((abs((sv - output) / sv) * 100)))
            if l > 9:
                l = 9
            distribution[l] += 1
            t += da1.time
        t /= 100
        distribution /= run
        output_file.write("{}\t{}\t{}\n".format(data.split("/")[-1].split(".")[0], t, distribution))


if __name__ == '__main__':

    bqp_test(iters=10000)
