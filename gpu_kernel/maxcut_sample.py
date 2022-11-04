import time
import numpy as np
import pickle
from DA4 import DA
from glob import glob
from tqdm import tqdm


def bqp_test(path="../data/g05/", file="g05", iters=1000, spin=False, b_matrix=False):
    data_list = [tag for tag in glob("{}/{}*".format(path, file))]
    solution_file = open("{}/solution.txt".format(path), "r").readlines()
    solution = {}
    for s in solution_file:
        f, v = s.split(" ")
        if b_matrix == True :
            solution[f] = int(v)
        else :
            solution[f] = -int(v)

    output_file = open("{}hist{}.txt".format(file, iters), "w")
    data_list = np.sort(data_list)
    run=100
    for data in tqdm(data_list):
        sv = solution[data.split("/")[-1]]
        distribution = np.zeros(10)
        t = 0
        min_e = 0
        f = open(data, "r").readlines()

        bin_size = f[0].split(" ")[0]
        Q = np.zeros([int(bin_size) + 1, int(bin_size) + 1])
        init_bin = np.zeros([int(bin_size) + 1])
        init_bin[-1] = 1
        for ele in f[1:]:
            i, j, v = ele.split()
            if b_matrix == True :
                Q[int(i) - 1, int(j) - 1] += int(v)
                if (int(i) != int(j)) :
                    Q[int(j) - 1, int(i) - 1] += int(v)
            else :
                if (int(i) == int(j)) :
                    print('No edge connected at the same Node',int(i),int(j))
                else :
                    Q[int(i) - 1, int(j) - 1] += int(v)
                    Q[int(j) - 1, int(i) - 1] += int(v)
                    Q[int(i) - 1, int(i) - 1] += -int(v)
                    Q[int(j) - 1, int(j) - 1] += -int(v)

        for n in tqdm(range(run)):
   
            #### annealing ####
            init_e = init_bin @ Q
            da1 = DA(Q, init_bin, maxStep=iters,
                     betaStart=0.01, betaStop=1.61, kernel_dim=(32 * 2,), spin=spin, energy = init_e)
            da1.run()
            # print(da1.binary)
            # print(f'time spent: {da1.time}')

            bin1 = np.expand_dims(da1.binary, axis=1)
            output = np.matmul(np.matmul(bin1.T, Q), bin1)[0][0]
            print("solution", sv, output)

            if output < min_e :
                min_e = output

            if (sv - output) >= 0:
                l = 0
                distribution[0] += 1
                t += da1.time
                print('find optimal solution', sv, output)
                break
            else:
                l = int(np.ceil((abs((sv - output) / sv) * 100)))
            if l > 9:
                l = 9
            distribution[l] += 1
            t += da1.time
    
        output_file.write("{}\t{}\t{}\t{}\n".format(data.split("/")[-1].split(".")[0], min_e, t, distribution))
        output_file.flush()


if __name__ == '__main__':

    bqp_test(iters=10000)
    bqp_test(path="../data/beasley/", file="bqp", iters=40000, b_matrix=True)
