import time
import numpy as np
import pickle
import sys
from DA4 import DA
from glob import glob
from tqdm import tqdm



def qap_test(path="../data/qapdata/", file="nug12", iters=1000, spin=False):

    # Reads instance file and initializes flows, distances and number of nodes
    distances = np.array([])
    flows = np.array([])
    lists_for_matrix = []
    flag = "num_nodes"

    with open("{}.dat".format(path + file)) as data_file:
        for line in data_file:
            if line.strip():
                strip = list(map(int, line.split()))
                if len(strip) == 1 and flag == "num_nodes":
                    num_nodes = int(strip[0])
                    flag = "flows"
                else:
                    lists_for_matrix.append(strip)
                    if len(lists_for_matrix) == num_nodes and flag == "flows":
                        flows = np.array(lists_for_matrix)
                        lists_for_matrix = []
                        flag = "distances"
                    elif len(lists_for_matrix) == num_nodes and flag == "distances":
                        distances = np.array(lists_for_matrix)

    solution_file = open("{}.sln".format("../data/qapsoln/" + file))
    sol = solution_file.readline().split()
    best = float(sol[1])

    print("Distances:\n", distances)
    print("Flows: \n", flows)
    print("Number of nodes:", num_nodes)    
    print("Referance solution:", sol[1])    
    
    #build QUBO
    length_of_QUBO = num_nodes**2
    penalty = distances.max() * flows.max() * num_nodes + 1
    Q = np.zeros((length_of_QUBO+1,length_of_QUBO+1))

    # Optimization Function (add distances and flows)
    # Add distances and flows
    dist_x = 0
    dist_y = 0
    for v in range(0, length_of_QUBO):
        for j in range(v, length_of_QUBO):
            if j % num_nodes == 0 and v != j:
                dist_y += 1
            if v % num_nodes == 0 and j == v and v != 0:
                dist_x += 1
            Q[v][j] = Q[v][j] + distances[dist_x][dist_y] * flows[v % num_nodes][j % num_nodes]
            Q[j][v] = Q[v][j]

            if j == length_of_QUBO-1:
                dist_y = dist_x
                if v % num_nodes == num_nodes-1:
                    dist_y += 1

    # Constraint that each facility assigned to one location only
    for v in range(0, length_of_QUBO):
        for j in range(v, length_of_QUBO):
            if v == j:
                Q[v][j] = Q[v][j] + (-1.0) * penalty
            else:
                if int(j / num_nodes) == int(v / num_nodes):
                    Q[v][j] = Q[v][j] + 2.0 * penalty
                    Q[j][v] = Q[v][j]
    Q[length_of_QUBO][length_of_QUBO] += num_nodes*penalty

    # Constraint that every each location is assigned only one facility
    for v in range(0, length_of_QUBO):
        for j in range(v, length_of_QUBO):
            if v == j:
                Q[v][j] = Q[v][j] + (-1.0) * penalty
            else:
                if int(v%num_nodes) == int(j%num_nodes):
                    Q[v][j] = Q[v][j] + 2.0 * penalty
                    Q[j][v] = Q[v][j]
            
    Q[length_of_QUBO][length_of_QUBO] += num_nodes*penalty

    #### annealing ####
    run = 10;
    global_e = sys.float_info.max
    for n in range(run):
        init_bin = np.zeros([int(length_of_QUBO) + 1])
        init_bin[-1] = 1    
        init_e = init_bin @ Q
        da1 = DA(Q, init_bin, maxStep=iters,
             betaStart=0.01, betaStop=1.61, kernel_dim=(32 * 2,), spin=spin, energy = init_e)
        da1.run()

        bin1 = np.expand_dims(da1.binary, axis=1)
        output = np.matmul(np.matmul(bin1.T, Q), bin1)[0][0]/2
        if output < global_e :
            global_e = output
            ans_arr = np.delete(da1.binary,-1)
        if global_e <= best :
            break
    print("solution", output)
    ans_arr = np.reshape(ans_arr, (num_nodes, num_nodes))
    print(ans_arr)


if __name__ == '__main__':

    qap_test(iters=40000)
