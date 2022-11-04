import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_bool, c_float, c_int, cdll, POINTER
import time


class DA:

    '''
    Attributes:
    qubo : np.ndarray
        the qubo matrix in 2D
    binary : np.ndarray
        the initial spin in 1D
    maxStep : int
        the maximum steps for the algorithm
    dim : int
        the dimention of the spin array
    time : float
        the time spent on the last execution of the algorithm. default value 0 is set
    '''

    def __init__(
        self,
        qubo: np.ndarray = np.array([[0, 1], [1, 0]]),
        binary: np.ndarray = None,
        maxStep: int = 10000,
        betaStart: float = 0.01,
        betaStop: float = 100,
        kernel_dim: tuple = (32*16,),
        spin: bool = False,
        bin_label = None,
        act_idx = None,
        label_num = 1,
        tree_n = 3,
        energy = None
    ) -> None:
        '''
        Parameters:
        qubo : np.ndarray
            the qubo matrix in 2D. elements will be parsed to np.float32 which is equivalent to "float" in C. default qubo matrix [[0,1],[1,0]] is used.
        binary : np.ndarray | None
            the initial spin in 1D with values between {-1,1}. elements will be parsed to np.float32 which is equivalent to "float" in C. if none then a random initial spin is generated
        maxStep : int
            the maximum steps for the algorithm. default value 10,000 is used
        betaStart : float
        betaStop : float
        time
        energy
        '''
        self.qubo = qubo.astype(np.float32)
        self.maxStep = maxStep
        self.betaStart = betaStart
        self.betaStop = betaStop
        self.act_idx = act_idx
        self.label_num = label_num
        self.spin = spin
        self.tree_n = tree_n
        self.dim = np.shape(self.qubo)[0]
        if energy is not None:
            self.energy = energy.astype(np.float32)
        if act_idx is not None:
            self.act_idx = self.act_idx.astype(np.int32)
        else:
            self.act_idx = np.zeros(2)
            self.act_idx = self.act_idx.astype(np.int32)
        if bin_label is None:
            self.bin_label = np.zeros(self.dim)
            self.bin_label = self.bin_label.astype(np.int32)
        else:
            self.bin_label = bin_label
            self.bin_label = self.bin_label.astype(np.int32)

        if np.shape(self.qubo)[0] != np.shape(self.qubo)[1]:
            # print("qubo is not a square matrix")
            exit(-1)
        

        if(binary is None):
            self.binary = np.zeros(self.dim)+1
            self.binary[-1] = 1
            self.binary = self.binary.astype(np.int32)
        else:
            self.binary = binary.astype(np.int32)

        if np.shape(self.qubo)[0] != np.shape(self.binary)[0]:
            # print("qubo dimention and binary dimention mismatch")
            exit(-1)
        self.time = 0
        #self.energy = 0

        if len(kernel_dim) == 1:
            if kernel_dim[0] == 0:
                # print(f"grid size cannot be 0. Using default grid size.")
                kernel_dim[0] = 32*16
            self.blocks = kernel_dim[0]
            self.threads = self.dim//self.blocks + 1
            # print(f"grid size = {self.blocks} assigned.")
        elif len(kernel_dim) == 2:
            if any(kernel_dim) == 0:
                # print(f'grid size and block size cannot be 0. Using default grid size.')
                kernel_dim[0] = 32*16
                kernel_dim[1] = self.dim//kernel_dim[0] + 1
            self.blocks = kernel_dim[0]
            self.threads = kernel_dim[1]
            # print(
            #     f"grid size {self.blocks} assigned, block size {self.threads} assigned.")
        else:
            # print('kernel_dim has to be a tuple of length 2. Using default grid size.')
            self.blocks = 32*16
            self.threads = self.dim//self.blocks + 1

        
    def run(self) -> None:

        binary = ctplib.as_ctypes(self.binary)
        qubo = ctplib.as_ctypes(self.qubo.flatten())
        bin_label = ctplib.as_ctypes(self.bin_label)
        act_idx = ctplib.as_ctypes(self.act_idx)
        energy = ctplib.as_ctypes(self.energy)

        start = time.time()

        ################################## old version ##################################
        # da = cdll.LoadLibrary("./lib/cudaDA.so")
        # main = da.digitalAnnealing
        # main.restype = c_float
        # main.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(
        #     c_float), c_int, c_int, c_float, c_float, c_int, c_int,
        #                  c_bool, c_int]
        # main(binary, bin_label, act_idx, qubo, self.dim, self.maxStep, self.betaStart,
        #      self.betaStop, self.blocks, self.threads, self.spin, self.label_num)
        #################################################################################

        da = cdll.LoadLibrary("./lib/DA4.so")
        main = da.digitalAnnealing
        main.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(
            c_float), POINTER(c_float), c_int, c_int, c_float, c_float, c_int, c_int,
            c_bool, c_int]
        main.restype = c_float
        main(binary, bin_label, act_idx, qubo, energy, self.dim, self.maxStep, self.betaStart,
             self.betaStop, self.blocks, self.threads, self.spin, self.label_num)

        end = time.time()

        self.time = end-start

        self.binary = ctplib.as_array(binary)


if __name__ == '__main__':
    def load_qmatrix(file="small_sample_qubo_1.txt"):
        f = open(file, "r")
        lines = f.readlines()
        _, _, _, q_len, _, _ = lines[1].split(" ")
        q_len = int(q_len)
        q = np.zeros([q_len, q_len])
        for i in range(2, len(lines)):
            idx, idy, val = lines[i].split(" ")
            q[int(idx), int(idy)] = float(val)
        return q, q_len

    min_e = 0
    for n in range(200):
        np.random.seed(1)
        # dim = 713
        maxStep = 10000
        # qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
        qubo, dim = load_qmatrix()
        qubo = (qubo + qubo.T) / 2 /1000
        init_bin = np.ones(dim).astype(np.float32)
        init_e = init_bin @ qubo

        da = DA(qubo, init_bin, maxStep, energy=init_e, betaStop=40)
        da.run()
        # print(da.time)
        # print(da.binary)
        print("energy : {}".format(da.binary.T @ qubo @ da.binary))
        e_tmp = da.binary.T @ qubo @ da.binary
        if e_tmp < min_e:
            min_e = e_tmp
            bin_result = da.binary
    print(bin_result)
    print("energy : {}".format(bin_result.T @ qubo @ bin_result))
