import time
import numpy as np
import pickle
from gpu_kernel.DA4 import DA
from glob import glob
from tqdm import tqdm


class QUBO:
    @staticmethod
    def params2qubo_tp_v3(rsrp, capacity, rb, serving_list, ue_num, bs_num, max_rb=273, spin=False):
        def vecmul(vec1, vec2, result=None):
            l1 = len(vec1)
            l2 = len(vec2)
            if result is None:
                result = np.zeros([l1, l2])
            for i in range(l1):
                if vec1[i] == 0:
                    continue
                result[i] += vec1[i] * vec2
            return result

        def int2coe(n, bit_size=8, min=0, max=200):
            assert n >= min, "int2bit : int n nust >= {}".format(min)
            assert n <= max, "int2bit : int n nust <= {}".format(max)
            bits = np.zeros([bit_size])
            i = -1
            if n == 0:
                return bits, i + 1
            else:
                c = 1
                while n - c >= 0:
                    i += 1
                    bits[i] = c
                    n -= c
                    c *= 2

                if n > 0:
                    i += 1
                    bits[i] = n
                return bits, i + 1

        max_rb = 200
        x_dim = ue_num * bs_num

        # rb fo each ue, must under 200, 2^8 > 200
        d_dim = ue_num * bs_num * 8

        # 1. constrain of numbers fo ue conneting to each bs are not greater than 128
        power128 = 7 + 1  # 2^8 > 128
        r_dim = bs_num * power128

        # 2. constrain 2 has no slack variable

        # 3. constrain of demand-throughput >= 0
        power16 = 3 + 1  # 2^4 > 8
        s_dim = ue_num * bs_num * power16

        # 4. constrain of maximum number of bs is 273
        power256 = 8 + 1  # 2^9 > 273
        u_dim = bs_num * power256

        # 5. handover > CIO
        ##################################################
        ### index of e is definded j * j'(j' != j) * 6 ###
        ### 0, [1,2,3,4,5,6], [:]                      ###
        ### 1, [0,2,3,4,5,6], [:]                      ###
        ### 2, [0,1,3,4,5,6], [:]                      ###
        ### ...                                        ###
        ##################################################
        cio_range_dim = 5  # 5 for len of [1, 2, 4, 8, 5]
        power64 = 6 + 1
        e_dim = bs_num * bs_num * cio_range_dim
        v_dim = ue_num * bs_num * power64

        # 6.
        z_dim = ue_num * bs_num * power64

        # init Q matrix
        d_shift = x_dim  # var
        r_shift = d_shift + d_dim  # var
        s_shift = r_shift + r_dim  # slack
        u_shift = s_shift + s_dim  # slack
        e_shift = u_shift + u_dim  # var
        v_shift = e_shift + e_dim  # slack
        z_shift = v_shift + v_dim
        h_dim = z_shift + z_dim
        h = np.zeros([h_dim + 1, h_dim + 1])

        bin_label = np.zeros(h_dim)
        bin_label = QUBO.bin_label(bin_label, np.arange(r_shift, r_shift + r_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(s_shift, s_shift + s_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(u_shift, u_shift + u_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(v_shift, v_shift + v_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(z_shift, z_shift + z_dim), "slack")

        # Hamiltonian
        h2d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for i in range(ue_num):
                if rb[i, j] > max_rb:
                    rb[i, j] = max_rb
                d_coe, k_bar = int2coe(rb[i, j])  # this k_bar is actually k_bar + 2 in QUBO v2 doc.
                # print(rb[i, j])
                # print(d_coe)
                # print(k_bar)
                # exit()
                const = -156 * capacity[i, j] / (3 * ue_num * 10 ** 6) / 0.0005
                for k in range(k_bar):
                    h2d[d_shift + i * bs_num * power128 + j * power128 + k, -1] += const * d_coe[k] / 2
                    h2d[-1, d_shift + i * bs_num * power128 + j * power128 + k] += const * d_coe[k] / 2

        # constrain 1
        c12d = np.zeros([h_dim + 1, h_dim + 1])
        c1_noslack = np.zeros([bs_num, h_dim + 1])
        for j in range(bs_num):
            c11d = np.zeros([h_dim + 1])
            for i in range(ue_num):
                c11d[i * bs_num + j] = 1
                c1_noslack[j, i * bs_num + j] += 1
            for k in range(power128):
                c11d[r_shift + j * power128 + k] = 2 ** k
            c11d[-1] = -128
            c1_noslack[j, -1] += -128
            c12d = vecmul(c11d, c11d, c12d)
        c1_noslack *= -1

        # constrain 2
        c22d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            c21d = np.zeros([h_dim + 1])
            for j in range(bs_num):
                c21d[i * bs_num + j] = 1
            c21d[-1] = -1
            c22d = vecmul(c21d, c21d, c22d)

        # constrain 3
        c32d = np.zeros([h_dim + 1, h_dim + 1])
        c3_noslack = np.zeros([ue_num, bs_num, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                c31d = np.zeros([h_dim + 1])

                # demand rb
                d_coe, k_bar = int2coe(rb[i, j])  # this k_bar is actually k_bar + 2 in QUBO v2 doc.
                c31d[i * bs_num + j] = k_bar
                c3_noslack[i, j, i * bs_num + j] += k_bar

                # given
                for k in range(k_bar):
                    c31d[d_shift + i * bs_num * power128 + j * power128 + k] = -1
                    c3_noslack[i, j, d_shift + i * bs_num * power128 + j * power128 + k] -= 1

                # slack variable
                for k in range(power16):
                    c31d[s_shift + i * (bs_num * power16) +
                         j * (power16) + k] -= 2 ** k
                c32d = vecmul(c31d, c31d, c32d)

        # constrain 4
        c42d = np.zeros([h_dim + 1, h_dim + 1])
        c4_noslack = np.zeros([bs_num, h_dim + 1])
        for j in range(bs_num):
            c41d = np.zeros([h_dim + 1])

            # constrain : distru < 273
            c41d[-1] = -max_rb
            c4_noslack[j, -1] += -max_rb

            # rb distrubution
            for i in range(ue_num):
                d_coe, k_bar = int2coe(rb[i, j])  # this k_bar is actually k_bar + 2 in QUBO v2 doc.

                for k in range(k_bar):
                    c41d[d_shift + i * bs_num * power128 + j * power128 + k] += d_coe[k]
                    c4_noslack[j, d_shift + i * bs_num * power128 + j * power128 + k] += d_coe[k]

            # slack variable
            for k in range(power256):
                c41d[u_shift + j * power256 + k] += 2 ** k

            c42d = vecmul(c41d, c41d, c42d)
        c4_noslack *= -1

        # constrain 5
        c52d = np.zeros([h_dim + 1, h_dim + 1])
        c5_noslack = np.zeros([ue_num, bs_num, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                serving_idx = int(serving_list[i])
                if serving_idx == j:
                    continue
                c51d = np.zeros([h_dim + 1])

                c51d[i * bs_num + j] = rsrp[i, j] - rsrp[i, serving_idx] - 10
                c5_noslack[i, j, i * bs_num + j] += rsrp[i, j] - \
                                                    rsrp[i, serving_idx] - 10
                for k in range(cio_range_dim):
                    if k == cio_range_dim - 1:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 5
                        c5_noslack[i, j, e_shift + serving_idx *
                                   (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 5
                    else:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** k
                        c5_noslack[i, j, e_shift + serving_idx *
                                   (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** k
                c51d[-1] += 20
                c5_noslack[i, j, -1] += 20

                for k in range(power64):
                    if k == power64 - 1:
                        c51d[v_shift + i * (bs_num * power64) +
                             j * power64 + k] -= 47
                    else:
                        c51d[v_shift + i * (bs_num * power64) +
                             j * power64 + k] -= 2 ** k
                c52d = vecmul(c51d, c51d, c52d)

        # constrain 6
        c62d = np.zeros([h_dim + 1, h_dim + 1])
        c6_noslack = np.zeros([ue_num, bs_num, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                serving_idx = int(serving_list[i])
                if serving_idx == j:
                    continue
                c61d = np.zeros([h_dim + 1])

                c61d[i * bs_num + serving_idx] = rsrp[i, j] - rsrp[i, serving_idx] + 10
                c6_noslack[i, j, i * bs_num + serving_idx] += rsrp[i, j] - \
                                                    rsrp[i, serving_idx] + 10
                for k in range(cio_range_dim):
                    if k == cio_range_dim - 1:
                        c61d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 5
                        c6_noslack[i, j, e_shift + serving_idx *
                                   (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 5
                    else:
                        c61d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** k
                        c6_noslack[i, j, e_shift + serving_idx *
                                   (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** k

                for k in range(power64):
                    if k == power64-1:
                        c61d[z_shift + i * (bs_num * power64) +
                            j * power64 + k] += 47
                    else:
                        c61d[z_shift + i * (bs_num * power64) +
                            j * power64 + k] += 2 ** k
                c62d = vecmul(c61d, c61d, c62d)
        c6_noslack *= -1

        result = h2d + 0.5*c32d + 0.02*c42d + 0.1 * c52d + 0.2 * c62d

        if spin:
            jh = np.zeros([h_dim + 1, h_dim + 1])
            for i in range(h_dim):
                for j in range(h_dim):
                    c2 = result[i, j]
                    jh[i, j] = c2
                    jh[i, -1] += -c2 / 2
                    jh[-1, i] += -c2 / 2
                    jh[j, -1] += -c2 / 2
                    jh[-1, j] += -c2 / 2
                    jh[-1, -1] += c2 / 4

                jh[i, -1] += result[i, -1]
                jh[-1, i] += result[-1, i]
                jh[-1, -1] += -(result[i, -1] + result[-1, i]) / 2
            result = jh

        return result, bin_label, h2d, [c1_noslack, c3_noslack, c4_noslack, c5_noslack, c6_noslack], [c12d, c22d, c32d, c42d, c52d, c62d]

    @staticmethod
    def params2qubo_tp_v2(rsrp, capacity, rb, serving_list, ue_num, bs_num, penalty=10, spin=False):
        def vecmul(vec1, vec2, result=None):
            l1 = len(vec1)
            l2 = len(vec2)
            if result is None:
                result = np.zeros([l1, l2])
            for i in range(l1):
                if vec1[i] == 0:
                    continue
                result[i] += vec1[i] * vec2
            return result

        def int2coe(n, bit_size=8, min=0, max=200):
            assert n >= min, "int2bit : int n nust >= {}".format(min)
            assert n <= max, "int2bit : int n nust <= {}".format(max)
            bits = np.zeros([bit_size])
            i = -1
            if n == 0:
                return bits, i + 1
            else:
                c = 1
                while n - c >= 0:
                    i += 1
                    bits[i] = c
                    n -= c
                    c *= 2

                if n > 0:
                    i += 1
                    bits[i] = n
                return bits, i + 1

        max_rb = 200
        x_dim = ue_num * bs_num


        # rb fo each ue, must under 200, 2^8 > 200
        d_dim = ue_num * bs_num * 8

        # 1. constrain of numbers fo ue conneting to each bs are not greater than 128
        power128 = 7 + 1  # 2^8 > 128
        r_dim = bs_num * power128

        # 2. constrain 2 has no slack variable

        # 3. constrain of demand-throughput >= 0
        power16 = 3 + 1 # 2^4 > 8
        s_dim = ue_num * bs_num * power16

        # 4. constrain of maximum number of bs is 273
        power256 = 8 + 1  # 2^9 > 273
        u_dim = bs_num * power256

        # 5. CIO
        ##################################################
        ### index of e is definded j * j'(j' != j) * 6 ###
        ### 0, [1,2,3,4,5,6], [:]                      ###
        ### 1, [0,2,3,4,5,6], [:]                      ###
        ### 2, [0,1,3,4,5,6], [:]                      ###
        ### ...                                        ###
        ##################################################
        cio_range_dim = 6  # 6 for len of [1, 2, 4, 8, 16, 9]
        e_dim = bs_num * bs_num * cio_range_dim
        v_dim = ue_num * bs_num * power256

        # init Q matrix
        d_shift = x_dim  # var
        r_shift = d_shift + d_dim  # var
        s_shift = r_shift + r_dim  # slack
        u_shift = s_shift + s_dim  # slack
        e_shift = u_shift + u_dim  # var
        v_shift = e_shift + e_dim  # slack
        h_dim = v_shift + v_dim
        h = np.zeros([h_dim + 1, h_dim + 1])

        bin_label = np.zeros(h_dim)
        bin_label = QUBO.bin_label(bin_label, np.arange(r_shift, r_shift + r_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(s_shift, s_shift + s_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(u_shift, u_shift + u_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(v_shift, v_shift + v_dim), "slack")

        # Hamiltonian
        h2d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for i in range(ue_num):
                if rb[i, j] > max_rb:
                    rb[i, j] = max_rb
                d_coe, k_bar = int2coe(rb[i, j]) # this k_bar is actually k_bar + 2 in QUBO v2 doc.
                # print(rb[i, j])
                # print(d_coe)
                # print(k_bar)
                # exit()
                const = -156 * capacity[i, j] / (3 * ue_num * 10 ** 6) / 0.0005
                for k in range(k_bar):
                    h2d[d_shift + i*bs_num*power128+j*power128+k, -1] += const*d_coe[k]/2
                    h2d[-1, d_shift + i*bs_num*power128+j*power128+k] += const*d_coe[k]/2


        # constrain 1
        c12d = np.zeros([h_dim + 1, h_dim + 1])
        c1_noslack = np.zeros([bs_num, h_dim + 1])
        for j in range(bs_num):
            c11d = np.zeros([h_dim + 1])
            for i in range(ue_num):
                c11d[i * bs_num + j] = 1
                c1_noslack[j, i * bs_num + j] += 1
            for k in range(power128):
                c11d[r_shift + j * power128 + k] = 2 ** k
            c11d[-1] = -128
            c1_noslack[j, -1] += -128
            c12d = vecmul(c11d, c11d, c12d)
        c1_noslack *= -1

        # constrain 2
        c22d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            c21d = np.zeros([h_dim + 1])
            for j in range(bs_num):
                c21d[i * bs_num + j] = 1
            c21d[-1] = -1
            c22d = vecmul(c21d, c21d, c22d)

        # constrain 3
        c32d = np.zeros([h_dim + 1, h_dim + 1])
        c3_noslack = np.zeros([ue_num, bs_num, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                c31d = np.zeros([h_dim + 1])

                # demand rb
                d_coe, k_bar = int2coe(rb[i, j]) # this k_bar is actually k_bar + 2 in QUBO v2 doc.
                c31d[i * bs_num + j] = k_bar
                c3_noslack[i, j, i * bs_num + j] += k_bar

                # given
                for k in range(k_bar):
                    c31d[d_shift + i*bs_num*power128+j*power128+k] = -1
                    c3_noslack[i, j, d_shift + i * bs_num * power128 + j * power128 + k] -= 1

                # slack variable
                for k in range(power16):
                    c31d[s_shift + i * (bs_num * power16) +
                         j * (power16) + k] -= 2 ** k
                c32d = vecmul(c31d, c31d, c32d)

        # constrain 4
        c42d = np.zeros([h_dim + 1, h_dim + 1])
        c4_noslack = np.zeros([bs_num, h_dim + 1])
        for j in range(bs_num):
            c41d = np.zeros([h_dim + 1])

            # constrain : distru < 273
            c41d[-1] = -40
            c4_noslack[j, -1] += -40

            # rb distrubution
            for i in range(ue_num):
                d_coe, k_bar = int2coe(rb[i, j])  # this k_bar is actually k_bar + 2 in QUBO v2 doc.

                for k in range(k_bar):
                    c41d[d_shift + i * bs_num * power128 + j * power128 + k] += d_coe[k]
                    c4_noslack[j, d_shift + i * bs_num * power128 + j * power128 + k] += d_coe[k]

            # slack variable
            for k in range(power256):
                c41d[u_shift + j * power256 + k] += 2 ** k

            c42d = vecmul(c41d, c41d, c42d)
        c4_noslack *= -1


        # constrain 5
        c52d = np.zeros([h_dim + 1, h_dim + 1])
        c5_noslack = np.zeros([ue_num, bs_num, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                serving_idx = int(serving_list[i])
                if serving_idx == j:
                    continue
                c51d = np.zeros([h_dim + 1])

                c51d[i * bs_num + j] = rsrp[i, j] - rsrp[i, serving_idx] - 10
                c5_noslack[i, j, i * bs_num + j] += rsrp[i, j] - \
                                            rsrp[i, serving_idx] - 10
                for k in range(cio_range_dim):
                    if k == cio_range_dim - 1:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 9 / 2
                        c5_noslack[i, j, e_shift + serving_idx *
                                 (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 9 / 2
                    else:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** (k - 1)
                        c5_noslack[i, j, e_shift + serving_idx *
                                 (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** (k - 1)
                c51d[-1] += 20
                c5_noslack[i, j, -1] += 20

                for k in range(power256):
                    if k == power256 - 1:
                        c51d[v_shift + i * (bs_num * power256) +
                             j * power256 + k] -= 185 / 2
                    else:
                        c51d[v_shift + i * (bs_num * power256) +
                             j * power256 + k] -= 2 ** (k - 1)
                c51d[-1] += 110
                c52d = vecmul(c51d, c51d, c52d)

        result = h2d + 0.15*c32d + 0.01*c42d +  0.15*c52d

        if spin:
            jh = np.zeros([h_dim + 1, h_dim + 1])
            for i in range(h_dim):
                for j in range(h_dim):
                    c2 = result[i, j]
                    jh[i, j] = c2
                    jh[i, -1] += -c2 / 2
                    jh[-1, i] += -c2 / 2
                    jh[j, -1] += -c2 / 2
                    jh[-1, j] += -c2 / 2
                    jh[-1, -1] += c2 / 4

                jh[i, -1] += result[i, -1]
                jh[-1, i] += result[-1, i]
                jh[-1, -1] += -(result[i, -1] + result[-1, i]) / 2
            result = jh

        return result, bin_label, h2d, [c1_noslack, c3_noslack, c4_noslack, c5_noslack], [c12d, c22d, c32d, c42d, c52d]

    @staticmethod
    def params2qubo_tp(rsrp, capacity, rb, serving_list, ue_num, bs_num, penalty=10, spin=False):
        def vecmul(vec1, vec2, result=None):
            l1 = len(vec1)
            l2 = len(vec2)
            if result is None:
                result = np.zeros([l1, l2])
            for i in range(l1):
                if vec1[i] == 0:
                    continue
                result[i] += vec1[i] * vec2
            return result

        max_rbnum_ue = 199
        x_dim = ue_num * bs_num

        # rb fo each ue, must under 200, 10*19+(1+2+4+2)
        digit_num = 4  # 4 for len of [1, 2, 4, 2]
        robinmax = (max_rbnum_ue // 10)
        robin10_dim = bs_num * robinmax  # t == 19
        robin2_dim = ue_num * bs_num * digit_num  # d == 1050*4

        # 1. constrain of numbers fo ue conneting to each bs are not greater than 128
        power128 = 7 + 1  # summation have to > 128
        r_dim = bs_num * power128

        # 2. constrain 2 has no slack variable

        # 3. constrain of demand-throughput >= 0
        s_dim = ue_num * bs_num * power128

        # 4. constrain of maximum number of bs is 273
        power256 = 8 + 1  # summation have to > 273
        y_dim = ue_num * bs_num * robinmax
        ybar_dim = ue_num * bs_num * digit_num
        u_dim = bs_num * power256

        # 5. CIO
        ##################################################
        ### index of e is definded j * j'(j' != j) * 6 ###
        ### 0, [1,2,3,4,5,6], [:]                      ###
        ### 1, [0,2,3,4,5,6], [:]                      ###
        ### 2, [0,1,3,4,5,6], [:]                      ###
        ### ...                                        ###
        ##################################################
        cio_range_dim = 6  # 6 for len of [1, 2, 4, 8, 16, 9]
        e_dim = bs_num * bs_num * cio_range_dim
        v_dim = ue_num * bs_num * power256

        # init jh matrix
        robin10_shift = x_dim                        # var
        robin2_shift = robin10_shift + robin10_dim   # var
        r_shift = robin2_shift + robin2_dim          # slack
        s_shift = r_shift + r_dim                    # slack
        y_shift = s_shift + s_dim                    # var
        ybar_shift = y_shift + y_dim                 # var
        u_shift = ybar_shift + ybar_dim              # slack
        e_shift = u_shift + u_dim                    # var
        v_shift = e_shift + e_dim                    # slack
        h_dim = v_shift + v_dim
        h = np.zeros([h_dim + 1, h_dim + 1])

        bin_label = np.zeros(h_dim)
        bin_label = QUBO.bin_label(bin_label, np.arange(r_shift, r_shift+r_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(s_shift, s_shift + s_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(u_shift, u_shift + u_dim), "slack")
        bin_label = QUBO.bin_label(bin_label, np.arange(v_shift, v_shift + v_dim), "slack")

        # Hamiltonian
        h2d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for i in range(ue_num):
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    h2d[robin10_shift + j * robinmax + k, i * bs_num + j] += -10 / 0.0005 * 156 * capacity[
                        i, j] / (3 * ue_num * 10 ** 6) / 2
                    h2d[i * bs_num + j, robin10_shift + j * robinmax + k] += -10 / 0.0005 * 156 * capacity[
                        i, j] / (3 * ue_num * 10 ** 6) / 2
                for k in range(digit_num):
                    if k == digit_num - 1:
                        h2d[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k, i * bs_num + j] += \
                            -2 / 0.0005 * 156 * \
                            capacity[i, j] / (3 * ue_num * 10 ** 6) / 2
                        h2d[i * bs_num + j, robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] += \
                            -2 / 0.0005 * 156 * \
                            capacity[i, j] / (3 * ue_num * 10 ** 6) / 2
                    else:
                        h2d[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k, i * bs_num + j] += \
                            -(2 ** k) / 0.0005 * 156 * \
                            capacity[i, j] / (3 * ue_num * 10 ** 6) / 2
                        h2d[i * bs_num + j, robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] += \
                            -(2 ** k) / 0.0005 * 156 * \
                            capacity[i, j] / (3 * ue_num * 10 ** 6) / 2

        # constrain 1
        c12d = np.zeros([h_dim + 1, h_dim + 1])
        c1_check = np.zeros([h_dim + 1])
        for j in range(bs_num):
            c11d = np.zeros([h_dim + 1])
            for i in range(ue_num):
                c11d[i * bs_num + j] = 1
                c1_check[i * bs_num + j] += 1
            for k in range(power128):
                c11d[r_shift + j * power128 + k] = 2 ** k
            c11d[-1] = -128
            c1_check[-1] = -128
            c12d = vecmul(c11d, c11d, c12d)
        c1_check *= -1

        # constrain 2
        c22d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            c21d = np.zeros([h_dim + 1])
            for j in range(bs_num):
                c21d[i * bs_num + j] = 1
            c21d[-1] = -1
            c22d = vecmul(c21d, c21d, c22d)

        # constrain 3
        c32d = np.zeros([h_dim + 1, h_dim + 1])
        c3_check = np.zeros([h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                c31d = np.zeros([h_dim + 1])

                # demand rb
                if rb[i, j] > 199:
                    c31d[-1] = 199
                    c3_check[-1] += 199
                else:
                    c31d[-1] = rb[i, j]
                    c3_check[-1] += rb[i, j]

                # rb distrubution
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    c31d[robin10_shift + j * robinmax + k] -= 10
                    c3_check[robin10_shift + j * robinmax + k] -= 10
                for k in range(digit_num):
                    if k == digit_num - 1:
                        c31d[robin2_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] -= 2
                        c3_check[robin2_shift + i *
                                 (bs_num * digit_num) + j * digit_num + k] -= 2
                    else:
                        c31d[robin2_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] -= 2 ** k
                        c3_check[robin2_shift + i *
                                 (bs_num * digit_num) + j * digit_num + k] -= 2 ** k

                # slack variable
                for k in range(power128):
                    c31d[s_shift + i * (bs_num * power128) +
                         j * (power128) + k] -= 2 ** k
                c32d = vecmul(c31d, c31d, c32d)

        # constrain 4
        c42d = np.zeros([h_dim + 1, h_dim + 1])
        c4_check = np.zeros(h_dim+1)
        for j in range(bs_num):
            c41d = np.zeros([h_dim + 1])

            # constrain : distru < 273
            c41d[-1] = -273
            c4_check[-1] += -273

            # rb distrubution
            for i in range(ue_num):
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    c41d[y_shift + i * (bs_num * robinmax) +
                         j * robinmax + k] += 10
                    c4_check[y_shift + i *
                             (bs_num * robinmax) + j * robinmax + k] += 10
                for k in range(digit_num):
                    if k == digit_num - 1:
                        c41d[ybar_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] += 2
                        c4_check[ybar_shift + i *
                                 (bs_num * digit_num) + j * digit_num + k] += 2
                    else:
                        c41d[ybar_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] += 2 ** k
                        c4_check[ybar_shift + i *
                                 (bs_num * digit_num) + j * digit_num + k] += 2 ** k

            # slack variable
            for k in range(power256):
                c41d[u_shift + j * power256 + k] += 2 ** k

            c42d = vecmul(c41d, c41d, c42d)
        c4_check *= -1

        c452d = np.zeros([h_dim + 1, h_dim + 1])
        p = 1
        for i in range(ue_num):
            for j in range(bs_num):
                for k in range(robinmax):
                    c452d[robin10_shift + j * robinmax +
                          k, i * bs_num + j] += p * 1
                    c452d[i * bs_num + j, robin10_shift +
                          j * robinmax + k] += p * 1
                    c452d[robin10_shift + j * robinmax + k, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c452d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, robin10_shift + j * robinmax + k] += p * -2
                    c452d[i * bs_num + j, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c452d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, i * bs_num + j] += p * -2
                    c452d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, -1] += p * 3
                    c452d[-1, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * 3

                for k in range(digit_num):
                    c452d[robin2_shift + i * (bs_num * digit_num) +
                          j * digit_num + k, i * bs_num + j] += p * 1
                    c452d[i * bs_num + j, robin2_shift + + i *
                          (bs_num * digit_num) + j * digit_num + k] += p * 1
                    c452d[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k, ybar_shift + i * (
                        bs_num * digit_num) + j * (
                        digit_num) + k] += p * -2
                    c452d[ybar_shift + i * (bs_num * digit_num) + j * (
                        digit_num) + k, robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] += p * -2
                    c452d[i * bs_num + j, ybar_shift + i * (bs_num * digit_num) + j * (
                        digit_num) + k] += p * -2
                    c452d[ybar_shift + i * (bs_num * digit_num) + j * (
                        digit_num) + k, i * bs_num + j] += p * -2
                    c452d[ybar_shift + i * (bs_num * digit_num) + j * (
                        digit_num) + k, -1] += p * 3
                    c452d[-1, ybar_shift + i * (bs_num * digit_num) + j * (
                        digit_num) + k] += p * 3

        # constrain 5
        c52d = np.zeros([h_dim + 1, h_dim + 1])
        c5_check = np.zeros([h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                serving_idx = int(serving_list[i])
                if serving_idx == j:
                    continue
                c51d = np.zeros([h_dim + 1])

                c51d[i * bs_num + j] = rsrp[i, j] - rsrp[i, serving_idx] - 10
                c5_check[i * bs_num + j] += rsrp[i, j] - \
                    rsrp[i, serving_idx] - 10
                for k in range(cio_range_dim):
                    if k == cio_range_dim - 1:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 9 / 2
                        c5_check[e_shift + serving_idx *
                                 (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 9 / 2
                    else:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** (k - 1)
                        c5_check[e_shift + serving_idx *
                                 (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** (k - 1)
                c51d[-1] += 20
                c5_check[-1] += 20

                for k in range(power256):
                    if k == power256 - 1:
                        c51d[v_shift + i * (bs_num * power256) +
                             j * power256 + k] -= 185 / 2
                    else:
                        c51d[v_shift + i * (bs_num * power256) +
                             j * power256 + k] -= 2 ** (k - 1)
                c51d[-1] += 110
                c52d = vecmul(c51d, c51d, c52d)

        # constrain 6
        c62d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for k in range(robinmax - 1):
                c62d[robin10_shift + j * robinmax + k + 1, -1] += 1 / 2
                c62d[-1, robin10_shift + j * robinmax + k + 1] += 1 / 2
                c62d[robin10_shift + j * robinmax + k + 1,
                     robin10_shift + j * robinmax + k] -= 1 / 2
                c62d[robin10_shift + j * robinmax + k,
                     robin10_shift + j * robinmax + k + 1] -= 1 / 2

        # result = h2d + penalty * (c12d + c22d + c32d + c42d + c452d + c52d + c62d)
        result = h2d + c452d + 3*c62d + c52d
        # result = h2d + (c12d/128**2 + 2*c22d + c32d/45**2 +
        #      c42d / 273**2 + 2*c452d/273**2 + c52d/110**2 + 2*c62d)
        # p1 = 0.6035295380
        # p2 = 3.8406425145
        # p3 = 0.2743316081
        # p4 = 0.2743316081
        # p45 = 38 #1497.850850688
        # p5 = 1.9203212572
        # p6 = 38.4064251458
        # result = h2d + \
        #          (p1 * c12d + p2 * c22d + p3 * c32d +
        #           p4 * c42d + p45 * c452d  + p5 *  c52d + p6 * c62d)

        if spin:
            jh = np.zeros([h_dim + 1, h_dim + 1])
            for i in range(h_dim):
                for j in range(h_dim):
                    c2 = result[i, j]
                    jh[i, j] = c2
                    jh[i, -1] += -c2 / 2
                    jh[-1, i] += -c2 / 2
                    jh[j, -1] += -c2 / 2
                    jh[-1, j] += -c2 / 2
                    jh[-1, -1] += c2 / 4

                jh[i, -1] += result[i, -1]
                jh[-1, i] += result[-1, i]
                jh[-1, -1] += -(result[i, -1] + result[-1, i]) / 2
            result = jh

        return result, bin_label, h2d, c1_check, c3_check, c4_check, c5_check, c12d, c22d, c32d, c42d, c452d, c52d, c62d

    @staticmethod
    def params2qubo_std(rsrp, sinr, rb, ue_num, bs_num):
        assert (ue_num, bs_num) == rsrp.shape, "rsrp must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, rsrp.shape)
        assert (ue_num, bs_num) == sinr.shape, "sinr must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, sinr.shape)
        assert (ue_num, bs_num) == rb.shape, "rb must have same shape with ({}, {})," \
                                             " but given {}".format(
                                                 ue_num, bs_num, rb.shape)

        bin_size = bs_num * ue_num
        Q = np.zeros([bin_size, bin_size])
        for j in range(bs_num):
            for i in range(ue_num):
                for k in range(ue_num):
                    Q[i + j * ue_num, k + j * ue_num] += rb[i, j] * rb[k, j]

        for k in range(bs_num):
            for l in range(ue_num):
                for j in range(bs_num):
                    for i in range(ue_num):
                        Q[i + j * ue_num, l + k * ue_num] += - \
                            1 * rb[i, j] * rb[l, k] / bs_num

        Q /= bs_num
        return Q

    @staticmethod
    def params2jh_std(rsrp, sinr, rb, ue_num, bs_num):
        assert (ue_num, bs_num) == rsrp.shape, "rsrp must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, rsrp.shape)
        assert (ue_num, bs_num) == sinr.shape, "sinr must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, sinr.shape)
        assert (ue_num, bs_num) == rb.shape, "rb must have same shape with ({}, {})," \
                                             " but given {}".format(
                                                 ue_num, bs_num, rb.shape)

        bin_size = bs_num * ue_num
        J = np.zeros([bin_size, bin_size])
        H = np.zeros([bin_size])
        C = 0
        for j in range(bs_num):
            for i in range(ue_num):
                for k in range(ue_num):
                    coe1 = rb[i, j] * rb[k, j]
                    J[i + j * ue_num, k + j * ue_num] += coe1
                    H[i + j * ue_num] += coe1 / 2
                    H[k + j * ue_num] += coe1 / 2
                    C += coe1 / 4

        for k in range(bs_num):
            for l in range(ue_num):
                for j in range(bs_num):
                    for i in range(ue_num):
                        coe2 = -1 * rb[i, j] * rb[l, k] / bs_num
                        J[i + j * ue_num, l + k * ue_num] += coe2
                        H[i + j * ue_num] += coe2 / 2
                        H[l + k * ue_num] += coe2 / 2
                        C += (coe2 / 4)

        J /= bs_num
        H /= bs_num
        C /= bs_num
        return J, H, C

    @staticmethod
    def data_parse(data_path):
        '''

        :param data_path: .pkl file path
        :return: prb, has shape [time step, UE num, BS num]
        '''
        with open(data_path, "rb") as f:
            coe = pickle.load(f)
            rsrp = coe["RSRP_dBm"]
            sinr = coe["SINR_dB"]
            prb = coe["PRB"]
        return rsrp, sinr, prb

    @staticmethod
    def write_file(file_name, Q):
        f = open(file_name, "w")
        q_size = len(Q)
        ele_num = (q_size * (q_size - 1)) / 2 + q_size
        f.write("%d %d %d\n" % (q_size, q_size, ele_num))
        for i in range(q_size):
            for j in range(i, q_size):
                f.write("%d %d %20.16g\n" % (i, j, Q[i, j]))
        f.close()

    @staticmethod
    def read_file(file_name):
        f = open(file_name, "r")
        data = f.readlines()
        q_size, _, ele_num = data[0].split()
        Q = np.zeros([int(q_size), int(q_size)])
        for n in range(1, int(ele_num) + 1):
            i, j, value = data[n].split()
            Q[int(i), int(j)] = float(value)
            Q[int(j), int(i)] = float(value)
        return Q

    @staticmethod
    def pkl2txt(pkl, save_path):
        with open(pkl, "rb") as f:
            coe = pickle.load(f)
            rsrp = coe["RSRP_dBm"]
            sinr = coe["SINR_dB"]
            prb = coe["PRB"]

            t, ue_num, bs_num = rsrp.shape
            for i in range(0, t, 100):
                data_file = open("{}/t{}.txt".format(save_path, i), "w")
                data_file.write("%d %d\n" % (ue_num, bs_num))
                for ue in range(ue_num):
                    for bs in range(bs_num):
                        data_file.write("%d %d %20.16g %20.16g %20.16g\n"
                                        % (ue, bs, rsrp[i, ue, bs], sinr[i, ue, bs], prb[i, ue, bs]))
                data_file.close()

    @staticmethod
    def read_raw_data(file_name):
        f = open(file_name, "r")
        data = f.readlines()
        ue_num, bs_num = data[0].split()
        rsrp_arr = np.zeros([int(ue_num), int(bs_num)])
        sinr_arr = np.zeros([int(ue_num), int(bs_num)])
        prb_arr = np.zeros([int(ue_num), int(bs_num)])
        for n in range(1, int(ue_num) * int(bs_num) + 1):
            i, j, rsrp, sinr, prb = data[n].split()
            rsrp_arr[int(i), int(j)] = rsrp
            sinr_arr[int(i), int(j)] = sinr
            prb_arr[int(i), int(j)] = prb
        return rsrp_arr, sinr_arr, prb_arr, int(ue_num), int(bs_num)

    @staticmethod
    def check_constrain(binary, ue_num=7, bs_num=2, eq_constrains=None, c1=None, c2=None, c3=None, c4=None, c5=None, c6=None):
        if c1 is not None:
            result = np.zeros(bs_num)
            for j in range(bs_num):
                result[j] = binary.T @ c1[j]
            print("c1 inequal check : {}".format(result))
        elif c3 is not None:
            result = np.zeros((ue_num, bs_num))
            for i in range(ue_num):
                for j in range(bs_num):
                    result[i, j] = binary.T @ c3[i, j]
            print("c3 inequal check : {}".format(result))
        elif c4 is not None:
            result = np.zeros(bs_num)
            for j in range(bs_num):
                result[j] = binary.T @ c4[j]
            print("c4 inequal check : {}".format(result))
        elif c5 is not None:
            result = np.zeros((ue_num, bs_num))
            for i in range(ue_num):
                for j in range(bs_num):
                    result[i, j] = binary.T @ c5[i, j]
            print("c5 inequal check : {}".format(result))
        elif c6 is not None:
            result = np.zeros((ue_num, bs_num))
            for i in range(ue_num):
                for j in range(bs_num):
                    result[i, j] = binary.T @ c6[i, j]
            print("c6 inequal check : {}".format(result))
        else:
            result = []
            for c in eq_constrains:
                r = binary.T @ c @ binary
                # if r >= 0:
                result.append(r[0][0])
        return result

    @staticmethod
    def check_Q(Q):
        l = len(Q)
        for i in range(l):
            for j in range(l):
                if Q[i, j] != Q[j, i]:
                    return False
        return True

    @staticmethod
    def init_bin(capacity, q_size, bs_num):
        init_b = np.zeros([q_size])
        for i in range(len(capacity)):
            index = np.where(capacity[i] == np.max(capacity[i]))[0]
            init_b[i * bs_num + index] = 1
        return init_b

    @staticmethod
    def init_serving_list(capacity, ue_num):
        serving_list = np.zeros([ue_num])
        for i in range(len(capacity)):
            index = np.where(capacity[i] == np.max(capacity[i]))[0]
            serving_list[i] = index
        return serving_list

    @staticmethod
    def int2bit(n, bit_size):
        assert n >= 0, "int2bit : int n nust >= 0"
        assert bit_size >= 1, "int2bit : bit_size nust >= 1"
        bits = np.zeros([bit_size])
        idx = 0
        while (n > 0):
            r = n % 2
            if r == 1:
                bits[idx] = 1
            n = int(n / 2)
            idx += 1
        return bits

    @staticmethod
    def init_cio(bs_num, cio_value=None):
        # cio_value is in range [-10, 10]
        if cio_value is None:
            cio_value = np.zeros((bs_num, bs_num))
        assert cio_value.shape == (bs_num, bs_num), \
            "cio_value must have same shape with ({},{}), but given {}".format(
                bs_num, bs_num, cio_value.shape)

        cio = np.zeros((bs_num, bs_num, 6))  # 6 for len of [1, 2, 4, 8, 16, 9]
        for i in range(bs_num):
            for j in range(bs_num):
                cio_trans = (cio_value[i, j] - 10) * -2
                if cio_trans > 31:
                    cio[i, j, -1] = 1
                    cio_trans -= 9
                bits = QUBO.int2bit(cio_trans, 6)
                cio[i, j] += bits

        return cio

    @staticmethod
    def gen_answer_tp(serving_list, ue_num, bs_num, prb, rsrp, cio_setting, x=None):

        serving_list = serving_list.astype(int)
        for i in range(ue_num):
            for j in range(bs_num):
                if prb[i, j] > 199:
                    prb[i, j] = 199

        max_rbnum_ue = 199
        x_dim = ue_num * bs_num

        # rb fo each ue, must under 200, 10*19+(1+2+4+2)
        digit_num = 4  # 4 for len of [1, 2, 4, 2]
        robinmax = (max_rbnum_ue // 10)
        robin10_dim = bs_num * robinmax  # t == 19
        robin2_dim = ue_num * bs_num * digit_num  # d == 1050*4

        # 1. constrain of numbers fo ue conneting to each bs are not greater than 128
        power128 = 7 + 1  # summation have to > 128
        r_dim = bs_num * power128

        # 2. constrain 2 has no slack variable

        # 3. constrain of demand-throughput >= 0
        s_dim = ue_num * bs_num * power128

        # 4. constrain of maximum number of bs is 273
        power256 = 8 + 1  # summation have to > 273
        y_dim = ue_num * bs_num * robinmax
        ybar_dim = ue_num * bs_num * digit_num
        u_dim = bs_num * power256

        # 5. CIO
        ##################################################
        ### index of e is definded j * j'(j' != j) * 6 ###
        ### 0, [1,2,3,4,5,6], [:]                      ###
        ### 1, [0,2,3,4,5,6], [:]                      ###
        ### 2, [0,1,3,4,5,6], [:]                      ###
        ### ...                                        ###
        ##################################################
        cio_range_dim = 6  # 6 for len of [1, 2, 4, 8, 16, 9]
        e_dim = bs_num * bs_num * cio_range_dim
        v_dim = ue_num * bs_num * power256

        # init jh matrix
        robin10_shift = x_dim
        robin2_shift = robin10_shift + robin10_dim
        r_shift = robin2_shift + robin2_dim
        s_shift = r_shift + r_dim
        y_shift = s_shift + s_dim
        ybar_shift = y_shift + y_dim
        u_shift = ybar_shift + ybar_dim
        e_shift = u_shift + u_dim
        v_shift = e_shift + e_dim
        h_dim = v_shift + v_dim

        cio_list = np.array([1, 2, 4, 8, 16, 9])
        if x is None:
            x = np.zeros(ue_num)
            for i in range(ue_num):
                maxdelta = 0
                idx = serving_list[i]
                for j in range(bs_num):
                    delta = rsrp[i, j] - rsrp[i, serving_list[i]]
                    t = ((cio_setting[j, serving_list[i]] @ cio_list) / 2 - 10)
                    if delta - ((cio_setting[j, serving_list[i]] @ cio_list) / 2 - 10) > maxdelta:
                        maxdelta = delta
                        idx = j
                x[i] = idx
        x = x.astype(int)

        result = np.zeros(h_dim + 1)
        rb = np.zeros([ue_num, bs_num])
        # calculate connet number of each bs
        connet_counter = np.zeros([bs_num])
        for i in range(ue_num):
            connet_counter[x[i]] += 1
            result[i * bs_num + x[i]] = 1
            # rb is an array only have value on conneted bs
            rb[i, x[i]] = prb[i, x[i]]

        # constrain 5
        cio_list_len = len(cio_list)
        for i in range(ue_num):
            for j in range(bs_num):
                serving_n = serving_list[i]
                if serving_n == j:
                    continue
                e = cio_setting[serving_n, j]
                result[e_shift + serving_n * (bs_num * cio_list_len) + j * cio_list_len:e_shift + serving_n * (
                    bs_num * cio_list_len) + j * cio_list_len + cio_list_len] = e
                slack = (result[i * bs_num + j] * (
                    rsrp[i, j] - rsrp[i, serving_n] - 10) - e @ cio_list / 2 + 20 + 110) * 2
                if slack > 255:
                    slack -= 185
                    result[v_shift + i *
                           (bs_num * power256) + j * power256 + power256] = 1
                bits = QUBO.int2bit(slack, 8)
                result[v_shift + i * (bs_num * power256) + j * power256:v_shift + i * (
                    bs_num * power256) + j * power256 + power256 - 1] = bits

        # Hamiltonian
        for j in range(bs_num):
            bs_assigned_rb = 0
            if connet_counter[j] > 0:

                #### run robin 10 ####
                # cumulation
                cumulate = np.zeros([19])
                t = np.zeros([19])
                for i in range(ue_num):
                    rb_ij = rb[i, j]
                    quotient = int(rb_ij / 10)
                    if quotient == 0:
                        continue
                    cumulate[:quotient] += 1

                # assign
                maxrb = 273
                for k in range(robinmax):
                    if cumulate[k] != 0:
                        if maxrb - cumulate[k] * 10 >= 0:
                            t[k] = 1
                            maxrb -= cumulate[k] * 10
                        else:
                            break
                    else:
                        break
                result[robin10_shift + j * robinmax: robin10_shift +
                       j * robinmax + robinmax] = t

                #### run robin digit ####
                digit_trg = rb[:, j] % 10
                digit_assign = np.zeros(ue_num)
                while (maxrb > 0):
                    for i in range(ue_num):
                        if maxrb == 0:
                            break
                        if digit_assign[i] < digit_trg[i]:
                            digit_assign[i] += 1
                            maxrb -= 1
                    if np.array_equal(digit_assign, digit_trg):
                        break

                for i in range(ue_num):
                    if digit_assign[i] == 0:  # not connet to this BS
                        d = np.array([0, 0, 0, 0])
                        dn = 0
                    if digit_assign[i] == 1:
                        d = np.array([1, 0, 0, 0])
                        dn = 1
                    elif digit_assign[i] == 2:
                        d = np.array([0, 1, 0, 0])
                        dn = 2
                    elif digit_assign[i] == 3:
                        d = np.array([1, 1, 0, 0])
                        dn = 3
                    elif digit_assign[i] == 4:
                        d = np.array([0, 0, 1, 0])
                        dn = 4
                    elif digit_assign[i] == 5:
                        d = np.array([1, 0, 1, 0])
                        dn = 5
                    elif digit_assign[i] == 6:
                        d = np.array([0, 1, 1, 0])
                        dn = 6
                    elif digit_assign[i] == 7:
                        d = np.array([1, 1, 1, 0])
                        dn = 7
                    elif digit_assign[i] == 8:
                        d = np.array([0, 1, 1, 1])
                        dn = 8
                    elif digit_assign[i] == 9:
                        d = np.array([1, 1, 1, 1])
                        dn = 9
                    result[robin2_shift + i * (bs_num * digit_num) + j * digit_num:robin2_shift + i * (
                        bs_num * digit_num) + j * digit_num + digit_num] = d

                    #### constrain 3 ####
                    rb_ij = prb[i, j]
                    slack = rb_ij - np.sum(t[:int(rb_ij / 10)]) * 10 - dn
                    if rb[i, j] > 0:  # only conneted ue contribute assigned rb
                        bs_assigned_rb += np.sum(t[:int(rb_ij / 10)]) * 10 + dn
                    s_tmp = QUBO.int2bit(slack, bit_size=8)
                    result[s_shift + i * (bs_num * power128) + j * power128:s_shift + i * (
                        bs_num * power128) + j * power128 + power128] = s_tmp

                    a = np.sum(t[:int(rb_ij / 10)]) * 10
                    b = d @ np.array([1, 2, 4, 2])
                    c = s_tmp @ np.array([1, 2, 4, 8, 16, 32, 64, 128])
                    test = prb[i, j] - np.sum(t[:int(rb_ij / 10)]) * 10 - d @ np.array([1, 2, 4, 2]) - s_tmp @ np.array(
                        [1, 2, 4, 8, 16, 32, 64, 128])
                    assert test == 0

                #### constrain 4 slack ####
                slack = 273 - bs_assigned_rb
                u_tmp = QUBO.int2bit(slack, bit_size=9)
                result[u_shift + j * power256:u_shift +
                       j * power256 + power256] = u_tmp

                assert bs_assigned_rb + \
                    u_tmp @ np.array([1, 2, 4, 8, 16, 32, 64,
                                     128, 256]) - 273 == 0

        #### constrain 4 ####
        for i in range(ue_num):
            for j in range(bs_num):
                for k in range(robinmax):
                    result[y_shift + i * (bs_num * robinmax) + j * robinmax + k] = result[
                        robin10_shift + j * robinmax + k] * \
                        result[i * bs_num + j]

                    test = result[robin10_shift + j * robinmax + k] * result[i * bs_num + j] - 2 * result[
                        robin10_shift + j * robinmax + k] * result[y_shift + i * (bs_num * robinmax) + j * (
                            robinmax) + k] - 2 * result[i * bs_num + j] * result[y_shift + i * (bs_num * robinmax) + j * (
                                robinmax) + k] + 3 * result[y_shift + i * (bs_num * robinmax) + j * (
                                    robinmax) + k]
                    assert test == 0

                for k in range(digit_num):
                    result[ybar_shift + i * (bs_num * digit_num) + j * digit_num + k] = result[
                        robin2_shift + i * (
                            bs_num * digit_num) + j * digit_num + k] * \
                        result[i * bs_num + j]

                    test = result[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] * result[
                        i * bs_num + j] - 2 * result[
                        robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] * result[
                        ybar_shift + i * (bs_num * digit_num) + j * (
                            digit_num) + k] - 2 * result[i * bs_num + j] * result[
                        ybar_shift + i * (bs_num * digit_num) + j * (
                            digit_num) + k] + 3 * result[ybar_shift + i * (bs_num * digit_num) + j * (
                                digit_num) + k]
                    assert test == 0

        #### constrain 1 ####
        for j in range(bs_num):
            slack = 128 - connet_counter[j]
            r_tmp = QUBO.int2bit(slack, bit_size=8)
            result[r_shift + j * power128:r_shift +
                   j * power128 + power128] = r_tmp

        result[-1] = 1
        return result

    @staticmethod
    def bin_label(bin_state, idx, type):
        '''

        Args:
            bin_state: binary state array
            idx: idx list needs to be labeled
            type: a string vars must be one of ["fixed", "slack"]

        Returns: labeled binary state array

        '''
        label_type = ["fixed", "slack"]

        assert type in label_type, 'a string vars type must be one of ["fixed", "slack"]'
        if type == "fixed":
            bin_state[idx] = 1
        elif type == "slack":
            bin_state[idx] = 2
        return bin_state

    @staticmethod
    def onehot(bin_label, groups, init_bin, init_idx=3):
        '''

        Args:
            bin_label: binary label array initial as 3 (1 is fixed, 2 is slack)
            groups: a 2D list, each element is a group array. ex. [(g11, g12, g13, ), (g21, g22, g23, g24), ...]
            init_bin: initial bianry distribution

        Returns: updated bin_label, activate label idx

        '''
        act_idx = np.zeros(len(groups)+3)

        group_idx = init_idx
        for g in groups:
            for i in g:
                bin_label[i] = group_idx
                if init_bin[i] == 1:
                    act_idx[group_idx] = i
            group_idx += 1

        return bin_label, act_idx, len(groups)+3

def bqp_test(path="../data/beasley/", file="bqp", iters=1000, spin=False):
    data_list = [tag for tag in glob("{}/{}*".format(path, file))]
    solution_file = open("{}/solution.txt".format(path), "r").readlines()
    solution = {}
    for s in solution_file:
        f, v = s.split(" ")
        solution[f] = int(v)

    output_file = open("{}hist{}.txt".format(file, iters), "w")
    data_list = np.sort(data_list)
    for data in tqdm(data_list):
        sv = solution[data.split("/")[-1]]
        distribution = np.zeros(10)
        t = 0
        for n in tqdm(range(100)):
            f = open(data, "r").readlines()

            bin_size = f[0].split(" ")[0]
            Q = np.zeros([int(bin_size) + 1, int(bin_size) + 1])
            init_bin = np.zeros([int(bin_size) + 1])
            init_bin[-1] = 1
            for ele in f[1:]:
                i, j, v = ele.split()
                Q[int(i) - 1, int(j) - 1] = int(v)
                Q[int(j) - 1, int(i) - 1] = int(v)

            #### annealing ####
            da1 = DA(Q, init_bin, maxStep=iters,
                     betaStart=0.01, betaStop=100, kernel_dim=(32 * 2,), spin=spin)
            da1.run()
            # print(da1.binary)
            # print(f'time spent: {da1.time}')

            bin1 = np.expand_dims(da1.binary, axis=1)
            output = np.matmul(np.matmul(bin1.T, Q), bin1)[0][0]

            if (sv - output) > 0:
                l = 0
            else:
                l = int(np.ceil((abs((sv - output) / sv) * 100)))
            distribution[l] += 1
            t += da1.time
        t /= 100
        distribution /= 100
        output_file.write("{}\t{}\t{}\n".format(data.split("/")[-1].split(".")[0], t, distribution))

def g05_test(path="../data/g05/", file="g05", iters=100, spin=False):
    data_list = [tag for tag in glob("{}/{}_100*".format(path, file))]
    solution_file = open("{}/solution.txt".format(path), "r").readlines()
    solution = {}
    for s in solution_file:
        f, v = s.split(" ")
        solution[f] = int(v)

    output_file = open("{}hist{}.txt".format(file, iters), "w")
    data_list = np.sort(data_list)
    for data in tqdm(data_list):
        sv = solution[data.split("/")[-1]]
        distribution = np.zeros(20)
        t = 0
        f = open(data_list[0], "r").readlines()

        bin_size = f[0].split(" ")[0]
        Q = np.zeros([int(bin_size) + 1, int(bin_size) + 1])
        for ele in f[1:]:
            i, j, v = ele.split()
            Q[int(i) - 1, int(j) - 1] = int(v)
            Q[int(j) - 1, int(i) - 1] = int(v)

        for n in tqdm(range(100)):
            init_bin = np.zeros([int(bin_size) + 1]) + 1
            init_bin[-1] = 1

            #### annealing ####
            da1 = DA(Q, init_bin, maxStep=iters,
                     betaStart=0.01, betaStop=100, kernel_dim=(32 * 2,), spin=spin)
            da1.run()

            bin1 = np.expand_dims(da1.binary, axis=1)
            # output = np.matmul(np.matmul(bin1.T, Q), bin1)[0][0]
            output = 0
            for i in range(int(bin_size)):
                for j in range(i, int(bin_size)):
                    output += Q[i, j] * (1 - da1.binary[i] * da1.binary[j])
            output /= 2

            if (sv - output) < 0:
                l = 0
                # print(data)
                # print(da1.binary)
                # print(output)
                # print(abs((sv - output) / sv))
                # print((sv - output))
                # print(np.ceil((abs((sv - output) / sv) * 100)))
                # exit()
            else:
                l = int(np.ceil((abs((sv - output) / sv) * 100)))

            try:
                distribution[l] += 1
            except:
                print(data)
                # print(da1.binary)
                print(output)
                print(abs((sv - output) / sv))
                print((sv - output))
                print(np.ceil((abs((sv - output) / sv) * 100)))
                exit()
            t += da1.time
        t /= 100
        distribution /= 100
        output_file.write("{}\t{}\t{}\n".format(data.split("/")[-1], t, distribution))

def mlb_sample_DA(sweeps=10000, max_rb=273, data_path="/home/QuantumAnnealing/data/5g_raw/3MB/"):
    # pkl_path = "/Users/musktang/Downloads/mlb_data.pkl"
    # # rsrp, sinr, prb = QUBO.data_parse(pkl_path)

    # #### convert pkl file to mmDataFormat txt ####
    # data_path = "/Users/musktang/pycharm_project/mobile-load-balancing/data/raw"
    # QUBO.pkl2txt(pkl_path, data_path)

    #### load raw data from mmDataFormat txt ####
    rsrp, capacity, prb, ue_num, bs_num = QUBO.read_raw_data(
        "{}/{}".format(data_path, "small_sample.txt"))

    #### parameters to QUBO matrix ####
    serving_list = QUBO.init_serving_list(capacity, ue_num)
    # Q = QUBO.params2qubo_tp(rsrp, capacity, prb,
    #                         serving_list, ue_num, bs_num, penalty=10)
    Q, label, H, ineq_constrains, eq_contrains = QUBO.params2qubo_tp_v3(rsrp, capacity, prb,
                            serving_list, ue_num, bs_num)
    # print("Q is symmetry : {}".format(QUBO.check_Q(Q[0])))

    # #### write qubo matrix as txt ####
    # file = "/Users/musktang/pycharm_project/mobile-load-balancing/data/jhmatrix/small_sample.txt"
    # file = "./a.txt"
    # np.savetxt(file,Q[0])
    # # QUBO.write_file(file, Q[0])
    # # QUBO.read_file(file)

    #### gen Q matrix ####
    init_bin = QUBO.init_bin(capacity, len(Q), bs_num)
    init_bin[-1] = 1
    e = init_bin @ Q

    #### add binary label ####
    group = []
    for i in range(ue_num):
        group.append([(bs_num*i+idx) for idx in range(bs_num)])

    bin_lebel, act_idx, label_num = QUBO.onehot(label, group, init_bin)

    #### annealing ####
    da1 = DA(Q, init_bin, maxStep=sweeps,
             betaStart=0.01, betaStop=10, kernel_dim=(1,), bin_label=label,
             act_idx=act_idx, label_num=label_num, tree_n=3, energy=e)

    da1.run()
    # print(da1.binary)
    print(f'time spent: {da1.time}')

    bin1 = np.expand_dims(da1.binary, axis=1)

    # cio_setting = QUBO.init_cio(bs_num)
    # ans_bin = QUBO.gen_answer_tp(serving_list, ue_num,
    #                           bs_num, prb, rsrp, cio_setting)

    #### check answer ####
    throughput = np.matmul(np.matmul(bin1.T, H), bin1)[0]
    constrain_pass = QUBO.check_constrain(bin1, eq_constrains=eq_contrains)
    # no_slack_constrain_pass = QUBO.check_constrain(bin1, Q[3:7], no_slack=True)
    print("check constrain pass : {}".format(constrain_pass))
    # print("check no slack constrain pass : {}".format(no_slack_constrain_pass))
    print("mlb throughput : {}".format(throughput))
    print("answer : {}".format(bin1.T))

    QUBO.check_constrain(bin1, c1=ineq_constrains[0])
    QUBO.check_constrain(bin1, c3=ineq_constrains[1])
    QUBO.check_constrain(bin1, c4=ineq_constrains[2])
    QUBO.check_constrain(bin1, c5=ineq_constrains[3])
    QUBO.check_constrain(bin1, c6=ineq_constrains[4])

if __name__ == '__main__':
    mlb_sample_DA(sweeps=100000, max_rb=40)
    # g05_test(iters=10000, spin=True)
    # bqp_test(iters=10000)
