import socketio
import pickle
import argparse
import random
import os
import jwt  # pip install pyjwt
import datetime
from ttictoc import tic, toc
from pyqubo import Array


# ======================================================
#                 INIT SPIN SUPPORT
# ======================================================

def validate_init_spin_file(filepath, expected_vars):
    """
    Validate initial spin file format and dimension.
    Conditions:
    1. Only 0 or 1 allowed.
    2. Must have at least one line.
    3. Each line must be space-separated bits.
    4. All lines must have equal number of bits.
    5. Bit count must match expected_vars (QUBO size).
    """
    if not os.path.exists(filepath):
        raise ValueError(f"Init spin file does not exist: {filepath}")

    lines = []
    with open(filepath, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            # forbid empty line
            if not line:
                raise ValueError(f"Init spin file has empty line at line {line_no}")

            bits = line.split()
            if len(bits) == 0:
                raise ValueError(f"Line {line_no} is empty or malformed.")

            # number of bits shoud be the same as the number of Qubo variables
            if len(bits) != expected_vars:
                raise ValueError(
                    f"Dimension Mismatch at line {line_no}: "
                    f"File has {len(bits)} bits, but QUBO needs {expected_vars} variables."
                )

            # (1) Only 0 or 1 allowed
            for b in bits:
                if b not in ("0", "1"):
                    raise ValueError(
                        f"Invalid bit '{b}' at line {line_no}. Only 0 or 1 allowed."
                    )

            lines.append(bits)

    # (2) Must have at least 1 line
    if len(lines) == 0:
        raise ValueError("Init spin file contains no valid lines.")

    # (4) All lines must have same length
    expected_len = len(lines[0])
    for idx, row in enumerate(lines, start=1):
        if len(row) != expected_len:
            raise ValueError(
                f"Line {idx} has {len(row)} bits, but expected {expected_len} bits."
            )

    print(f"[Client] Init spin file OK ({len(lines)} chains, {expected_len} bits each). Match QUBO size.")
    return True


def load_init_spin_file(path):
    """
    Load init_spin file where each line is space-separated 0/1 bits.
    RETURN: list of lists (one chain seed per line)
    """
    spin_list = []
    with open(path) as f:
        for line in f:
            bits = [int(x) for x in line.strip().split()]
            if len(bits) > 0:
                spin_list.append(bits)
    return spin_list


# ======================================================
#                     CLIENT PART
# ======================================================

# For local testing or production
server_ip = "60.250.149.247"
server_port = 8080
server_uri = f"http://{server_ip}:{server_port}"

sio = socketio.Client()

offset = 0


@sio.event
def connect():
    print(f"Connected to Compal GPU Annealer 2.0 Cloud Service server at {server_uri}")


@sio.event
def disconnect():
    print("Disconnected from Compal GPU Annealer 2.0 Cloud Service server")


@sio.on('job token')
def on_response(data):
    print(f"Receive job_token: {data}")


@sio.on("upload error")
def on_response(data):
    print(f"Received response: {data['message']}")
    sio.disconnect()


@sio.on("job finished")
def job_response(data_dict):
    job_token = data_dict["job_token"]
    sio.emit("request solution", job_token)


@sio.on("job error")
def job_response(data_dict):
    message = data_dict["message"]
    print(message)
    sio.disconnect()


@sio.on("return solution")
def handle_solution(result):
    result_decode = pickle.loads(result)
    sampleset = result_decode["sampleset"]
    info = result_decode["info"]
    run_time = result_decode["run_time"]

    print(sampleset)
    print(info)
    print("run_time:", run_time)

    for i, sample in enumerate(sampleset.record):
        print("sample:", sample[0], len(sample[0]))
        print("energy:", sample[1] + offset)

    sio.sleep(1)
    sio.disconnect()


@sio.on("return message")
def handle_solution(result):
    message = pickle.loads(result)
    print(message)
    sio.disconnect()


# ======================================================
#                     MAIN ENTRY
# ======================================================

def main():
    global offset
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_spin", type=str, default=None,
                        help="Path to init spin file (optional)")
    parser.add_argument("--timeout", type=int, default=10)

    # [Authentication]
    # 使用者只需提供私鑰路徑，不需提供使用者名稱
    parser.add_argument("--private_key", type=str, default=None,
                        help="Path to your private key (.pem) for VIP authentication")

    args = parser.parse_args()

    # -------------------------------
    # Create your Hamiltonian
    # -------------------------------
    n_node = 10
    x = Array.create('c', (n_node, n_node), 'BINARY')

    H = 0
    for i in range(n_node):
        for j in range(n_node):
            H += (i + 1) * (j + 1) * x[i][j] * x[(j + 1) % n_node, j]

    # Compile model
    model = H.compile()
    # Use to_qubo() to get qubo and offset
    qubo, offset = model.to_qubo(index_label=True)

    # -------------------------------
    # Check QUBO variables
    # -------------------------------
    if not qubo:
        num_vars = 0
    else:
        max_index = 0
        for (u, v) in qubo.keys():
            m = max(u, v)
            if m > max_index:
                max_index = m
        num_vars = max_index + 1

    # -------------------------------
    # Prepare INIT SPIN
    # -------------------------------
    init_spin_list = None
    if args.init_spin:
        validate_init_spin_file(args.init_spin, num_vars)
        init_spin_list = load_init_spin_file(args.init_spin)
        print(f"[Client] Loaded {len(init_spin_list)} chains from init_spin.")

    # -------------------------------
    # Build request dict
    # -------------------------------
    request_dict = {
        "qubo": qubo,
        "offset": offset,
        "timeout": args.timeout,
    }

    if init_spin_list is not None:
        request_dict["init_spin"] = init_spin_list

    # ==============================================================================
    # [AUTH] Token Generation (Server Auto-Match)
    # ==============================================================================
    if args.private_key:
        if os.path.exists(args.private_key):
            try:
                # 1. 讀取私鑰
                with open(args.private_key, 'rb') as f:
                    private_key_data = f.read()

                # 2. 建立 Payload (不需 sub，Server 會自動嘗試匹配公鑰)
                payload = {
                    "iat": datetime.datetime.utcnow(),
                    "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=60)  # Token 60秒有效
                }

                # 3. 簽名並放入請求
                token = jwt.encode(payload, private_key_data, algorithm="RS256")
                request_dict["auth_token"] = token
                print(f"[Auth] Token generated using {args.private_key}. Waiting for server verification...")

            except Exception as e:
                print(f"[Auth] Error generating token: {e}. Falling back to IP authentication.")
        else:
            print(f"[Auth] Private key file not found: {args.private_key}. Falling back to IP authentication.")
    else:
        print("[Auth] No private key provided. Using Standard IP-based authentication.")

    # -------------------------------
    # Encode for transmission
    # -------------------------------
    qubo_data = pickle.dumps(request_dict, protocol=2)

    # -------------------------------
    # Connect and send to server
    # -------------------------------
    try:
        sio.connect(server_uri)
        print("my sid is", sio.sid)

        sio.emit("upload qubo", qubo_data)
        print("Emitted event: upload qubo")

        tic()
        sio.wait()
        print("Client total time:", toc())
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        if sio.connected:
            sio.disconnect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Client failed: {e}")
        if sio.connected:
            sio.disconnect()