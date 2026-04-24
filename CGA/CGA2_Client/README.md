# CGA_Client

Client-side scripts for accessing **Compal GPU Annealer (CGA)** with token-based authentication.

## Overview

This folder contains simple Python utilities for:

- generating a public/private key pair
- connecting to a CGA server
- submitting a QUBO problem
- retrieving optimization results

This client is intended for **remote CGA usage** and does **not** require local solver installation.

---

## Contents

- `keygen.py`  
  Generate an RSA key pair for authentication.

- `clients_token.py`  
  Submit a QUBO job to the CGA server using your private key.

---

## Requirements

Recommended Python version:

- Python 3.10

Install required packages:

```bash
pip install python-socketio
pip install "python-socketio[client]"
pip install ttictoc
pip install pyqubo
pip install cryptography
pip install pyjwt