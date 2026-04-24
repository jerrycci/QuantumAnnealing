# client_keygen.py
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import os


def generate_user_keys(username):
    # 產生金鑰對
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # 存私鑰 (使用者自己留著)
    priv_filename = f"{username}_private.pem"
    with open(priv_filename, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # 存公鑰 (這檔案要寄給 Server 管理員)
    pub_filename = f"{username}_public.pem"
    with open(pub_filename, "wb") as f:
        f.write(private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print(f"Done! Keep '{priv_filename}' safe.")
    print(f"Send '{pub_filename}' to the Server admin.")


if __name__ == "__main__":
    user = input("Enter your username (e.g., client_A): ").strip()
    generate_user_keys(user)