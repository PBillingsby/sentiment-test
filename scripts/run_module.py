from config.constants import MODULE_REPO, TARGET_COMMIT, WEB3_PRIVATE_KEY
import argparse
import subprocess

parser = argparse.ArgumentParser(
    description="Run the Lilypad module with specified input."
)

parser.add_argument(
    "input", type=str, help="The input to be processed by the Lilypad module."
)

args = parser.parse_args()

input = args.input

command = [
    "lilypad",
    "run",
    "--target",
    "0xC44CB6599bEc03196fD230208aBf4AFc68514DD2",
    f"{MODULE_REPO}:{TARGET_COMMIT}",
    "--web3-private-key",
    WEB3_PRIVATE_KEY,
    "-i",
    f"input={input}",
]

try:
    result = subprocess.run(command, check=True, text=True)
    print("Lilypad module executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
