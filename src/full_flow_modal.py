import subprocess
import modal 
import os

import update_word_scores
import choose_grid
import generate_puz_file

stub = modal.Stub()

image_with_rust = modal.Image.debian_slim().run_commands(
    ["apt update",
     "apt install -y build-essential curl file git",
     "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
     "/root/.cargo/bin/cargo --version"]
)
image_with_rust = image_with_rust.pip_install_from_requirements("../requirements.txt")

@stub.function(secret=modal.Secret.from_name("crossgen_secrets"),
                mounts=[modal.Mount(local_dir="/Users/srijithpoduval/Documents/Projects/crossword_generator/src/raw_data", remote_dir="/root/raw_data"),
                        modal.Mount(local_dir="/Users/srijithpoduval/Documents/Projects/crossword_generator/src/fillgrid", remote_dir="/root/fillgrid")],
                image=image_with_rust)
def full_flow():
    # create generated_data folder
    if not os.path.exists("generated_data"):
        os.makedirs("generated_data")    

    # first run choose grid
    subprocess.run(["python", "update_word_scores.py"], check=True)
    print("making sure prev words are not used...")

    # first run choose grid
    subprocess.run(["python", "choose_grid.py"], check=True)
    print("chose grid...")

    # then fill in the grid
    # subprocess.run(["cargo", "clean"], cwd="fillgrid", check=True)
    # subprocess.run(["cargo", "update"], cwd="fillgrid", check=True)
    subprocess.run(["/root/.cargo/bin/cargo", "build"], cwd="fillgrid", check=True)
    subprocess.run(["/root/.cargo/bin/cargo", "run"], cwd="fillgrid", check=True)
    print("filled grid...")

    # then generate the puzzle file
    subprocess.run(["python", "generate_puz_file.py"], check=True)
    print("generated puz file...")

    # then upload the crossword and tweet it out
    # subprocess.run(["python", "upload_puz.py"], check=True)
    # print("uploading puz file and tweeting out...")


if __name__ == "__main__":
    with stub.run():
        full_flow.call()