import subprocess
import modal    
import os
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import uuid
import time

image_with_rust = modal.Image.debian_slim().run_commands(
    ["apt update",
     "apt install -y build-essential curl file git",
     "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
     "/root/.cargo/bin/cargo --version"]
)
image_with_rust = image_with_rust.pip_install_from_requirements("../requirements.txt")

stub = modal.Stub("crossgen",
                  image=image_with_rust,
                  secret=modal.Secret.from_name("crossgen_secrets"),
                  mounts=[modal.Mount(local_dir="/Users/srijithpoduval/Documents/Projects/crossword_generator/src/", remote_dir="/root/")]
                  )

volume = modal.SharedVolume()

@stub.function(schedule=modal.Cron("0 12 * * *"), secret=modal.Secret.from_name("crossgen_secrets"), shared_volumes={"/root/generated_data": volume})
def full_flow():
    import update_word_scores
    import choose_grid
    import generate_puz_file
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
    subprocess.run(["/root/.cargo/bin/cargo", "build"], cwd="fillgrid", check=True)
    subprocess.run(["/root/.cargo/bin/cargo", "run"], cwd="fillgrid", check=True)
    print("filled grid...")

    # then generate the puzzle file
    subprocess.run(["python", "generate_puz_file.py"], check=True)
    print("generated puz file...")

@stub.wsgi(secret=modal.Secret.from_name("crossgen_secrets"), shared_volumes={"/root/generated_data": volume})
def flask_app():
    app = Flask(__name__)

    # @app.route("/", methods=['GET', 'POST'])
    # def generate_puzzle():
    #     full_flow.call()

    @app.route("/sms", methods=['GET', 'POST'])
    def incoming_sms():
        """Send a dynamic reply to an incoming text message"""
        # Get the message the user sent our Twilio number
        body = request.values.get('Body', None)

        # Start our TwiML response
        resp = MessagingResponse()

        # Determine the right reply for this message
        if body == 'UPLOAD_PUZZLE':
            subprocess.run(["python", "upload_puz.py"])
            print("uploading puz file and tweeting out...")
        elif body == 'REDO_PUZZLE':
            full_flow.call()
            print("redoing puz file...")
        else:
            words = body.split()
            words = [word.strip().lower() for word in words]
            words = [word for word in words if word != ""]
            subprocess.run(["python", "generate_puz_file.py"]+words, check=True)
            print("regenerated puz file...")
        return "OK"

    return app


if __name__ == "__main__":
    stub.deploy()