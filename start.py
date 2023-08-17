import subprocess

def run_chainlit():
    command = ["chainlit", "run", "main.py", "-w"]
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nChainlit process terminated by user.")

if __name__ == "__main__":
    run_chainlit()
