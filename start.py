import subprocess

def run_chainlit():
    command = ["chainlit", "run", "main.py", "-w"]
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nChainlit已停止.")

if __name__ == "__main__":
    run_chainlit()
