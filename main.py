import subprocess
import os,sys


"""
This main was written to simplify the app_main starts !!!
"""

def main():
    while True:
        print("Checking for the environment key ...")
        if "OPENAI_API_KEY" in os.environ:
            print("... Checked (Env. key set!)")
            break
        else:
            print("... Not Set!!, set it to move further -")
            print(".. or input (quit) in order to leave!")
            key = input("Openai key: ")
            if key.strip().lower() == "quit":
                sys.exit(-1)
            else:
                os.environ["OPENAI_API_KEY"] = key.strip().lower()
                break

    result = subprocess.run(
        ["streamlit", "run", "app_main.py"],
        capture_output=True,
        text=True
    )
    print("Stdout:", result.stdout)
    print("Stderr:", result.stderr)
    print("Return Code:", result.returncode)

if __name__ == '__main__':
    main()
