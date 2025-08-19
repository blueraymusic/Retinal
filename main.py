import subprocess
import os,sys


"""
This main was written to simplify the app_main starts !!!
"""

def main():
    result = subprocess.run(
        ["streamlit", "run", "app/Analyze_Scan.py"],
        capture_output=True,
        text=True
    )
    print("Stdout:", result.stdout)
    print("Stderr:", result.stderr)
    print("Return Code:", result.returncode)

if __name__ == '__main__':
    main()
