import subprocess
import shlex

def run_baseline_command(cmd_string: str):
    cmd = shlex.split(cmd_string)
    return subprocess.call(cmd)
