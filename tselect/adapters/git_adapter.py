import subprocess


def get_changed_files():
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        return files
    except Exception as e:
        print("Failed to detect changed files from git:", e)
        return []
