import subprocess

def get_changed_files(base="upstream/main", target="HEAD"):
    try:
        # Use only the top commit — works for ghstack/PR branches
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{target}~1...{target}"],
            capture_output=True, text=True, check=True,
        )
        files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        if files:
            return files

        # Fallback: merge-base diff
        merge_base = subprocess.run(
            ["git", "merge-base", target, base],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{merge_base}...{target}"],
            capture_output=True, text=True, check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]

    except Exception as e:
        print("Failed to detect changed files from git:", e)
        return []
