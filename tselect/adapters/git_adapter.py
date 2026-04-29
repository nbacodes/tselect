import subprocess

def get_changed_files(base="upstream/main", target="HEAD"):
    try:
        # Count commits ahead of upstream/main
        merge_base = subprocess.run(
            ["git", "merge-base", target, base],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        commit_count = subprocess.run(
            ["git", "rev-list", "--count", f"{merge_base}...{target}"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        n = int(commit_count)

        # If small number of commits, diff all of them from merge-base
        if 1 <= n <= 10:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{merge_base}...{target}"],
                capture_output=True, text=True, check=True,
            )
        else:
            # Too many commits — likely a stale branch, use top commit only
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{target}~1...{target}"],
                capture_output=True, text=True, check=True,
            )

        files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        return files

    except Exception as e:
        print("Failed to detect changed files from git:", e)
        return []
