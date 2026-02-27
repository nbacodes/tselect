from pathlib import Path
from tselect.utils.loader import load_json

def detect_baseline_command(repo_root: Path):
    """
    Detect baseline command generically using project JSON.
    """

    # Look for config JSON
    json_path = repo_root / "config" / "testSuiteTorchInductor.json"

    if json_path.exists():
        test_json = load_json(json_path)

        test_root = test_json.get("test_root")

        if test_root:
            print("\nDetected PyTest project")
            print(f"Using test_root from JSON: {test_root}")

            # Currently for pytest
            return [
                "python",
                "-m",
                "pytest",
                test_root,
                "--continue-on-collection-errors",
                "-q",
            ]

    # fallback (generic)
    print("\nNo JSON found — using generic pytest baseline")
    return ["python", "-m", "pytest"]
