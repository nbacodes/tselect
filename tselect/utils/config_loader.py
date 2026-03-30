"""
config_loader.py
----------------
Loads tselect.yaml from the repo root.

Design:
  - Every field has a hardcoded default so tselect works even with
    a minimal tselect.yaml (just source_dirs + test_dirs)
  - If tselect.yaml is missing entirely, falls back to auto-detection
    so existing repos don't break
  - Config is a plain dict — no magic classes, easy to inspect/debug
"""

from pathlib import Path
from .loader import load_yaml

# ── Defaults ──────────────────────────────────────────────────────────────────
# These apply when tselect.yaml is missing or a field is not specified.

DEFAULTS = {
    "repo": {
        "language":    None,   # None = auto-detect
        "source_dirs": None,   # None = auto-detect via layout inferer
        "test_dirs":   None,   # None = auto-detect via layout inferer
    },
    "graph": {
        "rebuild_after_days":  7,
        "collect_batch_size":  50,
        "ignore_dirs": [
            ".git", "__pycache__", ".venv", "node_modules",
            "build", "dist", ".tox", ".eggs", "*.egg-info",
        ],
    },
    "runner": {
        "extra_args": [],
        "ignore_changed_patterns": [
            "*.json", "*.yaml", "*.yml", "*.csv", "*.db",
            "*.md", "*.txt", "*.lock", "*.toml", "*.cfg",
            "*.ini", "*.png", "*.jpg", "*.jpeg", "*.svg",
        ],
    },
    "ci": {
        "post_pr_comment":      False,
        "artifact_storage":     "none",
        "fail_on_test_failure": True,
    },
    "ml": {
        "enabled":          False,
        "min_history_days": 14,
    },
    "ai": {
        "enabled":              True,
        "groq_api_key":         "",     # set via tselect init or manually in yaml
        "model":                "llama-3.3-70b-versatile",
        "timeout":              15,
        "confidence_threshold": 0.75,
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """
    Merge overrides into defaults, recursively for nested dicts.
    Override values win; missing override keys keep defaults.
    """
    result = dict(defaults)
    for key, value in overrides.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_tselect_config(repo_root: Path) -> dict:
    """
    Load tselect.yaml from repo root, merging with defaults.

    Returns a complete config dict with all fields guaranteed present.
    Callers never need to check for missing keys.

    Usage:
        config = load_tselect_config(repo_root)
        source_dirs  = config["repo"]["source_dirs"]          # list or None
        batch_size   = config["graph"]["collect_batch_size"]  # int
        groq_api_key = config["ai"]["groq_api_key"]           # str or ""
    """
    config_path = Path(repo_root) / "tselect.yaml"

    if not config_path.exists():
        return _deep_merge(DEFAULTS, {})

    try:
        user_config = load_yaml(config_path) or {}
    except Exception as e:
        print(f"  ⚠️  Could not parse tselect.yaml: {e}")
        print(f"       Using defaults.")
        return _deep_merge(DEFAULTS, {})

    # backward compat: old configs used ai.api_key, new ones use ai.groq_api_key
    # if someone has the old key, migrate it transparently
    ai_section = user_config.get("ai", {})
    if "api_key" in ai_section and "groq_api_key" not in ai_section:
        ai_section["groq_api_key"] = ai_section.pop("api_key")
        user_config["ai"] = ai_section

    return _deep_merge(DEFAULTS, user_config)


def should_ignore_file(filename: str, patterns: list) -> bool:
    """
    Check if a changed file should be ignored based on patterns.
    Uses simple fnmatch-style glob matching on filename only.

    Examples:
        should_ignore_file("accuracy.csv", ["*.csv"]) → True
        should_ignore_file("scheduler.py", ["*.csv"]) → False
    """
    import fnmatch
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False