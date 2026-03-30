"""
init_command.py
---------------
tselect init — interactively generates tselect.yaml for any repo.

Design:
  - Scans repo, finds candidate source and test directories
  - Aggregates at depth-1 (top-level dirs) for clean defaults
  - Smart default: test dirs matched by name heuristic (test/, tests/, etc.)
                   source dirs matched by highest file count
  - User picks by number — no typing paths
  - Prompts for Groq API key (used by AI filtering layer)
  - Writes tselect.yaml to repo root
  - Works for any Python repo on first run
"""

from pathlib import Path
from collections import defaultdict


IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "node_modules",
    "build", "dist", ".tox", ".eggs", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".graph",
    "third_party", "vendor", "extern", "external",
    "submodules", "deps", "dependencies",
}

LANG_EXT_MAP = {
    ".py":   "python",
    ".java": "java",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".go":   "go",
    ".cpp":  "cpp",
}

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec"}


def _should_ignore(path: Path) -> bool:
    return any(p in IGNORE_DIRS for p in path.parts)


def _detect_language(repo_root: Path) -> str:
    from collections import Counter
    counter = Counter()
    for path in repo_root.rglob("*"):
        if _should_ignore(path):
            continue
        if path.is_file() and path.suffix in LANG_EXT_MAP:
            counter[LANG_EXT_MAP[path.suffix]] += 1
    if not counter:
        return "python"
    if "python" in counter:
        return "python"
    return counter.most_common(1)[0][0]


def _find_candidate_dirs(repo_root: Path, language: str) -> tuple[list, list]:
    """
    Walk repo and count source/test files per TOP-LEVEL subdirectory (depth-1).

    Depth-1 aggregation means:
        torch/_inductor/scheduler.py  → counted under "torch"
        test/inductor/test_scheduler.py → counted under "test"

    This gives clean defaults like "torch/" and "test/" instead of
    "torch/_inductor" and "test/inductor".

    Returns sorted (dir, file_count) lists for source and test dirs.
    No directory names are hardcoded — classification is purely by:
      - Whether any path component matches TEST_DIR_NAMES
      - Whether the file is named test_*.py / *_test.py
    """
    source_counts = defaultdict(int)
    test_counts   = defaultdict(int)

    ext_map = {e for e, l in LANG_EXT_MAP.items() if l == language}

    for path in repo_root.rglob("*"):
        if _should_ignore(path):
            continue
        if not path.is_file():
            continue
        if path.suffix not in ext_map:
            continue

        try:
            rel   = path.relative_to(repo_root)
            parts = rel.parts

            # depth-1: always aggregate under the top-level subdir
            top_dir = parts[0] if len(parts) >= 2 else parts[0]

            is_test = (
                any(part.lower() in TEST_DIR_NAMES for part in parts) or
                path.name.startswith("test_") or
                path.name.endswith("_test.py")
            )

            if is_test:
                test_counts[top_dir] += 1
            else:
                source_counts[top_dir] += 1

        except ValueError:
            continue

    # sort by file count descending
    source_dirs = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    test_dirs   = sorted(test_counts.items(),   key=lambda x: x[1], reverse=True)

    return source_dirs, test_dirs


def _default_index(candidates: list, is_test: bool) -> int:
    """
    Return the 0-based index of the best default candidate.

    Source dirs: highest file count (index 0 after sort).
    Test dirs:   first candidate whose top-level name matches TEST_DIR_NAMES.
                 Falls back to index 0 if none match.

    No directory names are hardcoded — matching is done against
    TEST_DIR_NAMES which is the shared canonical set.
    """
    if not is_test:
        return 0  # highest file count wins for source

    for i, (d, _) in enumerate(candidates):
        top = Path(d).parts[0].lower()
        if top in TEST_DIR_NAMES:
            return i

    return 0  # fallback


def _prompt_dirs(label: str, candidates: list, is_test: bool, max_show: int = 10) -> list:
    """
    Show numbered list, let user pick one or more by number.

    The default (Enter with no input) is determined by _default_index()
    so it is always the smartest choice without any hardcoding.

    Returns list of selected dir strings.
    """
    default_idx  = _default_index(candidates, is_test)
    default_dir  = candidates[default_idx][0]

    print(f"\n  Candidate {label} directories (by file count):")
    print()

    shown = candidates[:max_show]
    for i, (d, count) in enumerate(shown, 1):
        marker = " ◀ default" if i - 1 == default_idx else ""
        print(f"    {i:2d}.  {d:<45}  ({count} files){marker}")

    if len(candidates) > max_show:
        print(f"    ... and {len(candidates) - max_show} more")

    print()
    print(f"  Enter numbers to select (e.g. 1,2  or just 1):")
    print(f"  Press Enter to accept default ({default_dir})")
    print()

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""

        if not raw:
            return [default_dir]

        try:
            indices  = [int(x.strip()) for x in raw.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(shown):
                    selected.append(shown[idx - 1][0])
                else:
                    print(f"  ⚠️  {idx} is out of range, ignoring")

            if selected:
                return selected
            else:
                print("  No valid selection, try again.")

        except ValueError:
            print("  Please enter numbers like: 1  or  1,2,3")


def _prompt_groq_api_key() -> str:
    """
    Prompt the user for their Groq API key.

    The key is used by tselect's AI filtering layer (pre_filter.py)
    to intelligently reduce false-positive test selections using an LLM.

    The key is written to tselect.yaml under ai.groq_api_key.
    Users can skip this and add the key manually later.
    """
    print()
    print("  ─" * 30)
    print("  AI Filtering Layer (Groq)")
    print("  ─" * 30)
    print()
    print("  tselect uses Groq's LLM API to filter out false-positive")
    print("  test selections. This requires a free Groq API key.")
    print()
    print("  Get one at: https://console.groq.com/keys")
    print()
    print("  Enter your Groq API key (or press Enter to skip):")
    print()

    try:
        key = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if key:
        print("  ✅ Groq API key saved.")
    else:
        print("  ⏭️  Skipped. Add it later under 'ai.groq_api_key' in tselect.yaml")

    return key


def _write_yaml(
    repo_root: Path,
    language: str,
    source_dirs: list,
    test_dirs: list,
    groq_api_key: str = "",
):
    config_path = repo_root / "tselect.yaml"

    src_lines  = "\n".join(f"    - {d}" for d in source_dirs)
    test_lines = "\n".join(f"    - {d}" for d in test_dirs)

    # Only write the key field if the user provided one,
    # otherwise leave a clear placeholder comment
    if groq_api_key:
        ai_key_line = f"  groq_api_key: {groq_api_key}"
    else:
        ai_key_line = "  groq_api_key: \"\"   # Add your key: https://console.groq.com/keys"

    content = f"""# tselect.yaml — generated by 'tselect init'
# Commit this file to your repo.

repo:
  language: {language}
  source_dirs:
{src_lines}
  test_dirs:
{test_lines}

graph:
  rebuild_after_days: 7
  collect_batch_size: 50

runner:
  extra_args: []
  ignore_changed_patterns:
    - "*.json"
    - "*.yaml"
    - "*.yml"
    - "*.csv"
    - "*.db"
    - "*.md"
    - "*.txt"
    - "*.lock"
    - "*.toml"
    - "*.cfg"
    - "*.ini"
    - "*.png"
    - "*.jpg"

ci:
  post_pr_comment: false
  artifact_storage: none
  fail_on_test_failure: true

ai:
  enabled: true
  model: llama-3.3-70b-versatile
{ai_key_line}
  timeout: 15
  confidence_threshold: 0.75

ml:
  enabled: false
"""
    config_path.write_text(content)
    return config_path


def run_init(repo_root: Path):
    """
    Main entry point for `tselect init`.
    """
    config_path = repo_root / "tselect.yaml"

    print()
    print("=" * 60)
    print("  tselect init — Configure this repo")
    print("=" * 60)

    if config_path.exists():
        print(f"\n  tselect.yaml already exists at {config_path}")
        try:
            overwrite = input("  Overwrite it? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            overwrite = "n"
        if overwrite != "y":
            print("  Aborted. Existing tselect.yaml unchanged.")
            return

    print()
    print("  Scanning repo to find source and test directories...")
    print("  (This may take a few seconds on large repos)")

    language = _detect_language(repo_root)
    print(f"\n  Detected language: {language}")

    if language not in {"python", "java", "javascript", "typescript", "go"}:
        print(f"\n  ⚠️  Language '{language}' is not fully supported yet.")
        print(f"     Supported: python")
        print(f"     Coming soon: java, javascript, typescript, go")

    source_candidates, test_candidates = _find_candidate_dirs(repo_root, language)

    if not source_candidates:
        print("\n  ❌ Could not find any source files. Is this the right directory?")
        return

    if not test_candidates:
        print("\n  ❌ Could not find any test files.")
        print("     Make sure your test files follow naming conventions:")
        print("       Python: test_*.py or *_test.py")
        return

    # ── interactive selection ──
    source_dirs  = _prompt_dirs("SOURCE", source_candidates, is_test=False)
    test_dirs    = _prompt_dirs("TEST",   test_candidates,   is_test=True)
    groq_api_key = _prompt_groq_api_key()

    # ── confirm ──
    print()
    print("  ─" * 30)
    print(f"  language    : {language}")
    print(f"  source_dirs : {source_dirs}")
    print(f"  test_dirs   : {test_dirs}")
    print(f"  groq key    : {'set ✅' if groq_api_key else 'not set ⏭️'}")
    print("  ─" * 30)
    print()

    try:
        confirm = input("  Write tselect.yaml with these settings? (Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "y"

    if confirm in {"", "y"}:
        path = _write_yaml(repo_root, language, source_dirs, test_dirs, groq_api_key)
        print(f"\n  ✅ tselect.yaml written to: {path}")
        print()
        print("  Next steps:")
        print("    tselect build-graph   # build dependency graph")
        print("    tselect run           # select tests for changed files")
    else:
        print("  Aborted. No file written.")