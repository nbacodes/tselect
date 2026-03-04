"""
layout.py
---------
Discovers source files and test files in a repo.

Two modes:
  1. Config-driven (tselect.yaml present with source_dirs/test_dirs)
     → only scans the specified directories — fast and precise
     → this is what PyTorch users will use

  2. Auto-detection (no tselect.yaml or dirs not specified)
     → scans entire repo, infers by language + naming conventions
     → works for any new repo on first run, before user adds tselect.yaml
     → warns user to add tselect.yaml for better performance

The config-driven path solves the 33,782-file problem:
  Before: rglob("*") on all of pytorch/ → 33,782 files, slow
  After:  scan only torch/_inductor/ + test/inductor/ → 464 files, fast
"""

from pathlib import Path
from collections import Counter


LANG_EXT_MAP = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".cpp":  "cpp",
    ".c":    "c",
    ".java": "java",
    ".go":   "go",
}

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec"}

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build",
    "__pycache__", ".venv", ".tox", ".eggs",
}


class RepoLayout:
    def __init__(self, repo_root, language, source_files, test_files):
        self.repo_root    = repo_root
        self.language     = language
        self.source_files = source_files
        self.test_files   = test_files


class RepoLayoutInferer:
    def __init__(self, repo_root: Path, config: dict = None):
        self.repo_root = Path(repo_root)
        self.config    = config or {}

        # pull settings from config, with fallbacks
        graph_cfg        = self.config.get("graph", {})
        repo_cfg         = self.config.get("repo", {})

        self.ignore_dirs  = set(graph_cfg.get("ignore_dirs", DEFAULT_IGNORE_DIRS))
        self.language_cfg = repo_cfg.get("language")      # None = auto-detect
        self.source_dirs  = repo_cfg.get("source_dirs")   # None = auto-detect
        self.test_dirs    = repo_cfg.get("test_dirs")     # None = auto-detect

    # ─────────────────────────────────────────────
    # Language detection (used only in auto mode)
    # ─────────────────────────────────────────────

    def detect_language(self) -> str:
        if self.language_cfg:
            return self.language_cfg

        counter = Counter()
        for path in self.repo_root.rglob("*"):
            if self._should_ignore(path):
                continue
            if path.is_file() and path.suffix in LANG_EXT_MAP:
                counter[LANG_EXT_MAP[path.suffix]] += 1

        if not counter:
            raise RuntimeError(
                "Could not detect repo language.\n"
                "Add a tselect.yaml with 'repo.language: python' (or your language)."
            )

        # prefer python if present (tselect itself is python)
        if "python" in counter:
            return "python"
        return counter.most_common(1)[0][0]

    def _should_ignore(self, path: Path) -> bool:
        return any(p in self.ignore_dirs for p in path.parts)

    # ─────────────────────────────────────────────
    # Test file detection
    # ─────────────────────────────────────────────

    def is_test_file(self, path: Path, language: str) -> bool:
        name = path.name.lower()

        # any file inside a test directory
        if any(part.lower() in TEST_DIR_NAMES for part in path.parts):
            return True

        if language == "python":
            return name.startswith("test_") or name.endswith("_test.py")

        if language in {"javascript", "typescript"}:
            return ".test." in name or ".spec." in name

        if language == "java":
            return name.endswith("test.java") or name.endswith("tests.java")

        return False

    # ─────────────────────────────────────────────
    # Config-driven scan (fast path)
    # ─────────────────────────────────────────────

    def _scan_dirs(self, dirs: list, language: str, want_tests: bool) -> list:
        """
        Scan specific directories for source or test files.
        """
        results = []
        ext = next(
            (e for e, l in LANG_EXT_MAP.items() if l == language),
            ".py"
        )
        # collect all valid extensions for this language
        lang_exts = {e for e, l in LANG_EXT_MAP.items() if l == language}

        for d in dirs:
            full_dir = self.repo_root / d
            if not full_dir.exists():
                print(f"  ⚠️  Directory not found: {full_dir}")
                print(f"       Check 'source_dirs' / 'test_dirs' in tselect.yaml")
                continue

            for path in full_dir.rglob("*"):
                if self._should_ignore(path):
                    continue
                if not path.is_file():
                    continue
                if path.suffix not in lang_exts:
                    continue

                is_test = self.is_test_file(path, language)

                if want_tests and is_test:
                    results.append(path)
                elif not want_tests and not is_test:
                    results.append(path)

        return results

    # ─────────────────────────────────────────────
    # Auto-detection scan (slow path, no config)
    # ─────────────────────────────────────────────

    def _auto_scan(self, language: str) -> tuple[list, list]:
        """
        Scan entire repo, classify files by name/location.
        Used when no tselect.yaml is present.
        """
        source_files = []
        test_files   = []

        for path in self.repo_root.rglob("*"):
            if self._should_ignore(path):
                continue
            if not path.is_file():
                continue
            if path.suffix not in LANG_EXT_MAP:
                continue

            if self.is_test_file(path, language):
                test_files.append(path)
            else:
                source_files.append(path)

        return source_files, test_files

    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────

    def infer(self) -> RepoLayout:
        language = self.detect_language()

        if self.source_dirs and self.test_dirs:
            # ── FAST PATH: user specified dirs in tselect.yaml ──
            source_files = self._scan_dirs(self.source_dirs, language, want_tests=False)
            test_files   = self._scan_dirs(self.test_dirs,   language, want_tests=True)

        elif self.source_dirs and not self.test_dirs:
            # partial config — source specified but not tests
            source_files = self._scan_dirs(self.source_dirs, language, want_tests=False)
            # auto-detect test dirs
            _, test_files = self._auto_scan(language)
            print("  ℹ️  'test_dirs' not set in tselect.yaml — auto-detecting tests")

        else:
            # ── SLOW PATH: no config, scan entire repo ──
            print()
            print("  ℹ️  No tselect.yaml found — scanning entire repo (may be slow).")
            print("     Add a tselect.yaml to speed this up significantly.")
            print("     Run: tselect init   (coming soon — generates tselect.yaml for you)")
            print()
            source_files, test_files = self._auto_scan(language)

        return RepoLayout(
            repo_root    = self.repo_root,
            language     = language,
            source_files = source_files,
            test_files   = test_files,
        )
