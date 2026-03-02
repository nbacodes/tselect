from pathlib import Path
from collections import Counter

IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", "__pycache__", ".venv", ".tox"
}

LANG_EXT_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".go": "go",
}

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec"}


class RepoLayout:
    def __init__(self, repo_root, language, source_files, test_files):
        self.repo_root = repo_root
        self.language = language
        self.source_files = source_files
        self.test_files = test_files


class RepoLayoutInferer:

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)

    # -----------------------
    # Detect language
    # -----------------------
    def detect_language(self):
        counter = Counter()

        for path in self.repo_root.rglob("*"):
            if any(p in IGNORE_DIRS for p in path.parts):
                continue
            if path.is_file():
                ext = path.suffix
                if ext in LANG_EXT_MAP:
                    counter[LANG_EXT_MAP[ext]] += 1

        if not counter:
            raise RuntimeError("Could not detect repo language.")

        # Prefer Python if present (since TSelect itself is Python)
        if "python" in counter:
            return "python"

        return counter.most_common(1)[0][0]

    # -----------------------
    # Classify test files
    # -----------------------
    def is_test_file(self, path: Path, language: str):
        name = path.name.lower()

        if any(part.lower() in TEST_DIR_NAMES for part in path.parts):
            return True

        if language == "python":
            return name.startswith("test_") or name.endswith("_test.py")

        if language in {"javascript", "typescript"}:
            return ".test." in name or ".spec." in name

        return False

    # -----------------------
    # Build layout
    # -----------------------
    def infer(self):
        language = self.detect_language()

        source_files = []
        test_files = []

        for path in self.repo_root.rglob("*"):
            if any(p in IGNORE_DIRS for p in path.parts):
                continue
            if not path.is_file():
                continue

            if path.suffix not in LANG_EXT_MAP:
                continue

            if self.is_test_file(path, language):
                test_files.append(path)
            else:
                source_files.append(path)

        return RepoLayout(
            repo_root=self.repo_root,
            language=language,
            source_files=source_files,
            test_files=test_files,
        )
