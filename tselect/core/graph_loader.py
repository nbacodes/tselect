import json
from pathlib import Path


class GraphLoader:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.graph_path = (
            self.repo_root / ".graph" / "tselect" / "dependency_graph.json"
        )

    def exists(self):
        return self.graph_path.exists()

    def load(self):
        if not self.exists():
            raise RuntimeError(
                "No dependency graph found.\n"
                "Run: tselect build-graph"
            )

        with open(self.graph_path, "r") as f:
            return json.load(f)
