import json
import ast
from pathlib import Path
from collections import defaultdict


class GraphBuilder:
    def __init__(self, layout):
        self.layout = layout
        self.repo_root = layout.repo_root
        self.language = layout.language
        self.source_files = layout.source_files
        self.test_files = layout.test_files

    # ---------------------------
    # Python import extraction
    # ---------------------------
    def _extract_python_imports(self, file_path: Path):
        imports = set()

        try:
            tree = ast.parse(file_path.read_text())
        except Exception:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return imports

    # ---------------------------
    # Generic import dispatcher
    # ---------------------------
    def extract_imports(self, file_path: Path):
        if self.language == "python":
            return self._extract_python_imports(file_path)

        # future: tree-sitter integration
        return set()

    # ---------------------------
    # Build dependency graph
    # ---------------------------
    def build(self):
        forward_graph = defaultdict(set)

        # Map module path to file
        module_map = {}

        for src in self.source_files:
            rel = src.relative_to(self.repo_root)
            module_name = ".".join(rel.with_suffix("").parts)
            module_map[module_name] = str(rel)

        # Build forward graph
        for src in self.source_files:
            rel = src.relative_to(self.repo_root)
            module_name = ".".join(rel.with_suffix("").parts)

            imports = self.extract_imports(src)

            for imp in imports:
                if imp in module_map:
                    forward_graph[module_name].add(imp)

        # Reverse graph: source_file → tests
        reverse_graph = defaultdict(set)

        for test in self.test_files:
            imports = self.extract_imports(test)

            rel_test = str(test.relative_to(self.repo_root))

            for imp in imports:
                if imp in module_map:
                    src_file = module_map[imp]
                    reverse_graph[src_file].add(rel_test)

        return {
            "language": self.language,
            "full_reverse_graph": {
                k: sorted(list(v)) for k, v in reverse_graph.items()
            }
        }

    # ---------------------------
    # Save graph to disk
    # ---------------------------
    def save(self, graph_data):
        graph_dir = self.repo_root / ".graph" / "tselect"
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_path = graph_dir / "dependency_graph.json"

        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_path
