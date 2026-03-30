"""
fn_diff.py
----------
Tree-sitter based function/class extractor for changed lines.

Replaces the ast-based _functions_at_lines() in diff_parser.py.
Supports: .py, .cpp, .cu, .cuh, .h, .hpp, .cc, .c

Drop-in usage:
    from tselect.core.fn_diff import extract_symbols_at_lines
    symbols = extract_symbols_at_lines(file_path, changed_lines)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter availability check (graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────

_TS_AVAILABLE = False
try:
    from tree_sitter import Language, Parser as _TSParser
    _TS_AVAILABLE = True
except ImportError:
    pass

# Cache parsers so we don't reinstantiate on every file
_parser_cache: dict = {}

CPP_EXTENSIONS = {'.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.c'}
PY_EXTENSIONS  = {'.py'}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_symbols(file_path: Path) -> set[str]:
    """
    Extract ALL function/class names from a file — no line filtering.

    Used by graph_builder to build the public symbol index for source files.
    Supports .py (tree-sitter → ast fallback) and .cpp/.cu/.h (tree-sitter).

    Examples:
        torch/optim/sgd.py      → {"SGD", "SGD.step", "SGD.zero_grad", ...}
        torch/csrc/jit/ir.cpp   → {"Graph::insertNode", "Node::replaceInput", ...}
    """
    suffix = file_path.suffix.lower()
    if suffix in PY_EXTENSIONS:
        parser = _get_parser('python')
        if parser:
            try:
                tree = parser.parse(file_path.read_bytes())
                return {name for _, _, name in _collect_definitions(tree.root_node, 'python')}
            except Exception:
                pass
        return _ast_all_symbols(file_path)

    elif suffix in CPP_EXTENSIONS:
        parser = _get_parser('cpp')
        if parser:
            try:
                tree = parser.parse(file_path.read_bytes())
                return {name for _, _, name in _collect_definitions(tree.root_node, 'cpp')}
            except Exception:
                pass

    return set()


def get_all_identifiers(file_path: Path) -> set[str]:
    """
    Extract ALL identifiers referenced in a file.

    Used by graph_builder to build file_identifiers for the BFS stopping
    condition in graph_selector — "does this importer use anything that changed?"

    .py  → ast walk (Name + Attribute nodes)
    .cpp/.cu/.h → tree-sitter identifier node walk
    """
    suffix = file_path.suffix.lower()
    if suffix in PY_EXTENSIONS:
        return _ast_all_identifiers(file_path)
    elif suffix in CPP_EXTENSIONS:
        parser = _get_parser('cpp')
        if parser:
            return _ts_cpp_identifiers(parser, file_path)
    return set()


def extract_symbols_at_lines(file_path: Path, changed_lines: set[int]) -> set[str]:
    """
    Return the set of function/class names that contain any of the changed lines.

    Args:
        file_path:     absolute path to the source file
        changed_lines: 1-based line numbers from git diff

    Returns:
        - {"__unknown__"}          if parsing fails or lines can't be attributed
        - {"__module__", ...}      if some lines fall outside all definitions
        - {"ClassName.method", ...} for Python method-inside-class
        - {"function_name", ...}   for top-level or C++ functions

    Falls back gracefully:
        .py  → ast fallback if tree-sitter unavailable
        .cpp/.cu/.h → {"__unknown__"} if tree-sitter unavailable
        other ext   → set() (caller should skip / use file-level)
    """
    if not changed_lines:
        return {"__unknown__"}

    suffix = file_path.suffix.lower()

    if suffix in PY_EXTENSIONS:
        parser = _get_parser('python')
        if parser:
            return _parse_with_treesitter(parser, file_path, changed_lines, 'python')
        # graceful fallback
        return _ast_fallback(file_path, changed_lines)

    elif suffix in CPP_EXTENSIONS:
        parser = _get_parser('cpp')
        if parser:
            return _parse_with_treesitter(parser, file_path, changed_lines, 'cpp')
        # no fallback for C/C++ without tree-sitter
        print(f"[WARN] tree-sitter-cpp not available; skipping symbol extraction for {file_path.name}")
        return {"__unknown__"}

    else:
        # unsupported extension — let caller decide (file-level fallback)
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Parser setup
# ─────────────────────────────────────────────────────────────────────────────

def _get_parser(lang_key: str) -> Optional[object]:
    """Return a cached tree-sitter Parser for the given language, or None."""
    if not _TS_AVAILABLE:
        return None

    if lang_key in _parser_cache:
        return _parser_cache[lang_key]

    try:
        if lang_key == 'python':
            import tree_sitter_python as tspython
            lang = Language(tspython.language())
        elif lang_key == 'cpp':
            import tree_sitter_cpp as tscpp
            lang = Language(tscpp.language())
        else:
            return None

        parser = _TSParser(lang)
        _parser_cache[lang_key] = parser
        return parser

    except Exception as e:
        print(f"[WARN] Could not load tree-sitter grammar for '{lang_key}': {e}")
        _parser_cache[lang_key] = None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter extraction
# ─────────────────────────────────────────────────────────────────────────────

def _parse_with_treesitter(parser, file_path: Path, changed_lines: set[int], lang: str) -> set[str]:
    """Parse the file with tree-sitter and map changed lines to symbol names."""
    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception as e:
        print(f"[WARN] tree-sitter failed to parse {file_path}: {e}")
        if lang == 'python':
            return _ast_fallback(file_path, changed_lines)
        return {"__unknown__"}

    definitions = _collect_definitions(tree.root_node, lang)

    symbols      = set()
    covered_lines = set()

    for start_line, end_line, name in definitions:
        for line in changed_lines:
            if start_line <= line <= end_line:
                symbols.add(name)
                covered_lines.add(line)

    uncovered = changed_lines - covered_lines
    if uncovered:
        symbols.add("__module__")

    return symbols if symbols else {"__unknown__"}


def _collect_definitions(
    node,
    lang: str,
    parent_name: Optional[str] = None,
) -> list[tuple[int, int, str]]:
    """
    Recursively walk the tree-sitter CST.

    Returns list of (start_line_1based, end_line_1based, qualified_name).

    Python examples:
        top-level function     → ("my_func",)
        method inside class    → ("MyClass.my_method",)
        nested class           → ("Outer.Inner",)

    C/C++ examples:
        free function          → ("compute_output",)
        class method           → ("Scheduler::fuse",)   (qualified_identifier)
    """
    results = []
    node_type = node.type

    if lang == 'python':
        if node_type in ('function_definition', 'async_function_definition',
                         'decorated_definition'):
            # decorated_definition wraps the real def — recurse into it
            if node_type == 'decorated_definition':
                for child in node.children:
                    results.extend(_collect_definitions(child, lang, parent_name))
                return results

            name = _py_name(node)
            if name:
                full_name  = f"{parent_name}.{name}" if parent_name else name
                start, end = _node_lines(node)
                results.append((start, end, full_name))
                # methods inside this function (nested defs) — use full_name as parent
                for child in node.children:
                    if child.type == 'block':
                        results.extend(_collect_definitions(child, lang, full_name))
            return results

        elif node_type == 'class_definition':
            name = _py_name(node)
            if name:
                full_name  = f"{parent_name}.{name}" if parent_name else name
                start, end = _node_lines(node)
                results.append((start, end, full_name))
                # recurse into class body — methods use full_name as parent
                for child in node.children:
                    if child.type == 'block':
                        results.extend(_collect_definitions(child, lang, full_name))
            return results

    elif lang == 'cpp':
        if node_type == 'function_definition':
            name = _cpp_name(node)
            if name:
                start, end = _node_lines(node)
                results.append((start, end, name))
                # don't recurse into the body for further nesting (lambdas etc.)
            return results

    # Generic child walk for all other node types
    for child in node.children:
        results.extend(_collect_definitions(child, lang, parent_name))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Name extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _py_name(node) -> Optional[str]:
    """Extract the identifier name from a Python function/class definition node."""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8', errors='ignore')
    return None


def _cpp_name(node) -> Optional[str]:
    """
    Extract the function name from a C/C++ function_definition node.

    Handles:
        void foo(int x)                   → "foo"
        int* MyClass::bar(int x)          → "MyClass::bar"
        template<> void Baz::qux()        → "Baz::qux"
        auto fn() -> int                  → "fn"
    """
    for child in node.children:
        if child.type == 'function_declarator':
            return _cpp_declarator_name(child)
        # pointer/reference return type wraps a declarator
        if child.type in ('pointer_declarator', 'reference_declarator',
                          'abstract_pointer_declarator'):
            name = _cpp_name_from_declarator_chain(child)
            if name:
                return name
    return None


def _cpp_declarator_name(node) -> Optional[str]:
    """Dig into a function_declarator to find the identifier."""
    for child in node.children:
        if child.type in ('identifier', 'qualified_identifier',
                          'destructor_name', 'operator_name'):
            return child.text.decode('utf-8', errors='ignore')
        if child.type in ('pointer_declarator', 'reference_declarator',
                          'function_declarator'):
            result = _cpp_declarator_name(child)
            if result:
                return result
    return None


def _cpp_name_from_declarator_chain(node) -> Optional[str]:
    """Walk pointer/reference declarator chains to find the function_declarator."""
    for child in node.children:
        if child.type == 'function_declarator':
            return _cpp_declarator_name(child)
        if child.type in ('pointer_declarator', 'reference_declarator'):
            result = _cpp_name_from_declarator_chain(child)
            if result:
                return result
    return None


def _node_lines(node) -> tuple[int, int]:
    """Return (start_line, end_line) as 1-based integers."""
    return node.start_point[0] + 1, node.end_point[0] + 1


# ─────────────────────────────────────────────────────────────────────────────
# Graph-builder helpers (whole-file, no line filtering)
# ─────────────────────────────────────────────────────────────────────────────

def _ast_all_symbols(file_path: Path) -> set[str]:
    """ast fallback: top-level function/class/assignment names from a .py file."""
    symbols = set()
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return symbols
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
    return symbols


def _ast_all_identifiers(file_path: Path) -> set[str]:
    """ast: all Name + Attribute identifiers referenced in a .py file."""
    identifiers = set()
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return identifiers
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id and not node.id.startswith('__'):
                identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr and not node.attr.startswith('__'):
                identifiers.add(node.attr)
    return identifiers


def _ts_cpp_identifiers(parser, file_path: Path) -> set[str]:
    """tree-sitter: collect all identifier leaf nodes from a C/C++ file."""
    identifiers = set()
    try:
        tree = parser.parse(file_path.read_bytes())
    except Exception:
        return identifiers

    def _walk(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8', errors='ignore')
            if name and not name.startswith('__'):
                identifiers.add(name)
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return identifiers


# ─────────────────────────────────────────────────────────────────────────────
# ast fallback for .py (when tree-sitter not installed)
# ─────────────────────────────────────────────────────────────────────────────

def _ast_fallback(file_path: Path, changed_lines: set[int]) -> set[str]:
    """
    Pure-stdlib fallback using ast.
    Preserves Class.method qualified naming to match graph keys.
    """
    symbols = set()
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return {"__unknown__"}

    definitions = []

    def _walk(node, parent_class: Optional[str] = None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                end = getattr(child, 'end_lineno', child.lineno + 200)
                definitions.append((child.lineno, end, child.name, parent_class))
                _walk(child, parent_class=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end  = getattr(child, 'end_lineno', child.lineno + 1000)
                name = f"{parent_class}.{child.name}" if parent_class else child.name
                definitions.append((child.lineno, end, name, parent_class))
                _walk(child, parent_class=parent_class)
            else:
                _walk(child, parent_class=parent_class)

    _walk(tree)

    covered = set()
    for start, end, name, _ in definitions:
        for line in changed_lines:
            if start <= line <= end:
                symbols.add(name)
                covered.add(line)

    if changed_lines - covered:
        symbols.add("__module__")

    return symbols or {"__unknown__"}