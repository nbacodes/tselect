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


def get_dunder_all(file_path: Path) -> Optional[set[str]]:
    """
    Feature 6: Parse __all__ from a .py file.

    Returns the explicit export list if defined, or None if not present.
    Used by graph_builder to resolve star imports precisely:
        "from torch.optim import *"  →  exactly {"SGD", "Adam", "RMSprop"}
    instead of dumping every symbol in the file.

    Examples:
        __all__ = ["SGD", "Adam"]          → {"SGD", "Adam"}
        __all__ = ["SGD"] + ["Adam"]       → None (dynamic, can't resolve)
        (no __all__)                       → None (caller uses get_all_symbols)
    """
    if file_path.suffix.lower() not in PY_EXTENSIONS:
        return None

    parser = _get_parser('python')
    if not parser:
        return _ast_dunder_all(file_path)

    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception:
        return _ast_dunder_all(file_path)

    # walk top-level statements only — __all__ is always module-level
    for node in tree.root_node.children:
        if node.type != 'expression_statement':
            continue
        # look for assignment: __all__ = [...]
        for child in node.children:
            if child.type != 'assignment':
                continue
            targets = [c for c in child.children if c.type == 'identifier']
            if not targets or targets[0].text.decode() != '__all__':
                continue
            # find the list node
            for val in child.children:
                if val.type == 'list':
                    names = set()
                    for item in val.children:
                        if item.type == 'string':
                            # strip quotes
                            raw = item.text.decode('utf-8', errors='ignore')
                            name = raw.strip('"\'')
                            if name:
                                names.add(name)
                    return names if names else None
    return None


def get_decorator_registry(file_path: Path) -> dict[str, str]:
    """
    Feature 8: Extract decorator → function mappings from a .py file.

    Finds patterns like:
        @register_lowering(aten.add)
        def add_lowering(x, y): ...

        @register_op("mul")
        def mul_op(...): ...

    Returns:
        {
            "aten.add":  "add_lowering",
            "mul":       "mul_op",
        }

    Used by graph_builder to resolve decorator-wrapped registries.
    When a test calls aten.add, we now know it calls add_lowering.
    """
    if file_path.suffix.lower() not in PY_EXTENSIONS:
        return {}

    parser = _get_parser('python')
    if not parser:
        return _ast_decorator_registry(file_path)

    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception:
        return _ast_decorator_registry(file_path)

    registry = {}

    def _walk(node):
        if node.type == 'decorated_definition':
            fn_name   = None
            dec_args  = []

            for child in node.children:
                # get the function name
                if child.type in ('function_definition', 'async_function_definition'):
                    fn_name = _py_name(child)

                # get decorator arguments
                elif child.type == 'decorator':
                    for dec_child in child.children:
                        if dec_child.type == 'call':
                            # extract all arguments as strings
                            for arg in dec_child.children:
                                if arg.type == 'argument_list':
                                    for a in arg.children:
                                        text = a.text.decode('utf-8', errors='ignore').strip()
                                        if text and text not in (',', '(', ')'):
                                            dec_args.append(text)

            if fn_name and dec_args:
                for arg in dec_args:
                    registry[arg] = fn_name

        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return registry


def classify_change(file_path: Path, changed_lines: set[int]) -> dict[str, str]:
    """
    Feature 13: Classify what KIND of change happened per function.

    Instead of just knowing WHICH function changed, know HOW it changed:
        "signature"    — parameter list or return annotation modified
                         → callers may be broken, consider coverage rebuild
        "body"         — only internal logic changed, same interface
                         → callers unaffected, coverage map still valid
        "new_function" — function didn't exist before (all lines are new)
                         → not in coverage map, BFS fallback
        "deleted"      — function was removed (no lines in new file)
                         → find all callers, definitely run their tests

    Returns:
        {
            "SGD.step":        "body",
            "SGD.__init__":    "signature",
            "new_helper":      "new_function",
        }
    """
    if file_path.suffix.lower() not in PY_EXTENSIONS:
        return {sym: "body" for sym in extract_symbols_at_lines(file_path, changed_lines)}

    parser = _get_parser('python')
    if not parser:
        return {sym: "body" for sym in extract_symbols_at_lines(file_path, changed_lines)}

    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception:
        return {}

    definitions = _collect_definitions(tree.root_node, 'python')
    result = {}

    for start, end, name in definitions:
        lines_in_fn = {l for l in changed_lines if start <= l <= end}
        if not lines_in_fn:
            continue

        # find the function node to get signature line range
        sig_lines = _get_signature_lines(tree.root_node, name)

        if sig_lines and (lines_in_fn & sig_lines):
            result[name] = "signature"
        else:
            result[name] = "body"

    # new_function: changed lines not covered by any definition
    covered = {l for start, end, _ in definitions for l in changed_lines if start <= l <= end}
    if changed_lines - covered:
        result["__module__"] = "new_function"

    return result


def get_call_sites(file_path: Path) -> dict[str, set[str]]:
    """
    Feature 14: Extract what each function/method actually CALLS.

    Returns:
        {
            "TestOptimCPU.test_sgd_momentum": {"SGD", "step", "zero_grad"},
            "TestOptimCPU.test_adam":         {"Adam", "step"},
        }

    This is stronger than import analysis — a test file may import SGD
    but only test_sgd_momentum actually CALLS it. test_adam never does.

    Used by graph_builder._extract_method_level_references() to get
    precise per-method symbol usage instead of file-level import tracking.
    """
    if file_path.suffix.lower() not in PY_EXTENSIONS:
        return {}

    parser = _get_parser('python')
    if not parser:
        return _ast_call_sites(file_path)

    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception:
        return _ast_call_sites(file_path)

    return _ts_call_sites(tree.root_node)


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
# Feature 6: __all__ helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ast_dunder_all(file_path: Path) -> Optional[set[str]]:
    """ast fallback for __all__ parsing."""
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
                    return names if names else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7: TYPE_CHECKING guard detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_type_checking_block(node) -> bool:
    """
    Return True if a tree-sitter if_statement node is:
        if TYPE_CHECKING:
        if typing.TYPE_CHECKING:

    Used to skip imports inside these blocks — they're type-hint-only
    and never execute at runtime, so they create false positive graph edges.
    """
    for child in node.children:
        if child.type in ('identifier', 'attribute'):
            text = child.text.decode('utf-8', errors='ignore')
            if 'TYPE_CHECKING' in text:
                return True
    return False


def is_runtime_import(import_node, parent_chain: list) -> bool:
    """
    Feature 7 public helper: return False if the import is inside
    a TYPE_CHECKING block (meaning it's type-hint-only, not runtime).

    parent_chain: list of ancestor node types from tree-sitter walk.

    Usage in graph_builder: skip imports where this returns False.
    """
    for ancestor_type in parent_chain:
        if ancestor_type == 'if_statement':
            return False  # conservative — caller checks _is_type_checking_block
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8: decorator registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ast_decorator_registry(file_path: Path) -> dict[str, str]:
    """ast fallback for decorator registry extraction."""
    registry = {}
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return registry

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.decorator_list:
            continue

        fn_name = node.name
        for dec in node.decorator_list:
            # @register_lowering(aten.add)  → Call node
            if isinstance(dec, ast.Call):
                for arg in dec.args:
                    # aten.add → Attribute node
                    if isinstance(arg, ast.Attribute):
                        key = f"{_ast_dotted(arg.value)}.{arg.attr}"
                        registry[key] = fn_name
                    elif isinstance(arg, ast.Name):
                        registry[arg.id] = fn_name
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        registry[arg.value] = fn_name
    return registry


def _ast_dotted(node) -> str:
    """Reconstruct dotted name from ast.Attribute chain."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_dotted(node.value)}.{node.attr}"
    return "?"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 13: signature line detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_signature_lines(root_node, qualified_name: str) -> Optional[set[int]]:
    """
    Find the 1-based line numbers of just the signature of a function
    (from 'def' to the closing ')' of the parameter list, inclusive).

    Returns None if the function isn't found.
    """
    parts = qualified_name.split('.')

    def _find(node, depth: int, parent: str) -> Optional[set[int]]:
        target = parts[depth]

        for child in node.children:
            if child.type == 'class_definition':
                name = _py_name(child)
                if name == target and depth < len(parts) - 1:
                    body = next((c for c in child.children if c.type == 'block'), None)
                    if body:
                        result = _find(body, depth + 1, name)
                        if result:
                            return result

            elif child.type in ('function_definition', 'async_function_definition'):
                name = _py_name(child)
                if name == target and depth == len(parts) - 1:
                    # signature = from start of node to end of parameters
                    params = next(
                        (c for c in child.children if c.type == 'parameters'), None
                    )
                    if params:
                        start = child.start_point[0] + 1
                        end   = params.end_point[0] + 1
                        return set(range(start, end + 1))

            elif child.type == 'decorated_definition':
                result = _find(child, depth, parent)
                if result:
                    return result

            elif child.type == 'block':
                result = _find(child, depth, parent)
                if result:
                    return result

        return None

    return _find(root_node, 0, '')


# ─────────────────────────────────────────────────────────────────────────────
# Feature 14: call site extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_call_sites(root_node) -> dict[str, set[str]]:
    """
    tree-sitter: extract what each class method CALLS.

    Returns:
        {
            "TestOptimCPU.test_sgd_momentum": {
                "SGD",
                "step",
                "torch.ops.aten.max_pool2d_with_indices_backward",
                "ops.aten.max_pool2d_with_indices_backward",
                "aten.max_pool2d_with_indices_backward",
            },
        }

    Collects:
        - Direct calls:   SGD(...)              → "SGD"
        - Attribute:      opt.step()            → "step"
        - Full chains:    torch.ops.aten.add()  → all suffixes of the chain
                          so decorator registry can match "aten.add"
    """
    result = {}

    def _collect_calls(node) -> set[str]:
        calls = set()
        if node.type == 'call':
            fn = node.children[0] if node.children else None
            if fn:
                if fn.type == 'identifier':
                    calls.add(fn.text.decode('utf-8', errors='ignore'))
                elif fn.type == 'attribute':
                    # collect individual identifiers
                    for c in fn.children:
                        if c.type == 'identifier':
                            calls.add(c.text.decode('utf-8', errors='ignore'))
                    # also collect full dotted chain + ALL suffixes
                    # torch.ops.aten.add → also "ops.aten.add", "aten.add"
                    # so decorator_registry["aten.add"] can match
                    try:
                        full  = fn.text.decode('utf-8', errors='ignore')
                        parts = full.split('.')
                        for i in range(len(parts) - 1):  # min 2 parts
                            suffix = '.'.join(parts[i:])
                            if suffix:
                                calls.add(suffix)
                    except Exception:
                        pass
        for child in node.children:
            calls |= _collect_calls(child)
        return calls

    def _walk_class(class_node, class_name: str):
        for child in class_node.children:
            if child.type == 'block':
                for item in child.children:
                    fn_node = item
                    # handle decorated methods
                    if item.type == 'decorated_definition':
                        for c in item.children:
                            if c.type in ('function_definition', 'async_function_definition'):
                                fn_node = c
                                break

                    if fn_node.type in ('function_definition', 'async_function_definition'):
                        fn_name = _py_name(fn_node)
                        if fn_name:
                            body = next(
                                (c for c in fn_node.children if c.type == 'block'), None
                            )
                            if body:
                                key = f"{class_name}.{fn_name}"
                                result[key] = _collect_calls(body)

    for node in root_node.children:
        if node.type == 'class_definition':
            class_name = _py_name(node)
            if class_name:
                _walk_class(node, class_name)
        elif node.type == 'decorated_definition':
            for child in node.children:
                if child.type == 'class_definition':
                    class_name = _py_name(child)
                    if class_name:
                        _walk_class(child, class_name)

    return result


def _ast_call_sites(file_path: Path) -> dict[str, set[str]]:
    """ast fallback for call site extraction — also extracts full dotted chains."""
    result = {}
    try:
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        tree   = ast.parse(source)
    except Exception:
        return result

    def _dotted(node) -> Optional[str]:
        """Reconstruct full dotted name from ast.Attribute chain."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = _dotted(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None

    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for method in class_node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = set()
            for node in ast.walk(method):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
                        if isinstance(node.func.value, ast.Name):
                            calls.add(node.func.value.id)
                        # full dotted chain + all suffixes
                        full = _dotted(node.func)
                        if full:
                            parts = full.split('.')
                            for i in range(len(parts) - 1):  # min 2 parts
                                calls.add('.'.join(parts[i:]))
            key = f"{class_node.name}.{method.name}"
            result[key] = calls

    return result


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