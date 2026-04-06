"""
graph_selector.py
-----------------
Selects tests for changed files using the dependency graph.

Selection strategy (in priority order):

  0. Pre-flight checks:
       - Test file changed (.py) → self-select directly, skip graph
       - Non-code file changed → skip entirely
       - Non-.py file inside test/ → skip (pallas_expected_failures etc.)

  Tier 1 — Function-level (precise):
    Uses diff_parser to find which functions/classes changed.
    Looks them up in function_reverse_graph.
    Result: only test METHODS that reference those specific symbols.

    Method key fix (1a):
      diff_parser returns "SGD.step" (method inside class)
      graph stores "SGD" (class level)
      fix: if "SGD.step" not found → strip method → try "SGD"

  Tier 1b — Re-export routing:
    For transitive files that are DIRECT importers of the changed file,
    use the importer's function_reverse_graph entries with the changed
    symbols from the original file.

  Tier 2 — File-level fallback:
    Used when function-level lookup returns empty.

  Transitive expansion (schema 3.0):
    BFS through source_reverse_graph.
    TWO stopping conditions (both must pass to follow an importer):
      1. Fanout guard: len(test_dependents) <= fanout_threshold
         threshold from graph (dynamic, computed from distribution gap)
      2. Identifier overlap: importer uses at least one changed symbol
         if changed_symbols ∩ file_identifiers[importer] == empty → stop
    No depth counter — BFS runs until both guards stop everything.
"""

from pathlib import Path
from collections import defaultdict


DEFAULT_HIGH_FANOUT_THRESHOLD = 30

NON_CODE_EXTENSIONS = {
    '.yml', '.yaml', '.rst', '.md', '.txt',
    '.json', '.cfg', '.ini', '.toml', '.csv',
    '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.sh', '.bash', '.bat', '.ps1',
}
NON_CODE_FILENAMES = {
    'setup.py', 'setup.cfg', 'MANIFEST.in',
    'Makefile', 'CMakeLists.txt', 'LICENSE', 'NOTICE',
}

TEST_PATH_PREFIXES = ('test/', 'tests/', 'test\\', 'tests\\')


def _is_test_file(rel: str) -> bool:
    """
    Return True if this file is a runnable test file.
    Must start with test/ AND be a .py file.
    Files without .py inside test/ are data files
    (pallas_expected_failures/*, dynamo_expected_failures/*, etc.)
    """
    if not rel.startswith(TEST_PATH_PREFIXES):
        return False
    return Path(rel).suffix == '.py'


def _is_non_code_file(rel: str) -> bool:
    """Return True if this file is non-code and needs no test selection."""
    p = Path(rel)
    # files inside test/ that are not .py are data files
    if rel.startswith(TEST_PATH_PREFIXES) and p.suffix != '.py':
        return True
    return (
        p.suffix in NON_CODE_EXTENSIONS
        or p.name in NON_CODE_FILENAMES
        or not p.suffix
    )


def _expand_transitively(
    changed_file: str,
    source_reverse_graph: dict,
    file_reverse_graph: dict,
    file_identifiers: dict,
    changed_symbols: set,
    threshold: int = DEFAULT_HIGH_FANOUT_THRESHOLD,
) -> set:
    """
    BFS through source_reverse_graph.

    TWO stopping conditions per importer — both must pass to follow:
      1. Fanout guard: importer has <= threshold test dependents
      2. Identifier overlap: importer uses at least one changed symbol
         (only applied when changed_symbols is non-empty and meaningful)

    No depth counter. BFS runs until guards stop everything.
    """
    visited  = {changed_file}
    frontier = {changed_file}

    # normalize changed symbols for overlap check
    # strip method suffix: "SGD.step" → "SGD", keep "__module__" out
    base_symbols = set()
    for s in changed_symbols:
        if s not in ("__module__", "__imports__", "__all__", "__constant__"):
            base_symbols.add(s.split(".")[0])

    use_identifier_overlap = bool(base_symbols) and bool(file_identifiers)

    while frontier:
        next_frontier = set()
        for f in frontier:
            importers = set(source_reverse_graph.get(f, []))
            for importer in importers:
                if importer in visited:
                    continue

                # Guard 1: fanout
                if len(file_reverse_graph.get(importer, [])) > threshold:
                    continue

                # Guard 2: identifier overlap
                if use_identifier_overlap:
                    importer_ids = set(file_identifiers.get(importer, []))
                    if importer_ids and not (base_symbols & importer_ids):
                        continue

                next_frontier.add(importer)
                visited.add(importer)

        frontier = next_frontier

    return visited


def select_tests_from_graph(
    changed_files: list,
    graph: dict,
    repo_root: Path,
    config: dict = None,
) -> tuple:
    """
    Main entry point. Returns (selected, total_methods).
    """
    config    = config or {}

    # use dynamic threshold from graph if available, else config, else default
    threshold = (
        graph.get("fanout_threshold")
        or config.get("graph", {}).get("fanout_threshold", DEFAULT_HIGH_FANOUT_THRESHOLD)
        or DEFAULT_HIGH_FANOUT_THRESHOLD
    )

    reverse_graph        = graph.get("full_reverse_graph", {})
    function_graph       = graph.get("function_reverse_graph", {})
    test_inventory       = graph.get("test_inventory", {})
    source_reverse_graph = graph.get("source_reverse_graph", {})
    file_identifiers     = graph.get("file_identifiers", {})
    has_function_graph   = bool(function_graph)
    has_transitive       = bool(source_reverse_graph)

    from tselect.core.diff_parser import get_changed_functions

    changed_functions = {}
    if has_function_graph:
        try:
            changed_functions = get_changed_functions(repo_root, changed_files)
        except Exception:
            changed_functions = {}

    selected_tests      = {}
    skipped_high_fanout = []
    skipped_non_code    = []

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        # skip propagation from test files
        if rel.startswith(TEST_PATH_PREFIXES):
            pass

        # Pre-flight 1: test file self-select
        if _is_test_file(rel):
            classes = test_inventory.get(rel, {})
            if classes:
                if rel not in selected_tests:
                    selected_tests[rel] = {
                        "triggered_by":    [rel],
                        "matched_symbols": [],
                        "selection_mode":  "self",
                        "classes":         dict(classes),
                    }
                print(f"    Self-select: {rel} (test file changed directly)")
            else:
                print(f"    Self-select: {rel} (test file changed, not in inventory)")
            continue

        # Pre-flight 2: non-code file skip
        if _is_non_code_file(rel):
            skipped_non_code.append(rel)
            continue

        # Get changed symbols for this file
        rel_symbols_changed = changed_functions.get(rel, set())
        symbols_changed     = rel_symbols_changed

        if has_function_graph and symbols_changed not in (set(), {"__unknown__"}):
            function_selected = _function_level_select(
                rel, symbols_changed, function_graph, test_inventory
            )
            if function_selected:
                _merge_into_selected(selected_tests, function_selected)
                print(f"    Function-level hit: {rel} → skipping transitive expansion")
                continue

            # function lookup empty (e.g. decorator-registered aten ops)
            # fall back to file-level — much better than BFS explosion
            print(f"    [INFO] No function-level match for {rel} → file-level fallback")
            file_level_tests = set(reverse_graph.get(rel, []))
            if file_level_tests:
                _add_file_level_tests(
                    selected_tests, file_level_tests, test_inventory,
                    trigger=rel, mode="file"
                )
                continue

        elif symbols_changed in (set(), {"__unknown__"}):
            print(f"[WARN] No usable diff symbols for {rel} → skipping function-level")

        # Only fallback case reaches here
        if has_transitive:
            expanded = _expand_transitively(
                changed_file=rel,
                source_reverse_graph=source_reverse_graph,
                file_reverse_graph=reverse_graph,
                file_identifiers=file_identifiers,
                changed_symbols=rel_symbols_changed,
                threshold=threshold,
            )
        else:
            expanded = {rel}

        transitive_added = expanded - {rel}
        if transitive_added:
            print(f"    Transitive: {rel} → also checking {len(transitive_added)} dependent source files")

        direct_importers_of_rel = set(source_reverse_graph.get(rel, []))

        for expanded_file in expanded:

            file_level_tests = set(reverse_graph.get(expanded_file, []))
            if len(file_level_tests) > threshold:
                if expanded_file == rel:
                    skipped_high_fanout.append((rel, len(file_level_tests)))
                continue

            # DIRECTLY CHANGED FILE
            if expanded_file == rel:
                symbols_changed = rel_symbols_changed

                if has_function_graph:
                    if symbols_changed not in (set(), {"__unknown__"}):
                        function_selected = _function_level_select(
                            expanded_file, symbols_changed, function_graph, test_inventory
                        )
                        if function_selected:
                            _merge_into_selected(selected_tests, function_selected)
                            continue

                    elif symbols_changed == {"__unknown__"}:
                        print(f"[WARN] Low confidence diff for {rel} → shallow fallback")

                    else:
                        print(f"[WARN] No usable diff info for {rel} → skipping")
                        continue

                if not file_level_tests:
                    dir_mapping      = config.get("graph", {}).get("directory_mapping", [])
                    file_level_tests = _proximity_fallback(expanded_file, test_inventory, dir_mapping)
                    mode             = "proximity"
                else:
                    if expanded_file != rel:
                        continue
                    mode = "file"

                _add_file_level_tests(
                    selected_tests, file_level_tests, test_inventory,
                    trigger=rel, mode=mode
                )

            # TRANSITIVE FILE
            else:
                if has_function_graph and expanded_file in direct_importers_of_rel and rel_symbols_changed:
                    reexport_selected = _reexport_level_select(
                        importer_file    = expanded_file,
                        rel_symbols      = rel_symbols_changed,
                        original_trigger = rel,
                        function_graph   = function_graph,
                        test_inventory   = test_inventory,
                    )
                    if reexport_selected:
                        _merge_into_selected(selected_tests, reexport_selected)
                        continue

                if not file_level_tests:
                    dir_mapping      = config.get("graph", {}).get("directory_mapping", [])
                    file_level_tests = _proximity_fallback(expanded_file, test_inventory, dir_mapping)
                    mode             = "proximity"
                else:
                    mode = "file"

                _add_file_level_tests(
                    selected_tests, file_level_tests, test_inventory,
                    trigger=rel, mode=mode
                )

    if skipped_non_code:
        print()
        for f in skipped_non_code:
            print(f"    Skipping non-code file: {f}")

    if skipped_high_fanout:
        print()
        print("    High-fanout files skipped (infrastructure files):")
        for f, count in skipped_high_fanout:
            print(f"     {f}  ({count} test files depend on it)")
        print()

    total_methods = sum(
        cls_data["test_count"]
        for data in selected_tests.values()
        for cls_data in data["classes"].values()
    )

    mode_counts = defaultdict(int)
    for d in selected_tests.values():
        mode_counts[d.get("selection_mode", "unknown")] += 1

    if selected_tests:
        mode_str = "  |  ".join(
            f"{count} via {mode}"
            for mode, count in sorted(mode_counts.items())
        )
        print(f"  Selection mode: {mode_str}")

    return selected_tests, total_methods


# ─────────────────────────────────────────────────────────────────────────────
# Selection helpers
# ─────────────────────────────────────────────────────────────────────────────

# device suffixes PyTorch parametrize adds to test method names
_DEVICE_SUFFIXES = ('_cpu', '_cuda', '_mps', '_xpu', '_npu', '_hpu')


def _resolve_mixin_class(
    test_file: str,
    cls_name: str,
    method: str,
    classes: dict,
) -> dict:
    """
    Resolve mixin/base class test references to actual inventory entries.

    PyTorch uses CommonTemplate as a mixin — tests defined there get
    parametrized into CpuTests, GPUTests etc. with device suffixes:
        CommonTemplate::test_max_pool2d_with_indices_backward6
            → CpuTests::test_max_pool2d_with_indices_backward6_cpu
            → GPUTests::test_max_pool2d_with_indices_backward6_mps

    Strategy: for each class in inventory, strip device suffix from each
    method name and check if it matches our target method name.

    No hardcoding of class names or device names — purely suffix matching.

    Returns dict of {real_cls: cls_data} or empty dict if no match found.
    """
    if cls_name in classes:
        return {}  # class exists directly, no resolution needed

    resolved = {}
    for real_cls, cls_data in classes.items():
        matched_ids = []
        for nid in cls_data.get("node_ids", []):
            parts = nid.split("::")
            if len(parts) < 3:
                continue
            inv_method = parts[2]
            # strip device suffix to get base method name
            stripped = inv_method
            for suffix in _DEVICE_SUFFIXES:
                if inv_method.endswith(suffix):
                    stripped = inv_method[:-len(suffix)]
                    break
            if stripped == method:
                matched_ids.append(nid)

        if matched_ids:
            resolved[real_cls] = {
                "node_ids":   matched_ids,
                "test_count": len(matched_ids),
            }

    return resolved


def _function_level_select(
    src_file: str,
    symbols_changed: set,
    function_graph: dict,
    test_inventory: dict,
) -> dict:
    """
    Look up changed symbols in function_reverse_graph.

    Schema 3.0: function_graph values are test METHOD IDs:
        "test_optim.py::TestOptimCPU::test_sgd_momentum"

    Schema 2.x: function_graph values are test FILE paths:
        "test_optim.py"

    Handles both — detects by checking if value contains "::" (method ID)
    or not (file path).

    1a method key fix: SGD.step → try SGD if SGD.step not found.

    Mixin resolution: CommonTemplate::test_X → CpuTests::test_X_cpu etc.
    """
    if "__module__" in symbols_changed:
        all_symbols = {
            key.split("::", 1)[1]
            for key in function_graph
            if key.startswith(f"{src_file}::")
        }
        symbols_changed = all_symbols or symbols_changed

    selected  = defaultdict(lambda: {
        "triggered_by":    [],
        "matched_symbols": [],
        "selection_mode":  "function",
        "classes":         {},
    })
    found_any = False

    for sym in symbols_changed:
        key        = f"{src_file}::{sym}"
        graph_vals = function_graph.get(key, [])

        # 1a: method key fix — SGD.step → try SGD
        effective_sym = sym
        if not graph_vals and "." in sym:
            class_name = sym.split(".")[0]
            class_key  = f"{src_file}::{class_name}"
            graph_vals = function_graph.get(class_key, [])
            if graph_vals:
                effective_sym = class_name

        for val in graph_vals:
            # schema 3.0: val = "test_file.py::ClassName::test_method"
            if val.count("::") >= 2:
                parts     = val.split("::")
                test_file = parts[0]
                cls_name  = parts[1]
                method    = parts[2]
                node_id   = val

                classes  = test_inventory.get(test_file, {})

                # resolve CommonTemplate → CpuTests/GPUTests etc.
                resolved = _resolve_mixin_class(
                    test_file, cls_name, method, classes
                )

                if test_file not in selected:
                    selected[test_file]["triggered_by"]    = [src_file]
                    selected[test_file]["matched_symbols"].append(effective_sym)

                if resolved:
                    for real_cls, cls_data in resolved.items():
                        if real_cls not in selected[test_file]["classes"]:
                            selected[test_file]["classes"][real_cls] = {
                                "node_ids":   list(cls_data["node_ids"]),
                                "test_count": cls_data["test_count"],
                            }
                        else:
                            # accumulate — don't overwrite previous backward variants
                            existing = selected[test_file]["classes"][real_cls]
                            new_ids  = [
                                nid for nid in cls_data["node_ids"]
                                if nid not in existing["node_ids"]
                            ]
                            existing["node_ids"].extend(new_ids)
                            existing["test_count"] = len(existing["node_ids"])
                            
                elif cls_name in classes:
                    # class exists directly in inventory
                    selected[test_file]["classes"][cls_name] = classes[cls_name]
                else:
                    # not in inventory — use raw node_id as fallback
                    selected[test_file]["classes"][cls_name] = {
                        "node_ids":   [node_id],
                        "test_count": 1,
                    }
                found_any = True

            # schema 2.x: val = "test_file.py"
            else:
                test_file = val
                classes   = test_inventory.get(test_file, {})
                if not classes:
                    continue
                found_any = True
                selected[test_file]["triggered_by"]    = [src_file]
                selected[test_file]["matched_symbols"].append(effective_sym)
                selected[test_file]["classes"].update(classes)

    if not found_any:
        print(f"[WARN] No graph match for symbols in {src_file} → falling back to module-level")

        module_symbols = {
            key.split("::", 1)[1]
            for key in function_graph
            if key.startswith(f"{src_file}::")
        }

        if module_symbols:
            return _function_level_select(
                src_file,
                module_symbols,
                function_graph,
                test_inventory,
            )

        return {}

    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _reexport_level_select(
    importer_file: str,
    rel_symbols: set,
    original_trigger: str,
    function_graph: dict,
    test_inventory: dict,
) -> dict:
    """
    Re-export routing: use importer's function_graph with changed symbols.
    Handles both schema 3.0 (method IDs) and 2.x (file paths).
    """
    normalized_symbols = set()
    for sym in rel_symbols:
        if sym == "__module__":
            all_importer_syms = {
                key.split("::", 1)[1]
                for key in function_graph
                if key.startswith(f"{importer_file}::")
            }
            normalized_symbols.update(all_importer_syms)
        elif "." in sym:
            normalized_symbols.add(sym.split(".")[0])
        else:
            normalized_symbols.add(sym)

    if not normalized_symbols:
        return {}

    selected  = defaultdict(lambda: {
        "triggered_by":    [],
        "matched_symbols": [],
        "selection_mode":  "re-export",
        "classes":         {},
    })
    found_any = False

    for sym in normalized_symbols:
        key        = f"{importer_file}::{sym}"
        graph_vals = function_graph.get(key, [])

        for val in graph_vals:
            if val.count("::") >= 2:
                parts     = val.split("::")
                test_file = parts[0]
                cls_name  = parts[1]
                method    = parts[2]
                node_id   = val
                classes   = test_inventory.get(test_file, {})

                # resolve mixin classes here too
                resolved = _resolve_mixin_class(
                    test_file, cls_name, method, classes
                )

                selected[test_file]["triggered_by"]    = [original_trigger]
                selected[test_file]["matched_symbols"].append(sym)

                if resolved:
                    for real_cls, cls_data in resolved.items():
                        selected[test_file]["classes"][real_cls] = cls_data
                elif cls_name in (classes or {}):
                    selected[test_file]["classes"][cls_name] = classes[cls_name]
                else:
                    selected[test_file]["classes"][cls_name] = {
                        "node_ids":   [node_id],
                        "test_count": 1,
                    }
                found_any = True

            else:
                test_file = val
                classes   = test_inventory.get(test_file, {})
                if not classes:
                    continue
                found_any = True
                selected[test_file]["triggered_by"]    = [original_trigger]
                selected[test_file]["matched_symbols"].append(sym)
                selected[test_file]["classes"].update(classes)

    if not found_any:
        return {}

    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _add_file_level_tests(
    selected_tests: dict,
    file_level_tests: set,
    test_inventory: dict,
    trigger: str,
    mode: str,
) -> None:
    for test_file in file_level_tests:
        classes = test_inventory.get(test_file, {})
        if not classes:
            continue
        if test_file not in selected_tests:
            selected_tests[test_file] = {
                "triggered_by":    [trigger],
                "matched_symbols": [],
                "selection_mode":  mode,
                "classes":         dict(classes),
            }
        else:
            if trigger not in selected_tests[test_file]["triggered_by"]:
                selected_tests[test_file]["triggered_by"].append(trigger)


def _merge_into_selected(selected_tests: dict, new_selections: dict) -> None:
    for test_file, data in new_selections.items():
        if test_file not in selected_tests:
            selected_tests[test_file] = data
        else:
            selected_tests[test_file]["triggered_by"] = sorted(set(
                selected_tests[test_file]["triggered_by"] + data["triggered_by"]
            ))
            selected_tests[test_file]["matched_symbols"] = sorted(set(
                selected_tests[test_file]["matched_symbols"] + data["matched_symbols"]
            ))
            # ← NEW: don't downgrade existing class data
            # keep whichever has more tests (self-select > function-level)
            for cls_name, cls_data in data["classes"].items():
                existing = selected_tests[test_file]["classes"].get(cls_name)
                if existing is None:
                    selected_tests[test_file]["classes"][cls_name] = cls_data
                elif cls_data["test_count"] > existing["test_count"]:
                    selected_tests[test_file]["classes"][cls_name] = cls_data

def _proximity_fallback(rel: str, test_inventory: dict, dir_mapping: list) -> set:
    if dir_mapping:
        for mapping in dir_mapping:
            src_prefix  = mapping.get("source", "")
            test_prefix = mapping.get("tests", "")
            if rel.startswith(src_prefix):
                return {tf for tf in test_inventory if tf.startswith(test_prefix)}

    changed_parts = set(Path(rel).parts)
    candidates    = set()
    for test_file in test_inventory:
        test_parts = set(Path(test_file).parts)
        if len(changed_parts & test_parts) >= 2:
            candidates.add(test_file)
    return candidates


def _normalize(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).relative_to(repo_root))
    except ValueError:
        return str(path_str).lstrip("./")


def get_pytest_node_ids(selected: dict) -> list:
    node_ids = []
    for data in selected.values():
        for cls_data in data["classes"].values():
            node_ids.extend(cls_data.get("node_ids", []))
    return sorted(set(node_ids))


def get_summary_info(selected: dict) -> tuple:
    selected_classes = []
    class_test_count = {}
    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
            label = f"{test_file}::{cls_name}"
            selected_classes.append(label)
            class_test_count[cls_name] = cls_data["test_count"]
    return selected_classes, class_test_count