from typing import List, Set, Dict


def map_files_to_components(changed_files: List[str], ownership: Dict) -> Set[str]:
    affected = set()

    for component, paths in ownership.items():
        for changed in changed_files:
            for p in paths:
                if changed.startswith(p):
                    affected.add(component)

    return affected

def collect_tests_from_components(components: Set[str], test_json: Dict):
    """
    Returns:
        selected_classes: Set[str]
        class_test_count: Dict[str, int]
    """

    selected_classes = set()
    class_test_count = {}

    test_root = test_json.get("test_root", "")
    all_components = test_json.get("components", {})

    for comp in components:
        files = all_components.get(comp, {})

        for test_file, classes in files.items():
            for cls, data in classes.items():

                class_id = f"{test_root}/{test_file}::{cls}"
                selected_classes.add(class_id)

                # count number of tests inside class
                count = len(data.get("tests", {}))
                class_test_count[class_id] = count

    return selected_classes, class_test_count
