class GraphSelector:
    def __init__(self, graph_data):
        self.reverse_graph = graph_data.get("full_reverse_graph", {})

    def select_tests(self, changed_files):
        affected = set()

        for file in changed_files:
            tests = self.reverse_graph.get(file, [])
            affected.update(tests)

        return sorted(affected)
