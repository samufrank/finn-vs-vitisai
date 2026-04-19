"""Fix 4: Detach the classifier tail so it lands in the CPU partition.

Approach: connected-component analysis over FINN-domain nodes, treating
direct producer→consumer edges between FINN nodes as component edges
(non-FINN nodes act as component boundaries). The LARGEST FINN component
is the main accelerator partition; smaller isolated components are in the
CPU tail (or CPU head) and should be peeled (domain cleared) so the
partitioner treats them as CPU ops.

Rationale: neither "has no FINN predecessor" nor "has no FINN consumer"
alone correctly identifies tail-isolated FINN nodes:
- "no FINN predecessor" peels the head FINN node (Thresholding_0) which
  has no predecessors but is the first node of the main partition.
- "no FINN consumer" peels the last FINN node of the main partition
  (ElementwiseAdd_3) because its consumer is the non-FINN Reshape.

Connected components capture the correct notion: group FINN nodes by
direct-FINN-connection, then peel everything except the largest group.
"""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


class DetachClassifierTail(Transformation):
    """Peel FINN-domain nodes that are not in the main accelerator
    partition (determined by connected-component analysis)."""

    def apply(self, model: ModelWrapper):
        graph = model.graph
        inits = {i.name for i in graph.initializer}

        producer = {o: n for n in graph.node for o in n.output}
        consumers = {}
        for n in graph.node:
            for t in n.input:
                consumers.setdefault(t, []).append(n)

        # FINN nodes only
        finn_nodes = [n for n in graph.node
                      if n.domain.startswith("finn.custom_op.fpgadataflow")]

        if not finn_nodes:
            return model, False

        # Build union-find over FINN nodes; union whenever a FINN node's
        # output is directly consumed by another FINN node (edge exists
        # between FINN nodes with no non-FINN node in between).
        parent = {id(n): id(n) for n in finn_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for node in finn_nodes:
            for out in node.output:
                for c in consumers.get(out, []):
                    if c.domain.startswith("finn.custom_op.fpgadataflow"):
                        union(id(node), id(c))

        # Group by root
        groups = {}
        for n in finn_nodes:
            r = find(id(n))
            groups.setdefault(r, []).append(n)

        if len(groups) <= 1:
            # All FINN nodes in one component — no tail to peel
            return model, False

        # Largest group = main accelerator partition; keep it FINN
        main_root = max(groups, key=lambda k: len(groups[k]))

        # Peel every other group. Clear BOTH domain and backend attribute —
        # CreateDataflowPartition's partitioning function looks at the
        # ``backend`` attribute (not domain) to decide partition_id, so we
        # must strip it or the peeled nodes stay in the FINN partition and
        # create a "CPU → FINN → CPU" cycle that the partitioner rejects
        # with "cycle-free graph violated: partition depends on itself".
        peeled_count = 0
        for root, nodes in groups.items():
            if root == main_root:
                continue
            for n in nodes:
                n.domain = ""
                # Remove backend attribute so partitioner assigns -1 (CPU)
                backend_attrs = [a for a in n.attribute if a.name == "backend"]
                for a in backend_attrs:
                    n.attribute.remove(a)
                peeled_count += 1

        return model, False
