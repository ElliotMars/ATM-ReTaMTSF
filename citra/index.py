"""Spatial summaries over an already legal, immutable historical snapshot."""

from dataclasses import dataclass
import heapq
import math
from time import perf_counter
import numpy as np

from .kernel import exact_aggregate, immutable, vector


@dataclass
class QueryStats:
    nodes_visited: int = 0
    documents_scanned: int = 0
    exact_kernel_evaluations: int = 0
    summary_kernel_evaluations: int = 0
    metadata_records_checked: int = 0
    index_document_reads: int = 0
    build_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    fallback_exact: bool = False
    cache_hit: bool = False


@dataclass(frozen=True)
class QueryResult:
    aggregate: np.ndarray
    aggregate_bound: float
    certified_bound: float
    stats: QueryStats


@dataclass(frozen=True)
class Node:
    node_id: int
    count: int
    center: np.ndarray
    radius: float
    influence_sum: np.ndarray
    first_moment: np.ndarray
    second_residual: float
    document_refs: tuple
    version_metadata: tuple
    children: tuple


class CITRAIndex:
    def __init__(self, documents, embedding_dim, impact_dim, kernel, leaf_size=8):
        start = perf_counter()
        if leaf_size <= 0 or not isinstance(leaf_size, int):
            raise ValueError("leaf_size must be a positive integer")
        self.documents = tuple(documents)
        self.embedding_dim, self.impact_dim = embedding_dim, impact_dim
        self.kernel, self.leaf_size = kernel, leaf_size
        self.nodes = []
        self.build_document_reads = len(self.documents)  # dimension validation pass
        for d in self.documents:
            vector(d.embedding, embedding_dim)
            vector(d.impact_vector, impact_dim)
        self.root = self._build(tuple(range(len(self.documents)))) if self.documents else None
        self.build_seconds = perf_counter() - start

    def _build(self, refs):
        node_id = len(self.nodes)
        self.nodes.append(None)
        self.build_document_reads += len(refs)
        z = np.stack([self.documents[i].embedding for i in refs])
        v = np.stack([self.documents[i].impact_vector for i in refs])
        with np.errstate(over="raise", invalid="raise"):
            center = z.mean(axis=0)
            delta = z - center
            radius = float(np.linalg.norm(delta, axis=1).max())
            influence = v.sum(axis=0)
            moment = delta.T @ v  # [p,m], not [m,p]
            residual = float(np.sum(np.sum(delta * delta, axis=1) * np.linalg.norm(v, axis=1)))
        if not math.isfinite(residual) or not math.isfinite(radius):
            raise ValueError("nonfinite summaries; rescale inputs")
        children = ()
        if len(refs) > self.leaf_size:
            axis = int(np.argmax(np.ptp(z, axis=0)))
            order = np.argsort(z[:, axis], kind="stable")
            ordered = tuple(refs[int(i)] for i in order)
            middle = len(refs) // 2
            children = (self._build(ordered[:middle]), self._build(ordered[middle:]))
        node = Node(node_id, len(refs), immutable(center), radius,
                    immutable(influence), immutable(moment), residual, refs,
                    tuple((self.documents[i].doc_id, self.documents[i].version_id,
                           self.documents[i].available_at, self.documents[i].valid_from,
                           self.documents[i].valid_to, self.documents[i].ingested_at)
                          for i in refs), children)
        self.nodes[node_id] = node
        return node

    @property
    def summary_bytes(self):
        """Numeric summaries only; excludes Python object/reference overhead."""
        return sum(n.center.nbytes + n.influence_sum.nbytes + n.first_moment.nbytes
                   + 24 for n in self.nodes)

    def approximate_node(self, node, query):
        q = vector(query, self.embedding_dim)
        value = self.kernel.value(q, node.center)
        gradient = self.kernel.gradient_z(q, node.center)
        aggregate = immutable(value * node.influence_sum + node.first_moment.T @ gradient)
        bound = 0.5 * self.kernel.hessian_bound * node.second_residual
        if not math.isfinite(bound):
            raise ValueError("nonfinite error bound; rescale inputs")
        return aggregate, bound

    def aggregate_exact(self, query):
        vector(query, self.embedding_dim)
        return exact_aggregate(self.documents, query, self.kernel, self.impact_dim)

    def query(self, query, epsilon, lipschitz=1.0):
        start = perf_counter()
        q = vector(query, self.embedding_dim)
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        if not math.isfinite(lipschitz) or lipschitz < 0:
            raise ValueError("lipschitz must be finite and nonnegative")
        stats = QueryStats()
        if self.root is None:
            stats.elapsed_seconds = perf_counter() - start
            return QueryResult(immutable(np.zeros(self.impact_dim)), 0.0, 0.0, stats)
        if epsilon == 0:
            aggregate = self.aggregate_exact(q)
            stats.documents_scanned = len(self.documents)
            stats.exact_kernel_evaluations = len(self.documents)
            stats.fallback_exact = True
            stats.elapsed_seconds = perf_counter() - start
            return QueryResult(aggregate, 0.0, 0.0, stats)

        frontier = {}
        queue = []

        def visit(node):
            stats.nodes_visited += 1
            if not node.children:
                docs = tuple(self.documents[i] for i in node.document_refs)
                aggregate = exact_aggregate(docs, q, self.kernel, self.impact_dim)
                bound = 0.0
                stats.documents_scanned += node.count
                stats.exact_kernel_evaluations += node.count
            else:
                aggregate, bound = self.approximate_node(node, q)
                stats.summary_kernel_evaluations += 2  # value + gradient's value
                heapq.heappush(queue, (-bound, node.node_id, node))
            frontier[node.node_id] = (aggregate, bound)

        visit(self.root)
        # Re-sum nonnegative bounds rather than subtracting from an old total.
        # This prevents cancellation in the error budget during refinement.
        while lipschitz * math.fsum(b for _, b in frontier.values()) > epsilon:
            if not queue:
                raise ArithmeticError("exact frontier must have zero analytical remainder")
            _, _, node = heapq.heappop(queue)
            del frontier[node.node_id]
            for child in node.children:
                visit(child)
        bound = math.fsum(b for _, b in frontier.values())
        aggregate = immutable([math.fsum(float(a[j]) for a, _ in frontier.values())
                               for j in range(self.impact_dim)])
        stats.fallback_exact = stats.documents_scanned == len(self.documents)
        stats.elapsed_seconds = perf_counter() - start
        return QueryResult(aggregate, bound, lipschitz * bound, stats)
