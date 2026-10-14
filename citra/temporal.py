"""Append-only versions with explicit point-in-time accessibility semantics."""

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

from .kernel import RBFKernel, exact_aggregate, immutable, vector


def utc(value):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetime objects")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, eq=False)
class Document:
    doc_id: str
    version_id: str
    source: str
    entity_id: str
    issued_at: Optional[datetime]
    available_at: Optional[datetime]
    valid_from: datetime
    valid_to: Optional[datetime]
    event_start: Optional[datetime]
    event_end: Optional[datetime]
    embedding: object
    impact_vector: object
    ingested_at: Optional[datetime] = None
    availability_provenance: str = "unknown"

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in
                   (self.doc_id, self.version_id, self.source, self.entity_id)):
            raise ValueError("document identity fields must be nonempty strings")
        for name in ("issued_at", "available_at", "valid_from", "valid_to",
                     "event_start", "event_end", "ingested_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))
        if self.valid_from is None:
            raise ValueError("valid_from is required")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("validity interval must be nonempty")
        if (self.event_start is not None and self.event_end is not None
                and self.event_end < self.event_start):
            raise ValueError("reversed event interval")
        if (self.issued_at is not None and self.available_at is not None
                and self.available_at < self.issued_at):
            raise ValueError("availability cannot precede issuance")
        if self.availability_provenance not in ("unknown", "verified", "simulated"):
            raise ValueError("invalid availability provenance")
        object.__setattr__(self, "embedding", immutable(vector(self.embedding)))
        object.__setattr__(self, "impact_vector", immutable(vector(self.impact_vector)))

    def accessible(self, cutoff, allow_simulated=False):
        provenance_ok = (self.availability_provenance == "verified"
                         or (allow_simulated and self.availability_provenance == "simulated"))
        return (provenance_ok and self.available_at is not None
                and self.available_at <= cutoff
                and (self.issued_at is None or self.issued_at <= cutoff)
                and (self.ingested_at is None or self.ingested_at <= cutoff))


class TemporalDocumentStore:
    """One authoritative version per doc_id; revisions never mutate old records.

    Unknown availability is excluded. Synthetic availability needs opt-in.
    Ingestion, when known, additionally limits system accessibility.
    The newest accessible version whose valid_from has begun supersedes older
    versions. An expired newest version does not resurrect an older version.
    """

    def __init__(self, embedding_dim, impact_dim, sigma=1.0, leaf_size=8,
                 allow_simulated=False):
        if (not isinstance(embedding_dim, int) or embedding_dim <= 0
                or not isinstance(impact_dim, int) or impact_dim <= 0):
            raise ValueError("dimensions must be positive integers")
        if not isinstance(leaf_size, int) or leaf_size <= 0:
            raise ValueError("leaf_size must be a positive integer")
        self.embedding_dim = embedding_dim
        self.impact_dim = impact_dim
        self.kernel = RBFKernel(sigma)
        self.leaf_size = leaf_size
        self.allow_simulated = allow_simulated
        self._versions = {}
        self._generation = 0
        self._cached_index = None
        self.last_update_seconds = 0.0

    @property
    def documents(self):
        return tuple(d for versions in self._versions.values() for d in versions)

    def _validate(self, document):
        if not isinstance(document, Document):
            raise TypeError("expected Document")
        vector(document.embedding, self.embedding_dim)
        vector(document.impact_vector, self.impact_dim)

    def _changed(self, start):
        self._generation += 1
        self._cached_index = None
        self.last_update_seconds = perf_counter() - start

    def insert_document(self, document):
        start = perf_counter()
        self._validate(document)
        if document.doc_id in self._versions:
            raise ValueError("doc_id already exists; use append_revision")
        self._versions[document.doc_id] = (document,)
        self._changed(start)

    def append_revision(self, document):
        start = perf_counter()
        self._validate(document)
        versions = self._versions.get(document.doc_id)
        if versions is None:
            raise ValueError("unknown doc_id")
        if any(d.version_id == document.version_id for d in versions):
            raise ValueError("duplicate version_id")
        if any((d.source, d.entity_id) != (document.source, document.entity_id)
               for d in versions):
            raise ValueError("revision must retain source and entity")
        # Accept out-of-order ingestion; snapshot orders by known availability,
        # with append order breaking ties. Historical intervals stay immutable.
        self._versions[document.doc_id] = versions + (document,)
        self._changed(start)

    def snapshot(self, cutoff):
        cutoff = utc(cutoff)
        visible = []
        for versions in self._versions.values():
            eligible = [(d.available_at, i, d) for i, d in enumerate(versions)
                        if d.accessible(cutoff, self.allow_simulated)
                        and d.valid_from <= cutoff]
            if eligible:
                newest = max(eligible, key=lambda row: (row[0], row[1]))[2]
                if newest.valid_to is None or cutoff < newest.valid_to:
                    visible.append(newest)
        return tuple(visible)

    def visible_documents(self, cutoff):
        return self.snapshot(cutoff)

    def aggregate_exact(self, query, cutoff):
        vector(query, self.embedding_dim)
        return exact_aggregate(self.snapshot(cutoff), query, self.kernel, self.impact_dim)

    def query_certified(self, query, cutoff, epsilon, lipschitz=1.0):
        """Includes cold snapshot/index construction in time and read counters.

        The one-entry cache deliberately avoids claiming a dynamic temporal
        index: new cutoffs/updates rebuild a legal-snapshot spatial tree.
        """
        from .index import CITRAIndex
        start = perf_counter()
        cutoff = utc(cutoff)
        key = (cutoff, self._generation)
        cache_hit = self._cached_index is not None and self._cached_index[0] == key
        if cache_hit:
            index = self._cached_index[1]
        else:
            index = CITRAIndex(self.snapshot(cutoff), self.embedding_dim,
                               self.impact_dim, self.kernel, self.leaf_size)
            self._cached_index = (key, index)
        result = index.query(query, epsilon, lipschitz)
        result.stats.cache_hit = cache_hit
        result.stats.metadata_records_checked = 0 if cache_hit else len(self.documents)
        result.stats.index_document_reads = 0 if cache_hit else index.build_document_reads
        result.stats.build_seconds = 0.0 if cache_hit else index.build_seconds
        result.stats.elapsed_seconds = perf_counter() - start
        return result
