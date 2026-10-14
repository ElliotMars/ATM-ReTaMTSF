"""CITRA correctness prototype; independent of the legacy ReTaMTSF pipeline."""

from .temporal import Document, TemporalDocumentStore
from .index import CITRAIndex, QueryResult, QueryStats
from .kernel import RBFKernel, exact_aggregate

__all__ = ["Document", "TemporalDocumentStore", "CITRAIndex", "QueryResult",
           "QueryStats", "RBFKernel", "exact_aggregate"]
