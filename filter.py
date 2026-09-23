"""Root filter alias forwarding to app.filter."""

from app.filter import _filter_by_metadata, filter_by_metadata

__all__ = ["_filter_by_metadata", "filter_by_metadata"]
