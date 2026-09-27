"""Exceptions raised by the pipeline."""


class FrcXdataError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(FrcXdataError):
    """A config file or environment variable is missing or invalid."""


class UnmappedLabelError(FrcXdataError):
    """A source label has no entry in the class map."""


class DataLeakError(FrcXdataError):
    """A field-test image turned up in data meant for training."""


class DirtyTreeError(FrcXdataError):
    """Results would be written from a working tree with uncommitted changes."""


class SplitLeakError(FrcXdataError):
    """A test image has a near duplicate in the train or valid split."""


class UploadCheckError(FrcXdataError):
    """Data sent to or received from Roboflow does not match the harmonized dataset."""


class PredictionCacheTooLargeError(FrcXdataError):
    """Cached predictions would be too large to commit under reports/."""
