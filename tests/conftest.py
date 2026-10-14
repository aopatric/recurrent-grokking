"""pytest setup shared by all tests."""

import os

# the hugging face loading test reads local files only; keep `datasets` off the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
