"""Synthetic data fixture shared by Shake-Shake tests; no competition data."""
from pathlib import Path
import tempfile
from unittest import mock
import test_pipeline

class ShakeFixture:
    @classmethod
    def setUpClass(cls):
        target = Path(__file__).resolve().parents[1] / "runs" / "cpu_test_temp"
        target.mkdir(parents=True, exist_ok=True)
        original = tempfile.TemporaryDirectory
        def directory(*args, **kwargs):
            kwargs["dir"] = str(target)
            return original(*args, **kwargs)
        with mock.patch.object(test_pipeline.tempfile, "TemporaryDirectory", directory):
            test_pipeline.PipelineTests.setUpClass.__func__(cls)
