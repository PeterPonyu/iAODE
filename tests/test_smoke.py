import iaode


def test_package_metadata() -> None:
    assert isinstance(iaode.__version__, str)
    assert iaode.__version__.startswith("0."), "Expected release-style version"
    assert "agent" in iaode.__all__


def test_get_data_dir_idempotent() -> None:
    cache_dir = iaode.datasets.get_data_dir()
    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert (cache_dir / "atacseq").is_dir()
    assert (cache_dir / "annotations").is_dir()

    # Ensure calling twice is safe and does not raise
    second = iaode.datasets.get_data_dir()
    assert second.samefile(cache_dir)
    assert (second / "atacseq").exists()
    assert (second / "annotations").exists()
