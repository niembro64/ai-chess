"""Tests for the protocol-3 JESTER launch-time opponent pool."""

from scripts._jester_launcher import refresh_archive_opponents, select_spaced_archives


def test_select_spaced_archives_excludes_the_two_newest(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    for generation in range(1_000, 13_000, 1_000):
        (archive / f"gen-{generation}.pt").write_bytes(str(generation).encode())

    selected = select_spaced_archives(archive, 3)
    assert [path.stem for path in selected] == ["gen-1000", "gen-5000", "gen-10000"]


def test_refresh_archive_opponents_preserves_manual_checkpoint(tmp_path):
    archive = tmp_path / "archive"
    pool = tmp_path / "opponents"
    archive.mkdir()
    pool.mkdir()
    for generation in range(1_000, 7_000, 1_000):
        (archive / f"gen-{generation}.pt").write_bytes(str(generation).encode())
    (pool / "manual.pt").write_bytes(b"manual")
    (pool / "archive-gen-00000001.pt").write_bytes(b"stale")

    selected = refresh_archive_opponents(tmp_path, pool, 2)

    assert [path.name for path in selected] == [
        "archive-gen-00001000.pt",
        "archive-gen-00004000.pt",
    ]
    assert (pool / "manual.pt").read_bytes() == b"manual"
    assert not (pool / "archive-gen-00000001.pt").exists()
