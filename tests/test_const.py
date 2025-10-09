from common_utils.const import TMPDIR

def test_tmpdir_defined():
    assert TMPDIR is not None
    assert isinstance(TMPDIR, str)
