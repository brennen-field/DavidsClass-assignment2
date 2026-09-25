from app import build_app


def test_app_builds():
    assert build_app() is not None
