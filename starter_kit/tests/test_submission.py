"""Loading submissions (no data or weights needed)."""

import textwrap

import evaluate


def _submission(folder, value):
    folder.mkdir()
    (folder / "model.py").write_text(f"VALUE = {value}\n")
    (folder / "generate.py").write_text(textwrap.dedent("""
        from model import VALUE
        def load_model(device):
            return VALUE
        def generate(model, labels, seed):
            return None
    """))
    return folder


def test_each_submission_imports_its_own_sibling_modules(tmp_path):
    first = evaluate.load_submission(_submission(tmp_path / "a", 1))
    second = evaluate.load_submission(_submission(tmp_path / "b", 2))
    assert first.load_model("cpu") == 1
    assert second.load_model("cpu") == 2
