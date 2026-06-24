from code_jepa.transforms.python_ast import hard_negative_views, positive_views


SAMPLE = '''
def contains(nums, target):
    """Return whether target is in nums."""
    for i in range(len(nums) - 1):
        if nums[i] == target and equals(nums[i], target) and i < len(nums):
            return True
    return False
'''


def test_positive_views_compile_and_include_rename_or_docstring() -> None:
    views = positive_views(SAMPLE)
    assert views
    assert all(view.role == "positive" for view in views)
    assert all("def contains" in view.code for view in views)
    assert {view.name for view in views} & {"rename_locals", "remove_docstrings", "ast_normalize"}


def test_hard_negatives_compile_and_record_spans() -> None:
    views = hard_negative_views(SAMPLE)
    names = {view.name for view in views}
    assert "flip_comparison" in names
    assert "swap_call_args" in names
    assert all(view.role == "negative" for view in views)
    assert all(view.changed_spans for view in views)


def test_comparison_negative_changes_equality() -> None:
    view = next(view for view in hard_negative_views(SAMPLE) if view.name == "flip_comparison")
    assert "!=" in view.code or "<=" in view.code or ">=" in view.code
