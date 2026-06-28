from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from code_jepa.data.python_ast import parse_and_compile
from code_jepa.data.prep_pipeline import (
    PipelineConfig,
    SegmentWriter,
    SourceFile,
    dataset_file_limit,
    records_from_source_file,
    records_from_task,
    task_from_row,
)
from code_jepa.transforms.python_ast import hard_negative_views_for_stage, positive_views_for_stage


SAMPLE_SOURCE = '''
import os
from pathlib import Path

class Bag:
    def __init__(self, items):
        self.items = items.copy()

    def add_path(self, path, reverse=False):
        def normalize(value):
            return str(value).strip()
        values = self.items.copy()
        for i in range(len(values)):
            if i >= len(values):
                return []
            try:
                with open(path) as handle:
                    values.append(normalize(handle.read()))
            except ValueError:
                return []
        return sorted(values, key=str, reverse=reverse)
'''


def cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=str(tmp_path),
        datasets=[],
        task_datasets=[],
        transform_stages=["v2"],
        splits=["train"],
        shard_size=4,
        min_loc=1,
        max_loc=80,
    )


def test_v2_transform_stage_adds_harder_compile_valid_negatives() -> None:
    views = hard_negative_views_for_stage(SAMPLE_SOURCE, "v2", max_views=8)
    names = {view.name for view in views}

    assert {"shift_range_bound", "remove_guard_branch", "flip_exception_type", "drop_keyword_argument", "replace_copy_with_alias", "drop_context_manager"} & names
    assert all(view.role == "negative" for view in views)
    assert all(view.changed_spans for view in views)
    assert all("start_byte" in view.changed_spans[0] for view in views)
    assert all("end_byte" in view.changed_spans[0] for view in views)
    assert all(view.confidence == "behavior_impacting" for view in views)
    assert all(parse_and_compile(view.code).compile_ok for view in views)


def test_positive_transform_stages_are_deltas_and_include_hard_rewrites() -> None:
    sample = '''
def total(xs):
    a = 1
    b = 2
    if len(xs) == 0:
        return True
    else:
        return False

def as_list(xs):
    out = []
    for x in xs:
        out.append(x + 1)
    return out

def summed(xs):
    total = 0
    for x in xs:
        total += x
    return total
'''
    v0 = {view.name for view in positive_views_for_stage(sample, "v0", max_views=20)}
    v1 = {view.name for view in positive_views_for_stage(sample, "v1", max_views=20)}
    v2 = {view.name for view in positive_views_for_stage(sample, "v2", max_views=20)}

    assert "rename_locals" in v0
    assert {"swap_independent_assignments", "bool_return_simplify", "remove_unreachable_else"} & v1
    assert not (v0 & v1)
    assert {"list_append_loop_to_comprehension", "accumulator_loop_to_sum"} & v2
    assert not (v0 & v2)


def test_dataset_specific_file_limit_overrides_global_limit(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    capped = PipelineConfig(**{**config.__dict__, "max_files_per_dataset": None, "dataset_max_files": ("codeparrot_clean_python=300000",)})

    assert dataset_file_limit("codeparrot_clean_python", capped) == 300000
    assert dataset_file_limit("codesearchnet_python", capped) is None


def test_records_from_source_file_builds_hierarchical_tokenizer_agnostic_records(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    source_file = SourceFile(
        dataset_key="unit_test",
        source_dataset="unit-test",
        source_split="train",
        source_row_index=0,
        repository_name="repo",
        path="bag.py",
        language="python",
        source=SAMPLE_SOURCE,
        metadata={},
    )

    records = records_from_source_file(source_file, "v2", config)
    unit_families = {row["unit_family"] for row in records["units"]}
    view_families = {row["family"] for row in records["views"]}
    context_views = [row for row in records["views"] if row["family"] == "context_triplet"]

    assert {"class", "method", "nested_function", "span_window"} <= unit_families
    assert {"focal_triplet", "context_triplet", "local_span_triplet", "ast_aux"} <= view_families
    assert records["triples"]
    assert context_views
    assert context_views[0]["code"].startswith("# <")
    assert all("input_ids" not in row for table in records.values() for row in table)


def test_records_from_source_file_skips_recursion_errors(tmp_path: Path, monkeypatch) -> None:
    import code_jepa.data.prep_pipeline as prep_pipeline

    source_file = SourceFile("unit_test", "unit-test", "train", 0, "repo", "bad.py", "python", "def f():\n    return 1\n", {})
    monkeypatch.setattr(prep_pipeline, "parse_and_compile", lambda _code: (_ for _ in ()).throw(RecursionError()))

    records = records_from_source_file(source_file, "v2", cfg(tmp_path))

    assert all(not rows for rows in records.values())


def test_segment_writer_flushes_reproducible_segment(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    source_file = SourceFile("unit_test", "unit-test", "train", 0, "repo", "bag.py", "python", SAMPLE_SOURCE, {})
    records = records_from_source_file(source_file, "v2", config)

    writer = SegmentWriter(tmp_path / "segments" / "unit_test" / "transform-v2", dataset_key="unit_test", segment_name="transform-v2", cfg=config)
    writer.add_many(records)
    writer.close()

    manifest = json.loads((tmp_path / "segments" / "unit_test" / "transform-v2" / "manifest.json").read_text())
    unit_shards = sorted((tmp_path / "segments" / "unit_test" / "transform-v2" / "units").glob("*.parquet"))
    triple_shards = sorted((tmp_path / "segments" / "unit_test" / "transform-v2" / "triples").glob("*.parquet"))

    assert manifest["tokenizer_agnostic"] is True
    assert unit_shards
    assert triple_shards
    assert pq.read_table(unit_shards[0]).num_rows > 0
    assert pq.read_table(triple_shards[0]).num_rows > 0


def test_task_records_create_same_task_and_cross_language_pairs(tmp_path: Path) -> None:
    task = {
        "task_id": "sum-two",
        "split": "train",
        "prompt": "sum two integers",
        "metadata": {"source": "unit"},
        "solutions": [
            {"language": "python", "code": "def solve(a, b):\n    return a + b\n", "solution_id": "py1"},
            {"language": "python", "code": "def solve(a, b):\n    total = a\n    total += b\n    return total\n", "solution_id": "py2"},
            {"language": "java", "code": "int solve(int a, int b) { return a + b; }\n", "solution_id": "java1"},
        ],
    }

    records = records_from_task(task, "unit_task", cfg(tmp_path))
    relation_types = {row["relation_type"] for row in records["semantic_pairs"]}

    assert len(records["semantic_pairs"]) == 3
    assert "same_task_different_solution" in relation_types
    assert "cross_language_same_task" in relation_types
    assert all(row["semantic_label"] == "close" for row in records["semantic_pairs"])


def test_apps_json_rows_create_real_semantic_pairs(tmp_path: Path) -> None:
    row = {
        "id": "apps-1",
        "question": "sum two integers",
        "solutions": json.dumps(
            [
                "def solve(a, b):\n    return a + b\n",
                "def solve(a, b):\n    total = a\n    total += b\n    return total\n",
            ]
        ),
    }

    task = task_from_row("apps", "train", 0, row)
    assert task is not None
    records = records_from_task(task, "apps", cfg(tmp_path))

    assert len(records["units"]) == 2
    assert len(records["semantic_pairs"]) == 1
    assert records["semantic_pairs"][0]["relation_type"] == "same_task_different_solution"


def test_codecontests_language_mapping_keeps_python2_cpp_python_java(tmp_path: Path) -> None:
    row = {
        "name": "langs",
        "description": "language mapping",
        "solutions": {
            "language": [1, 2, 3, 4],
            "solution": [
                "print 'YES'\n",
                "int main() { return 0; }\n",
                "print('YES')\n",
                "class Main { public static void main(String[] args) {} }\n",
            ],
        },
    }

    task = task_from_row("codecontests", "train", 0, row)
    assert task is not None
    records = records_from_task(task, "codecontests", cfg(tmp_path))
    units_by_language = {row["language"]: row for row in records["units"]}

    assert {"python2", "cpp", "python", "java"} <= set(units_by_language)
    assert units_by_language["python"]["parse_ok"] is True
    assert units_by_language["python2"]["parse_ok"] is False
    assert {row["relation_type"] for row in records["semantic_pairs"]} == {"cross_language_same_task"}
