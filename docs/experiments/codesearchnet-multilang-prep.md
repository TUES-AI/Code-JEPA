# CodeSearchNet multilingual prep

Run: `codesearchnet-multilang-20260628-142024`.

## S3 targets

- data: `s3://code-jepa/data/codesearchnet/`
- tokenizers: `s3://code-jepa/tokenizers/codesearchnet/`

## Commands

```bash
PYTHONPATH=src /Volumes/SSD/v/wiki/bin/python scripts/prepare_data.py --output-dir /Volumes/SSD/datasets/code-jepa/prep/codesearchnet-multilang-20260628-142024 --datasets codesearchnet --languages python java javascript go php ruby --splits train validation test --transform-stages v0 v1 v2 --max-positive-views 32 --max-negative-views 32 --strict-transform-coverage --task-datasets --cache-dir /Volumes/SSD/huggingface/huggingface --num-workers 8 --worker-buffer-size 16 --shard-size 20000
PYTHONPATH=src /Volumes/SSD/v/wiki/bin/python scripts/prepare_data.py --only-transform <transform> --scan-until-yield 100 ...
PYTHONPATH=src /Volumes/SSD/v/wiki/bin/python scripts/train_code_bpe_tokenizer.py --input-roots <six transform-v0/core roots> --vocab-size {16384,32768,50000} --max-units 0
```

## Total counts

| table | rows |
| --- | --- |
| files | 7,898,372 |
| relations | 14,614,022 |
| spans | 409,306,897 |
| triples | 59,404,781 |
| units | 18,156,201 |
| views | 120,452,807 |

## Per-language counts

| language | files | units | views | triples | spans |
| --- | --- | --- | --- | --- | --- |
| python | 1,358,796 | 4,055,996 | 33,114,610 | 20,939,764 | 86,652,723 |
| java | 1,580,775 | 3,833,851 | 26,491,951 | 11,708,873 | 77,809,313 |
| javascript | 551,864 | 1,671,675 | 10,806,284 | 5,749,255 | 41,444,568 |
| go | 2,055,045 | 3,379,881 | 16,363,063 | 6,801,782 | 89,979,389 |
| php | 2,138,808 | 4,773,822 | 30,894,639 | 12,908,509 | 104,573,396 |
| ruby | 213,084 | 440,976 | 2,782,260 | 1,296,598 | 8,847,508 |

## Per-language/stage counts

| language | stage | subsegment | files | units | views | triples |
| --- | --- | --- | --- | --- | --- | --- |
| go | v0 | core | 346,365 | 1,153,128 | 5,626,632 | 2,869,233 |
| go | v1 | core | 346,365 | 769,201 | 5,538,722 | 1,561,746 |
| go | v1 | only-bool_return_simplify | 346,365 | 353,203 | 166 | 76 |
| go | v1 | only-if_return_merge | 139,532 | 141,365 | 2,000 | 1,034 |
| go | v1 | only-remove_unreachable_else | 73,126 | 73,899 | 2,016 | 1,112 |
| go | v2 | core | 346,365 | 424,038 | 5,189,675 | 2,366,645 |
| go | v2 | only-accumulator_loop_to_builtin | 317,088 | 323,371 | 1,932 | 974 |
| go | v2 | only-append_loop_to_collection_literal | 139,839 | 141,676 | 1,920 | 962 |
| java | v0 | core | 496,163 | 1,800,632 | 9,188,070 | 5,037,503 |
| java | v1 | core | 496,163 | 1,267,151 | 9,174,853 | 3,531,604 |
| java | v2 | core | 496,163 | 669,351 | 8,127,410 | 3,138,986 |
| java | v2 | only-append_loop_to_collection_literal | 92,286 | 96,717 | 1,618 | 780 |
| javascript | v0 | core | 137,966 | 672,869 | 3,449,123 | 2,033,950 |
| javascript | v1 | core | 137,966 | 562,928 | 3,984,397 | 2,267,459 |
| javascript | v2 | core | 137,966 | 254,705 | 3,372,734 | 1,447,832 |
| javascript | v2 | only-accumulator_loop_to_builtin | 137,966 | 181,173 | 30 | 14 |
| php | v0 | core | 577,304 | 1,943,975 | 9,962,634 | 5,164,545 |
| php | v1 | core | 577,304 | 1,625,096 | 11,368,918 | 5,445,158 |
| php | v2 | core | 577,304 | 787,592 | 9,559,605 | 2,297,012 |
| php | v2 | only-accumulator_loop_to_builtin | 370,950 | 380,052 | 1,790 | 942 |
| php | v2 | only-append_loop_to_collection_literal | 35,946 | 37,107 | 1,692 | 852 |
| python | v0 | core | 452,932 | 1,640,080 | 9,892,938 | 8,006,432 |
| python | v1 | core | 452,932 | 1,508,222 | 11,981,631 | 8,598,100 |
| python | v2 | core | 452,932 | 907,694 | 11,240,041 | 4,335,232 |
| ruby | v0 | core | 53,271 | 180,728 | 923,466 | 446,477 |
| ruby | v1 | core | 53,271 | 127,167 | 925,819 | 345,450 |
| ruby | v2 | core | 53,271 | 79,410 | 932,666 | 504,517 |
| ruby | v2 | only-accumulator_loop_to_builtin | 53,271 | 53,671 | 309 | 154 |

## Appendability

Each transform stage is a parent directory with subsegments such as `core/` and `only-<transform>/`. New transform families can be appended later by writing another subsegment under the same `language/transform-vX/` parent and updating parent/root manifests.

## Transform coverage

All languages implement the same canonical v0/v1/v2 transform family names. Missing implementation is a startup error. Natural yield differs by language and by what CodeSearchNet function snippets contain.

Remaining zero-yield families after full run + targeted scans:

| language | stage | role | transform | attempted units |
| --- | --- | --- | --- | --- |
| python | v1 | positive | import_sort_same_block | 498,688 |
| java | v1 | positive | import_sort_same_block | 521,353 |

Explanation: `import_sort_same_block` remained zero for Python and Java because CodeSearchNet stores function/method snippets, not full files with import blocks. The transform is implemented and should yield on whole-file data.

Low-yield nonzero families:

| language | stage | role | transform | yielded | attempted units |
| --- | --- | --- | --- | --- | --- |
| go | v1 | positive | bool_return_simplify | 8 | 353,203 |
| go | v1 | positive | import_sort_same_block | 4 | 353,203 |
| go | v2 | negative | remove_guard_branch | 14 | 353,203 |
| java | v2 | positive | accumulator_loop_to_builtin | 7 | 521,353 |
| javascript | v2 | positive | accumulator_loop_to_builtin | 2 | 181,173 |
| javascript | v2 | positive | append_loop_to_collection_literal | 11 | 181,173 |
| python | v2 | positive | accumulator_loop_to_builtin | 3 | 498,688 |
| ruby | v2 | positive | accumulator_loop_to_builtin | 15 | 53,671 |
| ruby | v2 | positive | range_loop_to_while_or_equivalent | 19 | 53,671 |

## Tokenizers

| name | vocab | trained_units | input_shards | sample_tokens |
| --- | --- | --- | --- | --- |
| bpe16k | 16,384 | 7,391,412 | 373 | 15 |
| bpe32k | 32,768 | 7,391,412 | 373 | 15 |
| bpe50k | 50,000 | 7,391,412 | 373 | 15 |

All tokenizers load with `transformers.PreTrainedTokenizerFast`; special ids are `<pad>` 0, `<bos>` 1, `<eos>` 2, `<unk>` 3.

## Known weak/noisy transforms

- Non-Python transforms are conservative text/tree-sitter-span rewrites, not compiler-proven equivalences.
- Some v2 loop-to-builtin transforms are rare in CodeSearchNet snippets; they are implemented but low-yield.
- Import sorting is structurally absent from Python/Java CodeSearchNet function rows; whole-file corpora should exercise it.
- Hard negatives are behavior-impacting mutations, not guaranteed failing tests.
