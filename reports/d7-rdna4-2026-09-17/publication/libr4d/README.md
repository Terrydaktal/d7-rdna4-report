# libr4d submissions

These three independently applicable patches target libr4d commit
`5dc6302b87d598d1d3bf2ad3b50aab365461a63c`. Each has its own branch and signed-off
commit. All three are open as draft PRs from
[Terrydaktal/libr4d](https://codeberg.org/Terrydaktal/libr4d).
`manifest.json` records their PR URLs, submitted commits and patch hashes.

| Patch | Draft PR | Review scope | Publication checks |
| --- | --- | --- | --- |
| `attention.patch` | [#5](https://codeberg.org/StillDeadcode/libr4d/pulls/5) | Experimental query-isolated M8 attention with shared KV reads; current-header build and original native reproducer | Current-header CPU HIP compilation passed; original source identities verified |
| `causal-prefill.patch` | [#6](https://codeberg.org/StillDeadcode/libr4d/pulls/6) | Experimental causal prefill reference targeting existing R4D M1 arithmetic | Current-header CPU HIP compilation passed; preserved 52-case native reference evidence described separately |
| `serial-gdn-contract.patch` | [#7](https://codeberg.org/StillDeadcode/libr4d/pulls/7) | Stock-FLA serial oracle, ordered Triton recurrence, adapters and state/acceptance regressions | 78 CPU checks passed; one retained-compiler-artifact check skipped |

These are additive experimental submissions. They do not install changes in the
current libr4d public dispatcher. In particular, the stock-FLA reference still
needs a production HIP port and native integration qualification. The tested
complete Radiance repair is already submitted in
[Radiance #11](https://github.com/magiccodingman/vllm-radiance/pull/11), with its
shared support in [#10](https://github.com/magiccodingman/vllm-radiance/pull/10).

No patch should be described as a completed production libr4d integration merely
because its equivalent downstream experiment passed. The report's completed
10,000-position model comparison remains evidence for the exact pinned compiled
Radiance stack.

Draft descriptions are beside the patches. The submitted branches contain the
exact prepared commits; their source and validation evidence are unchanged.
The source in each patch can also be reviewed directly from this published bundle.

Each patch links directly to the [public report](../../REPORT.md). The manifest
records the current prepared commit and patch identities.

## Update the existing branches

After committing a reviewed update, push the corresponding branch from the
development host. Codeberg updates its existing PR automatically:

```sh
git -C /home/lewis/tasks/qwen-pr-libr4d-20260917 push \
  https://codeberg.org/Terrydaktal/libr4d.git fix/query-isolated-m8-attention
git -C /home/lewis/tasks/qwen-pr-libr4d-prefill-20260917 push \
  https://codeberg.org/Terrydaktal/libr4d.git fix/causal-gdn-prefill
git -C /home/lewis/tasks/qwen-pr-libr4d-gdn-20260917 push \
  https://codeberg.org/Terrydaktal/libr4d.git fix/serial-contract-gdn
```

The PRs target `StillDeadcode/libr4d:main`. Their descriptions are retained here;
Codeberg's `WIP:` title prefix marks each submission as a draft:

- [Attention title and body](attention-pr.md), with [attention.patch](attention.patch).
- [Causal-prefill title and body](causal-prefill-pr.md), with [causal-prefill.patch](causal-prefill.patch).
- [Serial-GDN title and body](serial-gdn-contract-pr.md), with [serial-gdn-contract.patch](serial-gdn-contract.patch).

The descriptions distinguish the tested pinned Radiance integration from these
experimental upstream ports. Fix 2's eager/compiled rounding changes belong to
the Radiance integration and the already-merged Triton repair; they do not add
another libr4d patch.
