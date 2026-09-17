# Prepared libr4d submissions

These three independently applicable patches target libr4d commit
`5dc6302b87d598d1d3bf2ad3b50aab365461a63c`. Each has its own branch and signed-off
commit. `manifest.json` records the commit and patch hashes. They have **not**
been opened as Codeberg PRs: this environment has no configured Codeberg account
authentication.

| Patch | Review scope | Publication checks |
| --- | --- | --- |
| `attention.patch` | Experimental query-isolated M8 attention with shared KV reads; current-header build and original native reproducer | Current-header CPU HIP compilation passed; original source identities verified |
| `causal-prefill.patch` | Experimental causal prefill reference targeting existing R4D M1 arithmetic | Current-header CPU HIP compilation passed; preserved 52-case native reference evidence described separately |
| `serial-gdn-contract.patch` | Stock-FLA serial oracle, ordered Triton recurrence, adapters and state/acceptance regressions | 78 CPU checks passed; one retained-compiler-artifact check skipped |

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

Draft PR descriptions are beside the patches. After Codeberg authentication and
a fork are available, the existing branches can be pushed and submitted without
recreating the work. The source in each patch can also be reviewed directly from
this published bundle.

Each patch links directly to the [public report](../../REPORT.md). The manifest
records the current prepared commit and patch identities.

## Publish the existing branches

Create a Codeberg fork of `StillDeadcode/libr4d`. Replace `YOUR_ACCOUNT` below
with your Codeberg username, then run these commands on the development host:

```sh
git -C /home/lewis/tasks/qwen-pr-libr4d-20260917 push \
  https://codeberg.org/YOUR_ACCOUNT/libr4d.git fix/query-isolated-m8-attention
git -C /home/lewis/tasks/qwen-pr-libr4d-prefill-20260917 push \
  https://codeberg.org/YOUR_ACCOUNT/libr4d.git fix/causal-gdn-prefill
git -C /home/lewis/tasks/qwen-pr-libr4d-gdn-20260917 push \
  https://codeberg.org/YOUR_ACCOUNT/libr4d.git fix/serial-contract-gdn
```

Open one draft pull request per branch against `StillDeadcode/libr4d:main`.
Use the title and body from the corresponding file:

- [Attention title and body](attention-pr.md), with [attention.patch](attention.patch).
- [Causal-prefill title and body](causal-prefill-pr.md), with [causal-prefill.patch](causal-prefill.patch).
- [Serial-GDN title and body](serial-gdn-contract-pr.md), with [serial-gdn-contract.patch](serial-gdn-contract.patch).

The descriptions distinguish the tested pinned Radiance integration from these
experimental upstream ports. Fix 2's eager/compiled rounding changes belong to
the Radiance integration and the already-merged Triton repair; they do not add
another libr4d patch.
