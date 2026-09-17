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

The patch mailboxes retain their original bytes and commit identities. Their
embedded historical report links point at the original private repository;
use the [public report](../../REPORT.md) and the updated descriptions beside
the patches when reviewing or preparing a submission.
