# pyccode AGENTS

This repository does not aim to rebuild the entire product shell of A. Its goal is to extract the core agent framework from A into a runnable, testable, and incrementally extensible Python kernel.

Current constraints:
- Preserve behavioral boundaries first instead of chasing file-for-file parity.
- Reuse the validated code shape, code style, tests, and repository layout from `pycodex` whenever that structure is compatible with A.
- Every important new module should be traceable to a concrete source area in A.
- Keep the primary loop first: `prompt/context -> model -> tool_use -> tool_result -> follow-up`.
- Leave UI, telemetry, GrowthBook, and feature-gated internal modules out for now.
- Keep new tools small and stable by default; prioritize file reads, directory listing, and content search.
- Align the default model-visible tool surface with the 22 tools seen in local non-bare A captures. Real execution can remain placeholder-backed while the older local execution subset stays available through `build_local_execution_tool_registry()`.

Documentation conventions:
- `docs/ARCHITECTURE.md` records the A-to-`pyccode` module mapping.
- `docs/ALIGNMENT.md` records the current alignment state against A and what is intentionally reused from `pycodex`.
- `docs/CONTEXT.md` records the current Anthropic Messages request and context contract.
- `docs/MESSAGES_SERVER.md` records server-side behavior observations and the boundary of the local fake server.
- `tests/compare_three_way_messages_requests.py` and the generated comparison artifacts are the standard entry points for request-shape comparison. The default target is the local non-bare A capture; the bare capture is only a smaller validation slice.
- Local request captures should give `pyccode` its own fake OAuth home so it does not accidentally read real credentials; the relevant artifact summaries should stay separate from real user state.
- Default A-shaped tool schemas live directly in the placeholder tool schema module. If a newer non-bare capture is collected, validate the code definitions against the extracted results instead of adding a separate schema snapshot file.
- The README should stay focused on goals, usage, and current scope rather than long-form project history.
