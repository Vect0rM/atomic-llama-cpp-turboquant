# Qwen 3.x NextN — shared-model speculative decoding

> Scope: **Qwen3.6** (and compatible) models with NextN / MTP auxiliary head weights in GGUF.
> The draft context now reuses the **target** `llama_model` (no second mmap of the combined
> `_MTP.gguf`); a second `llama_context` is built over the same model with
> `llama_context_params.nextn_draft = true`, which routes graph build to the NextN draft
> builder (`qwen35_nextn` / `qwen35moe_nextn`).
> Legacy standalone `*_mtp` GGUFs (`override_arch`) are still supported as a fallback for
> users who ship the draft head as a separate artifact.
> This path is **named `nextn`** in this fork to coexist with **Gemma 4 MTP** (`--spec-type mtp`), which uses a
> single target context and `llama_decode_mtp_*`.

See also `MTP.md` (Gemma) and `docs/speculative.md` for shared CLI concepts.

---

## 1. Architecture

| Piece | Role |
|-------|------|
| Target context | Standard `qwen35` / `qwen35moe` forward; graph publishes `t_h_pre_norm` (hidden before final norm). |
| Draft context | Built over the **same** `llama_model` with `cparams.nextn_draft = true`. The graph dispatcher picks `llm_build_qwen35*_nextn` against the target's NextN-layer tensors (`model.layers[n_main + i].nextn.*`). KV cache is sized only for the NextN layer (`kv_only_nextn = true`, overridden transparently in `llama_context` ctor). |
| Hidden transfer | Target and draft enable `embeddings_pre_norm`; `llama_decode` copies `t_h_pre_norm` rows into a CPU `embd_pre_norm` buffer. `common_speculative_state_nextn` reads via `llama_get_embeddings_pre_norm_ith` (no per-ubatch tensor hook). |
| Speculative driver | `common_speculative_state_nextn` in `common/speculative.cpp` (greedy Top-1 chain). |
| KV pairing | `llama_set_nextn(target, draft)` registers the draft context so `llama_context_nextn_seq_rm` can trim both KVs. |

The shared-model path eliminates the ~22 GB second mmap (one `MTLBuffer` per `llama_model`)
that used to OOM the 35B-A3B target on Apple Silicon (38 GB unified memory). See
`llama_model_has_nextn_layer()` (target arch ∈ {qwen35, qwen35moe} **and**
`hparams.nextn_predict_layers > 0`).

---

## 2. CLI / server

- `--spec-type nextn` — enable NextN drafting (not Gemma `mtp`).
- `--model-draft` / `-md` — pass the **same** path as `--model`; the server detects this
  and switches to the shared-model path (no second model load). Pointing at a standalone
  NEXTN_ONLY GGUF (`general.architecture = qwen35*_mtp`) still works but loads a second
  `llama_model`.
- `--draft-max` / `--spec-draft-n-max` — max chained draft tokens per round (see `common` / server arg naming).
- Gemma MTP flags (`--mtp-head`, `llama_decode_mtp_*`, `llama_model_load_mtp_from_file`) are **unchanged**.

---

## 3. C API (subset)

- `llama_set_nextn(target_ctx, draft_ctx)` — pair contexts for paired `seq_rm`.
- `llama_context_nextn_seq_rm(target_ctx, …)` — remove KV on target **and** on the registered draft context (`seq_id` 0 on draft).

Internal (see `src/llama-ext.h`, not in stable `include/llama.h`):

- `llama_set_embeddings_pre_norm(ctx, bool)` — enable extraction/copy of pre-norm hidden rows into `embd_pre_norm`.
- `llama_get_embeddings_pre_norm_ith(ctx, i)` — row `i` of the last decode’s pre-norm buffer (`i < 0` supported like other embedding getters).

---

## 4. Operations

- **Vocab**: draft and target share tokenizer; arch check ensures `qwen35`+`qwen35_mtp` (or MoE pair).
- **GDN rollback**: target may use `n_rs_seq` from speculative+GDN work; draft context forces `n_rs_seq = 0` (see `tools/server/server-context.cpp`).
- **Metal / Vulkan**: GDN partial rollback quality may still be upstream-limited; see PR #22400 notes in the project plan.

---

## 5. Verify GGUF

```bash
PYTHONPATH=gguf-py python3 scripts/verify-qwen36-nextn-gguf.py /path/to/model.gguf
```

---

## 6. Run scripts

- `scripts/run-qwen36-27b-nextn-server.sh`
- `scripts/run-qwen36-35ba3b-nextn-server.sh`

Set `MAIN_GGUF` to your Qwen3.6 GGUF; draft defaults to the same path.

---

## 7. Performance notes (Apple M4 Max, Metal)

Median TPS over 3 runs, prompt = 50-token instruction, `--draft-max=2 --draft-min=1`,
NextN draft DM=2 (single async chain), context 8192. See `.scratch/bench-logs/qwen-matrix-shared-*.md`.

| model | mode | short tps (n=128) | long tps (n=512) | accept (long) | Δ vs base (long) |
|---|---|---:|---:|---:|---:|
| qwen-27B dense | f16-base | 20.82 | 20.49 | — | — |
| qwen-27B dense | f16-nextn | 20.33 | 18.93 | 72.0% | **−7.6%** |
| qwen-27B dense | turbo3-base | 18.41 | 17.85 | — | — |
| qwen-27B dense | turbo3-nextn | 17.88 | 15.72 | 65.4% | **−11.9%** |
| qwen-35B-A3B MoE | f16-base | 69.31 | 69.30 | — | — |
| qwen-35B-A3B MoE | f16-nextn | 91.86 | 83.63 | 66.1% | **+20.7%** |
| qwen-35B-A3B MoE | turbo3-base | 62.46 | 61.97 | — | — |
| qwen-35B-A3B MoE | turbo3-nextn | 84.91 | 78.41 | 67.7% | **+26.5%** |

**Where NextN helps**: MoE targets (qwen-35B-A3B) — verify is heavy enough that the draft
compute fully overlaps via the async pipeline. Wins range from **+20% (f16, long)** to
**+36% (turbo3, short)**.

**Known limitation: 27B dense NextN draft is draft-compute-bound.** The NextN-layer is a
full transformer block, so on a dense model `t_draft ≈ 2.6× t_verify`. The async pipeline
cannot overlap that fully → speculative wins are negative or paritetical. turbo3 KV
quantization adds another **~7%** to draft compute (Metal dequant overhead inside the
NextN attention), pushing 27B turbo3-nextn long to **−12%** vs baseline. This is not a bug:
isolated diagnostics (`accept_token` 71.2% f16 ≈ 71.5% turbo3 — H1/H3 rejected,
`t_draft` 1354 → 1449 ms — H4 partially confirmed) point to physical compute limits on
M4 Max. Stick to f16 KV when running NextN on dense Qwen3.6 27B if every percent matters.
