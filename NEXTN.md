# Qwen 3.x NextN — second-context speculative decoding

> Scope: **Qwen3.6** (and compatible) models with NextN / MTP auxiliary head weights in GGUF,
> using a **second** `llama_context` built from the same checkpoint with
> `llama_model_params.override_arch` set to `qwen35_mtp` or `qwen35moe_mtp` (after reading GGUF metadata).
> This path is **named `nextn`** in this fork to coexist with **Gemma 4 MTP** (`--spec-type mtp`), which uses a
> single target context and `llama_decode_mtp_*`.

See also `MTP.md` (Gemma) and `docs/speculative.md` for shared CLI concepts.

---

## 1. Architecture

| Piece | Role |
|-------|------|
| Target context | Standard `qwen35` / `qwen35moe` forward; graph publishes `t_h_pre_norm` (hidden before final norm). |
| Draft context | `qwen35_mtp` / `qwen35moe_mtp` graph (`LLM_ARCH_QWEN35_NEXTN` / `LLM_ARCH_QWEN35MOE_NEXTN`); own KV; logits for draft tokens. |
| Hidden transfer | Target and draft enable `embeddings_pre_norm`; `llama_decode` copies `t_h_pre_norm` rows into a CPU `embd_pre_norm` buffer. `common_speculative_state_nextn` reads via `llama_get_embeddings_pre_norm_ith` (no per-ubatch tensor hook). |
| Speculative driver | `common_speculative_state_nextn` in `common/speculative.cpp` (greedy Top-1 chain). |
| KV pairing | `llama_set_nextn(target, draft)` registers the draft context so `llama_context_nextn_seq_rm` can trim both KVs. |

---

## 2. CLI / server

- `--spec-type nextn` — enable NextN drafting (not Gemma `mtp`).
- `--model-draft` / `-md` — path to the **same** GGUF as the main model; the server loads it again with `override_arch` so the draft graph is the NextN head.
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
