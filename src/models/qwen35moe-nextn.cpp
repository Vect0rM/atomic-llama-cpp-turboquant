#include "models.h"

// NextN draft head graph for MoE (GGUF arch string: qwen35moe_mtp).
llm_build_qwen35moe_nextn::llm_build_qwen35moe_nextn(
        const llama_model & model,
        const llm_graph_params & params) :
        llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35MOE_NEXTN requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "QWEN35MOE_NEXTN currently only supports a single NextN block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "NextN block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "NextN block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "NextN block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "NextN block missing ffn_gate_inp");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "nextn_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "nextn_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    // qwen35moe target arch uses hybrid memory (attn + GDN recurrent). When the draft
    // context shares the target llama_model, mctx is llama_memory_hybrid_context,
    // not llama_kv_cache_context — so we must route through the hybrid input and
    // pick its attn slot. (Casting hybrid -> pure KV directly would be UB and SIGSEGV.)
    auto * inp_hybrid     = build_inp_mem_hybrid();
    auto * inp_attn       = inp_hybrid->get_attn();

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "nextn_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "nextn_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "nextn_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "nextn_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "nextn_attn_norm", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "nextn_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "nextn_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "nextn_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "nextn_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "nextn_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "nextn_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "nextn_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "nextn_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "nextn_attn_post_norm", il);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "nextn_ffn_moe_out", il);

    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "nextn_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "nextn_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "nextn_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    cb(cur, "nextn_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "nextn_post_ffn", il);

    res->t_h_pre_norm = cur;
    res->t_nextn_out  = cur;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE_NEXTN: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "nextn_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "QWEN35MOE_NEXTN: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
