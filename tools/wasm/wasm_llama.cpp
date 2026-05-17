// tools/wasm/wasm_llama.cpp
// Browser-resident Gemma 4 MTP inference via Emscripten.
// Exposes three C-callable exports: wasm_llama_init, wasm_llama_chat_completion, wasm_llama_health.
// JS glue uses cwrap() to call these; caller owns the returned char* and must call wasm_llama_free_str.

#include "llama.h"
#include <emscripten.h>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static llama_model        * g_model      = nullptr;
static llama_context      * g_ctx        = nullptr;
static const llama_vocab  * g_vocab      = nullptr;
static bool                 g_mtp_loaded = false;

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", (unsigned)c);
                    out += hex;
                } else {
                    out += (char)c;
                }
                break;
        }
    }
    return out;
}

static char * heap_str(const std::string & s) {
    char * p = new char[s.size() + 1];
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

extern "C" {

// Load target + optional MTP drafter from virtual-FS paths (populated by JS via Emscripten FS API).
// Returns 0 on success, negative on failure.
EMSCRIPTEN_KEEPALIVE
int wasm_llama_init(const char * target_path, const char * drafter_path) {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model      = nullptr;
        g_vocab      = nullptr;
        g_mtp_loaded = false;
    }

    llama_backend_init();
    llama_log_set([](enum ggml_log_level lvl, const char * text, void *) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) { fprintf(stderr, "%s", text); }
    }, nullptr);

    llama_model_params mparams  = llama_model_default_params();
    mparams.n_gpu_layers        = 0;  // CPU-only in WASM

    g_model = llama_model_load_from_file(target_path, mparams);
    if (!g_model) return -1;

    if (drafter_path && drafter_path[0] != '\0') {
        llama_model_params dparams = llama_model_default_params();
        dparams.n_gpu_layers       = 0;
        int rc = llama_model_load_mtp_from_file(g_model, drafter_path, dparams);
        g_mtp_loaded = (rc == 0) && llama_model_has_mtp_assistant(g_model);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 2048;
    cparams.n_batch              = 512;

    g_ctx   = llama_init_from_model(g_model, cparams);
    g_vocab = llama_model_get_vocab(g_model);

    return g_ctx ? 0 : -2;
}

// Returns heap-allocated JSON. Caller must call wasm_llama_free_str.
EMSCRIPTEN_KEEPALIVE
char * wasm_llama_health(void) {
    if (!g_ctx) {
        return heap_str("{\"status\":\"not_initialized\",\"mtp_loaded\":false}");
    }
    std::ostringstream ss;
    ss << "{\"status\":\"ok\",\"mtp_loaded\":" << (g_mtp_loaded ? "true" : "false") << "}";
    return heap_str(ss.str());
}

// request_json: {"messages":[{"role":"user","content":"..."}],"max_tokens":512}
// Returns: {"choices":[{"message":{"role":"assistant","content":"..."}}],
//           "_mtp_enabled":bool,"_spec_accept_rate":null,"_latency_ms":N,"_tps":N}
EMSCRIPTEN_KEEPALIVE
char * wasm_llama_chat_completion(const char * request_json) {
    if (!g_ctx || !g_model || !g_vocab) {
        return heap_str("{\"error\":\"not initialized\"}");
    }

    // Extract last user content from request JSON (no heavy parser dep — scan for "content":"...")
    std::string req(request_json);
    std::string content;
    int max_tokens = 512;

    // max_tokens field
    size_t mt_pos = req.find("\"max_tokens\"");
    if (mt_pos != std::string::npos) {
        size_t vs = req.find_first_of("0123456789", mt_pos + 12);
        if (vs != std::string::npos) {
            max_tokens = std::stoi(req.substr(vs));
        }
    }

    // Last "content" string in the messages array
    size_t search = 0;
    while (true) {
        size_t pos = req.find("\"content\"", search);
        if (pos == std::string::npos) break;
        size_t qs = req.find('"', pos + 9);
        if (qs == std::string::npos) break;
        size_t qe = qs + 1;
        while (qe < req.size()) {
            if (req[qe] == '\\') { qe += 2; continue; }
            if (req[qe] == '"')  { break; }
            qe++;
        }
        content = req.substr(qs + 1, qe - qs - 1);
        search  = qe + 1;
    }

    // Build prompt via model's chat template
    llama_chat_message msgs[1] = { {"user", content.c_str()} };
    const char * tmpl = llama_model_chat_template(g_model, nullptr);
    std::vector<char> formatted(8192);
    int flen = llama_chat_apply_template(tmpl, msgs, 1, /*add_ass=*/true, formatted.data(), (int)formatted.size());
    if (flen > (int)formatted.size()) {
        formatted.resize(flen + 1);
        flen = llama_chat_apply_template(tmpl, msgs, 1, true, formatted.data(), flen);
    }
    if (flen < 0) flen = 0;
    std::string prompt(formatted.begin(), formatted.begin() + flen);

    // Tokenize
    int n_toks = -llama_tokenize(g_vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_toks);
    llama_tokenize(g_vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), n_toks, true, true);

    // Clear context state
    llama_memory_clear(llama_get_memory(g_ctx), /*data=*/true);

    // Greedy sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string response;
    auto t0 = std::chrono::steady_clock::now();
    int gen = 0;

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    while (gen < max_tokens) {
        if (llama_decode(g_ctx, batch) != 0) break;

        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;

        char piece[256];
        int n = llama_token_to_piece(g_vocab, tok, piece, sizeof(piece), 0, true);
        if (n > 0) response.append(piece, n);

        batch = llama_batch_get_one(&tok, 1);
        gen++;
    }

    llama_sampler_free(smpl);

    auto t1  = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tps = (ms > 0 && gen > 0) ? (gen * 1000.0 / ms) : 0.0;

    std::ostringstream ss;
    ss << "{";
    ss << "\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(response) << "\"}}],";
    ss << "\"_mtp_enabled\":" << (g_mtp_loaded ? "true" : "false") << ",";
    ss << "\"_spec_accept_rate\":null,";  // MTP threading requires SharedArrayBuffer (#739)
    ss << "\"_latency_ms\":" << (int64_t)ms << ",";
    ss << "\"_tps\":" << tps;
    ss << "}";

    return heap_str(ss.str());
}

EMSCRIPTEN_KEEPALIVE
void wasm_llama_free_str(char * ptr) {
    delete[] ptr;
}

}  // extern "C"
