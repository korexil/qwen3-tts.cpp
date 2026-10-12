#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "env_flag.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace qwen3_tts {

bool transformer_internal::ops::create_tensors(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = impl->model.config;

#ifdef QWEN3_TTS_CUDA_ENABLED
    const bool create_packed_qkv = true;
#else
    const bool create_packed_qkv = false;
#endif
    const int32_t packed_code_pred_layers =
        create_packed_qkv && !impl->skip_ggml_code_pred_layers ? cfg.code_pred_layers : 0;
    const int32_t packed_talker_layers =
        create_packed_qkv && cfg.hidden_size > 1024 ? cfg.n_layers : 0;
    const size_t ctx_size =
        (n_tensors + packed_code_pred_layers + packed_talker_layers) * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl->model.ctx = ggml_init(params);
    if (!impl->model.ctx) {
        error_msg = "Failed to create GGML context";
        return false;
    }

    const bool text_embd_cpu = env_flag_on("QWEN3_TTS_TEXT_EMBD_CPU");
    const bool qkv_dedup = env_flag_on("QWEN3_TTS_QKV_DEDUP") &&
        (packed_talker_layers > 0 || packed_code_pred_layers > 0);
    if (text_embd_cpu) {
        struct ggml_init_params p_cpu = { ggml_tensor_overhead() * 2, nullptr, true };
        impl->model.ctx_cpu = ggml_init(p_cpu);
        if (!impl->model.ctx_cpu) { error_msg = "Failed to create CPU text_embd context"; return false; }
    }
    if (qkv_dedup) {
        struct ggml_init_params p_meta = {
            ggml_tensor_overhead() * (size_t) (3 * (cfg.n_layers + cfg.code_pred_layers) + 4), nullptr, true };
        impl->model.ctx_meta = ggml_init(p_meta);
        if (!impl->model.ctx_meta) { error_msg = "Failed to create Q/K/V meta context"; return false; }
    }
    fprintf(stderr, "  text_embd_cpu=%d qkv_dedup=%d\n", text_embd_cpu ? 1 : 0, qkv_dedup ? 1 : 0);

    impl->model.layers.resize(cfg.n_layers);
    impl->model.code_pred_layers.resize(cfg.code_pred_layers);
    impl->model.code_pred_embd.resize(cfg.n_codebooks - 1);
    impl->model.code_pred_head.resize(cfg.n_codebooks - 1);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;

        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }

        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight") || strstr(name, "code_pred.mtp_proj.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.code_pred_hidden_size;
            n_dims = 2;
        } else if (strstr(name, "code_pred.small_to_mtp.bias") || strstr(name, "code_pred.mtp_proj.bias")) {
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "code_pred.blk.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.code_pred_intermediate_size;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int cb_idx = -1;
            if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.code_pred_hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else {
            continue;
        }

        struct ggml_context * target_ctx = impl->model.ctx;
        if (impl->model.ctx_cpu && strstr(name, "talker.text_embd.weight")) {
            target_ctx = impl->model.ctx_cpu;
        } else if (impl->model.ctx_meta &&
                   (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") ||
                    strstr(name, "attn_v.weight")) &&
                   ((packed_talker_layers > 0 && strstr(name, "talker.blk.")) ||
                    (packed_code_pred_layers > 0 && strstr(name, "code_pred.blk.")))) {
            target_ctx = impl->model.ctx_meta;
        }
        struct ggml_tensor * tensor = ggml_new_tensor(target_ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        impl->model.tensors[name] = tensor;

        if (strstr(name, "talker.text_embd.weight")) {
            impl->model.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            impl->model.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            impl->model.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            impl->model.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            impl->model.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            impl->model.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            impl->model.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            impl->model.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = impl->model.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight") || strstr(name, "code_pred.mtp_proj.weight")) {
            impl->model.code_pred_small_to_mtp_weight = tensor;
        } else if (strstr(name, "code_pred.small_to_mtp.bias") || strstr(name, "code_pred.mtp_proj.bias")) {
            impl->model.code_pred_small_to_mtp_bias = tensor;
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = impl->model.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_embd[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_head[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            impl->model.code_pred_output_norm = tensor;
        }
    }

    if (create_packed_qkv && !impl->skip_ggml_code_pred_layers) {
        for (int32_t il = 0; il < cfg.code_pred_layers; ++il) {
            auto & layer = impl->model.code_pred_layers[(size_t) il];
            if (!layer.attn_q || !layer.attn_k || !layer.attn_v) {
                error_msg = "Missing Code Predictor Q/K/V weights for packed layer " +
                    std::to_string(il);
                return false;
            }
            if (layer.attn_q->type != layer.attn_k->type ||
                layer.attn_q->type != layer.attn_v->type ||
                layer.attn_q->ne[0] != layer.attn_k->ne[0] ||
                layer.attn_q->ne[0] != layer.attn_v->ne[0]) {
                error_msg = "Incompatible Code Predictor Q/K/V weights for packed layer " +
                    std::to_string(il);
                return false;
            }

            layer.attn_qkv = ggml_new_tensor_2d(
                impl->model.ctx,
                layer.attn_q->type,
                layer.attn_q->ne[0],
                layer.attn_q->ne[1] + layer.attn_k->ne[1] + layer.attn_v->ne[1]);
            if (!layer.attn_qkv) {
                error_msg = "Failed to create packed Code Predictor QKV tensor for layer " +
                    std::to_string(il);
                return false;
            }
            ggml_format_name(layer.attn_qkv,
                             "code_pred.blk.%d.attn_qkv.packed", il);
            if (impl->model.ctx_meta) {
                struct ggml_tensor * q = layer.attn_q;
                struct ggml_tensor * k = layer.attn_k;
                struct ggml_tensor * v = layer.attn_v;
                const size_t nb1 = layer.attn_qkv->nb[1];
                layer.attn_q = ggml_view_2d(impl->model.ctx, layer.attn_qkv, q->ne[0], q->ne[1], nb1, 0);
                layer.attn_k = ggml_view_2d(impl->model.ctx, layer.attn_qkv, k->ne[0], k->ne[1], nb1,
                                            (size_t) q->ne[1] * nb1);
                layer.attn_v = ggml_view_2d(impl->model.ctx, layer.attn_qkv, v->ne[0], v->ne[1], nb1,
                                            (size_t) (q->ne[1] + k->ne[1]) * nb1);
                if (!layer.attn_q || !layer.attn_k || !layer.attn_v) {
                    error_msg = "Failed to create Q/K/V views for layer " + std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.attn_q, "%s.view", ggml_get_name(q));
                ggml_format_name(layer.attn_k, "%s.view", ggml_get_name(k));
                ggml_format_name(layer.attn_v, "%s.view", ggml_get_name(v));
            }
        }
    }

    if (create_packed_qkv && cfg.hidden_size > 1024) {
        for (int32_t il = 0; il < cfg.n_layers; ++il) {
            auto & layer = impl->model.layers[(size_t) il];
            if (!layer.attn_q || !layer.attn_k || !layer.attn_v) {
                error_msg = "Missing Talker Q/K/V weights for packed layer " +
                    std::to_string(il);
                return false;
            }
            if (layer.attn_q->type != layer.attn_k->type ||
                layer.attn_q->type != layer.attn_v->type ||
                layer.attn_q->ne[0] != layer.attn_k->ne[0] ||
                layer.attn_q->ne[0] != layer.attn_v->ne[0]) {
                error_msg = "Incompatible Talker Q/K/V weights for packed layer " +
                    std::to_string(il);
                return false;
            }

            layer.attn_qkv = ggml_new_tensor_2d(
                impl->model.ctx,
                layer.attn_q->type,
                layer.attn_q->ne[0],
                layer.attn_q->ne[1] + layer.attn_k->ne[1] + layer.attn_v->ne[1]);
            if (!layer.attn_qkv) {
                error_msg = "Failed to create packed Talker QKV tensor for layer " +
                    std::to_string(il);
                return false;
            }
            ggml_format_name(layer.attn_qkv, "talker.blk.%d.attn_qkv.packed", il);
            if (impl->model.ctx_meta) {
                struct ggml_tensor * q = layer.attn_q;
                struct ggml_tensor * k = layer.attn_k;
                struct ggml_tensor * v = layer.attn_v;
                const size_t nb1 = layer.attn_qkv->nb[1];
                layer.attn_q = ggml_view_2d(impl->model.ctx, layer.attn_qkv, q->ne[0], q->ne[1], nb1, 0);
                layer.attn_k = ggml_view_2d(impl->model.ctx, layer.attn_qkv, k->ne[0], k->ne[1], nb1,
                                            (size_t) q->ne[1] * nb1);
                layer.attn_v = ggml_view_2d(impl->model.ctx, layer.attn_qkv, v->ne[0], v->ne[1], nb1,
                                            (size_t) (q->ne[1] + k->ne[1]) * nb1);
                if (!layer.attn_q || !layer.attn_k || !layer.attn_v) {
                    error_msg = "Failed to create Q/K/V views for layer " + std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.attn_q, "%s.view", ggml_get_name(q));
                ggml_format_name(layer.attn_k, "%s.view", ggml_get_name(k));
                ggml_format_name(layer.attn_v, "%s.view", ggml_get_name(v));
            }
        }
    }

    return true;
}

bool transformer_internal::ops::load_tensor_data(TTSTransformer & self, const std::string & path, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    ggml_backend_t backend = init_preferred_backend("TTSTransformer", &error_msg);
    if (!backend) {
        return false;
    }

    impl->model.buffer = ggml_backend_alloc_ctx_tensors(impl->model.ctx, backend);
    if (!impl->model.buffer) {
        error_msg = "Failed to allocate tensor buffer";
        release_preferred_backend(backend);
        return false;
    }
    if (impl->model.ctx_cpu) {
        const bool map_text_embd = env_flag_on("QWEN3_TTS_TEXT_EMBD_MMAP");
        if (map_text_embd) {
            const int64_t index = gguf_find_tensor(ctx, "talker.text_embd.weight");
            struct ggml_tensor * tensor = impl->model.text_embd;
            if (index < 0 || !tensor || ggml_get_first_tensor(impl->model.ctx_cpu) != tensor ||
                    ggml_get_next_tensor(impl->model.ctx_cpu, tensor) != nullptr) {
                error_msg = "Mapped CPU context must contain exactly the text embedding tensor";
                release_preferred_backend(backend); return false;
            }
            const uint64_t data_offset = gguf_get_data_offset(ctx);
            const uint64_t tensor_offset = gguf_get_tensor_offset(ctx, index);
            if (tensor_offset > UINT64_MAX - data_offset) {
                error_msg = "Text embedding offset overflow";
                release_preferred_backend(backend); return false;
            }
            impl->model.text_embd_mapping = ReadonlyTensorMapping::open(
                path, data_offset + tensor_offset, ggml_nbytes(tensor), error_msg);
            if (!impl->model.text_embd_mapping) { release_preferred_backend(backend); return false; }
            void * data = impl->model.text_embd_mapping->data();
            const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
            if (reinterpret_cast<uintptr_t>(data) % alignment != 0) {
                error_msg = "Mapped text embedding is not GGML aligned";
                release_preferred_backend(backend); return false;
            }
            impl->model.buffer_cpu = ggml_backend_cpu_buffer_from_ptr(data, ggml_nbytes(tensor));
            if (!impl->model.buffer_cpu ||
                    ggml_backend_tensor_alloc(impl->model.buffer_cpu, tensor, data) != GGML_STATUS_SUCCESS) {
                error_msg = "Failed to wrap read-only text embedding";
                release_preferred_backend(backend); return false;
            }
            fprintf(stderr, "  read-only mapped text_embd=%zu bytes (no heap copy)\n", ggml_nbytes(tensor));
        } else {
            impl->model.buffer_cpu =
                ggml_backend_alloc_ctx_tensors_from_buft(impl->model.ctx_cpu, ggml_backend_cpu_buffer_type());
        }
        if (!impl->model.buffer_cpu) {
            error_msg = "Failed to allocate CPU text_embd buffer";
            release_preferred_backend(backend);
            return false;
        }
        fprintf(stderr, "  model buffer: device=%zu bytes, host text_embd=%zu bytes\n",
                ggml_backend_buffer_get_size(impl->model.buffer),
                ggml_backend_buffer_get_size(impl->model.buffer_cpu));
    } else {
        fprintf(stderr, "  model buffer: device=%zu bytes\n",
                ggml_backend_buffer_get_size(impl->model.buffer));
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        release_preferred_backend(backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = impl->model.tensors.find(name);
        if (it == impl->model.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        // The mapped input is immutable; reading it into read_buf and calling
        // tensor_set would both duplicate the table and write to read-only pages.
        if (impl->model.text_embd_mapping && tensor == impl->model.text_embd) {
            continue;
        }

        read_buf.resize(nbytes);

#ifdef _WIN32
        if (_fseeki64(f, (int64_t) data_offset + (int64_t) offset, SEEK_SET) != 0) {
#else
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (tensor->buffer) {  // metadata-only Q/K/V (QKV_DEDUP) have no buffer; packed copy below
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
        }

        int layer_idx = -1;
        if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
            layer_idx >= 0 && layer_idx < impl->model.config.code_pred_layers) {
            auto & layer = impl->model.code_pred_layers[(size_t) layer_idx];
            if (layer.attn_qkv && strstr(name, "attn_q.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(), 0, nbytes);
            } else if (layer.attn_qkv && strstr(name, "attn_k.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(),
                                        ggml_nbytes(layer.attn_q), nbytes);
            } else if (layer.attn_qkv && strstr(name, "attn_v.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(),
                                        ggml_nbytes(layer.attn_q) + ggml_nbytes(layer.attn_k),
                                        nbytes);
            }
        }

        layer_idx = -1;
        if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
            layer_idx >= 0 && layer_idx < impl->model.config.n_layers) {
            auto & layer = impl->model.layers[(size_t) layer_idx];
            if (layer.attn_qkv && strstr(name, "attn_q.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(), 0, nbytes);
            } else if (layer.attn_qkv && strstr(name, "attn_k.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(),
                                        ggml_nbytes(layer.attn_q), nbytes);
            } else if (layer.attn_qkv && strstr(name, "attn_v.weight")) {
                ggml_backend_tensor_set(layer.attn_qkv, read_buf.data(),
                                        ggml_nbytes(layer.attn_q) + ggml_nbytes(layer.attn_k),
                                        nbytes);
            }
        }
    }

    fclose(f);
    release_preferred_backend(backend);

    return true;
}

} // namespace qwen3_tts
