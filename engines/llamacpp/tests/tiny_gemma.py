# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Deterministic untrained CPU fixture, exclusively for native protocol tests."""
from pathlib import Path
import numpy as np
import gguf


def write_tiny_gemma(path):
    width, heads, feed_forward, vocabulary = 32, 4, 64, 259
    writer = gguf.GGUFWriter(Path(path), 'gemma4')
    writer.add_name('FlashAlign untrained protocol fixture')
    writer.add_context_length(512)
    writer.add_embedding_length(width)
    writer.add_block_count(2)
    writer.add_feed_forward_length(feed_forward)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_key_length(width//heads)
    writer.add_value_length(width//heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_uint32('gemma4.attention.sliding_window', 128)
    writer.add_array('gemma4.attention.sliding_window_pattern', [True, False])
    writer.add_uint32('gemma4.embedding_length_per_layer_input', 0)
    writer.add_uint32('gemma4.attention.key_length_swa', width//heads)
    writer.add_uint32('gemma4.attention.value_length_swa', width//heads)
    writer.add_tokenizer_model('llama')
    writer.add_token_list(['<unk>', '<s>', '</s>', *[f'<0x{i:02X}>' for i in range(256)]])
    writer.add_token_scores([0.] * vocabulary)
    writer.add_token_types([2, 3, 3, *([6]*256)])
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    random = np.random.default_rng(8731)
    writer.add_tensor('token_embd.weight', random.normal(0, .02, (vocabulary, width)).astype(np.float32))
    writer.add_tensor('output_norm.weight', np.ones(width, dtype=np.float32))
    for layer in range(2):
        for name in ('attn_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm'):
            writer.add_tensor(f'blk.{layer}.{name}.weight', np.ones(width, dtype=np.float32))
        for name in ('attn_q_norm', 'attn_k_norm'):
            writer.add_tensor(f'blk.{layer}.{name}.weight', np.ones(width//heads, dtype=np.float32))
        for name in ('attn_q', 'attn_k', 'attn_v', 'attn_output'):
            writer.add_tensor(f'blk.{layer}.{name}.weight', random.normal(0, .02, (width, width)).astype(np.float32))
        for name in ('ffn_gate', 'ffn_up'):
            writer.add_tensor(f'blk.{layer}.{name}.weight', random.normal(0, .02, (feed_forward, width)).astype(np.float32))
        writer.add_tensor(f'blk.{layer}.ffn_down.weight', random.normal(0, .02, (width, feed_forward)).astype(np.float32))
    writer.add_tensor('rope_freqs.weight', np.ones(width//heads//2, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
