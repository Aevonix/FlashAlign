# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Shared, CPU-only contract for native DeepSeek grouped profiles.

The copies in the setup package and engine package must match this source.
This validates supplied operators; it does not derive profile vectors.
"""
import math
import re

DEEPSEEK_GROUP_SCHEMA = 'flashalign.grouped_projection.v1'
DEEPSEEK_MODEL = {'id': 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
                  'revision': '1bae0e1088993b9b82de08f59de5be5a34e42170'}


def _positive_int(value):
    return type(value) is int and value > 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_grouped_profile(profile, hidden, sites, *, error=ValueError):
    """Validate a supplied native writer-group artifact, without deriving one."""
    if not isinstance(profile, dict) or hidden != 5120 or profile.get('model') != DEEPSEEK_MODEL:
        raise error('Grouped DeepSeek profiles require the pinned model and width')
    if (profile.get('schema') != DEEPSEEK_GROUP_SCHEMA
            or profile.get('transform', 'projection') != 'projection'
            or not isinstance(sites, list) or not sites or len(sites) > 4096):
        raise error('Invalid grouped profile schema, transform or sites')
    declared = profile.get('requires', [])
    if (not isinstance(declared, list) or any(not isinstance(item, str) for item in declared)
            or len(declared) != len(set(declared))):
        raise error('Invalid grouped profile requirements')
    operators = profile.get('operators')
    if not isinstance(operators, dict) or not operators or len(operators) > 4096:
        raise error('Grouped profiles must declare their operators')
    tensors = {}
    for name, operator in operators.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', name)
                or not isinstance(operator, dict) or set(operator) != {'basis', 'rank', 'scale'}):
            raise error('Invalid grouped profile operator')
        basis, rank = operator['basis'], operator['rank']
        if (not isinstance(basis, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,256}', basis)
                or not _positive_int(rank) or rank > 128 or not _number(operator['scale'])):
            raise error('Invalid grouped profile basis, rank or scale')
        if basis in tensors and tensors[basis] != [rank, hidden]:
            raise error('Inconsistent shared basis shape')
        tensors[basis] = [rank, hidden]
    required = {'grouped_basis', 'selected_tp_rows'}
    extensions = {'moe_out': {'combined_moe_output'}, 'shared_expert_out': {'shared_expert_output'},
                  'engram_value_out': {'engram_value_slice'},
                  'vision_aligner_out': {'vision_pre_bias', 'typed_image_span'},
                  'image_marker_out': {'typed_image_span'}}
    seen, used = set(), set()
    for row in sites:
        if not isinstance(row, dict):
            raise error('Invalid grouped profile site')
        layer, site, operator = row.get('layer'), row.get('site'), row.get('operator')
        if (type(layer) is not int or not 0 <= layer < 40 or not isinstance(site, str)
                or site not in {'attn_out', 'embedding_out', *extensions}
                or not isinstance(operator, str) or operator not in operators):
            raise error('Invalid grouped profile site or operator reference')
        if (layer, site) in seen:
            raise error('Duplicate grouped profile site')
        seen.add((layer, site)); used.add(operator)
        required.update(extensions.get(site, set()))
        stem = f'layers.{layer}'
        if site == 'attn_out':
            valid = row.get('tensor') == stem+'.attn.wo_b.weight'
        elif site == 'shared_expert_out':
            valid = row.get('tensor') == stem+'.ffn.shared_experts.w2.weight'
        elif site == 'moe_out':
            valid = (row.get('tensor_pattern') == stem+'.ffn.*.w2.weight'
                     and row.get('components') == ['routed_experts', 'shared_expert']
                     and row.get('application') == 'combined_moe_output_before_mhc_residual')
        elif site == 'engram_value_out':
            valid = (layer in (1, 14) and row.get('tensor') == stem+'.engram.wkv.weight'
                     and row.get('output_slice') == [20480, 25600])
        elif site == 'embedding_out':
            valid = layer == 0 and row.get('tensor') == 'embed.weight'
        elif site == 'vision_aligner_out':
            valid = (layer == 0 and row.get('tensor') == 'aligner.w2.weight'
                     and row.get('bias_tensor') == 'aligner.w2.bias'
                     and row.get('application') == 'before_unchanged_bias')
        else:
            valid = (layer == 0 and row.get('tensors') == ['image_start', 'image_end', 'image_newline']
                     and row.get('application') == 'typed_image_span_delimiters_without_aligner_bias')
        if not valid:
            raise error('Grouped profile target does not match its declared native output')
    if used != set(operators):
        raise error('Grouped profile contains unused operators')
    if set(profile.get('requires', [])) != required:
        raise error('Grouped profile requirements do not match its sites')
    return tensors
