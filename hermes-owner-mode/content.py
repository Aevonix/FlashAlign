# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Content bindings made by trusted harness adapters, independent of transport."""
import hashlib
import json
from dataclasses import dataclass


def snapshot(content):
    """Copy a nonempty Chat Completions content value without mutable aliases."""
    if isinstance(content, str):
        if not content:
            raise ValueError("Empty rendered message")
    elif isinstance(content, list) and content:
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get('type'), str):
                raise ValueError("Invalid content part")
            kind = part['type']
            if kind == 'text':
                if set(part) != {'type', 'text'} or not isinstance(part['text'], str):
                    raise ValueError("Invalid text part")
            elif kind == 'image_url':
                value = part.get('image_url')
                if (set(part) != {'type', 'image_url'} or not isinstance(value, dict)
                        or set(value) - {'url', 'detail'} or not isinstance(value.get('url'), str)
                        or not value['url']):
                    raise ValueError("Invalid image part")
            elif kind == 'input_audio':
                value = part.get('input_audio')
                if (set(part) != {'type', 'input_audio'} or not isinstance(value, dict)
                        or set(value) != {'data', 'format'}
                        or not all(isinstance(value[k], str) and value[k] for k in value)):
                    raise ValueError("Invalid audio part")
            elif kind in {'file', 'video_url'}:
                value = part.get(kind)
                if set(part) != {'type', kind} or not isinstance(value, dict) or not value:
                    raise ValueError("Invalid attachment part")
            else:
                raise ValueError("Unsupported content part")
    else:
        raise ValueError("Invalid rendered message")
    return json.loads(json.dumps(content, ensure_ascii=False, allow_nan=False))


def content_digest(content):
    return hashlib.sha256(json.dumps(snapshot(content), ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ContentBinding:
    own_text: str = ''
    content_sha256: str = ''
    content_parts: int = 0


def bind_content(content):
    """Keep media hashes, not copies of attachment payloads, in the receipt ledger."""
    value = snapshot(content)
    if isinstance(value, str):
        return ContentBinding(own_text=value)
    return ContentBinding(content_sha256=content_digest(value), content_parts=len(value))


def text_parts(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and set(p) == {'type', 'text'}
                                       and p['type'] == 'text' and isinstance(p['text'], str) for p in content):
        return '\n\n'.join(p['text'] for p in content)
    return None


def matches(content, expected, *, allow_text_context=False):
    """Additional host context may be text; bound media cannot be changed."""
    try:
        actual, wanted = snapshot(content), snapshot(expected)
        if actual == wanted:
            return True
        if isinstance(wanted, str):
            text = text_parts(actual)
            return bool(text is not None and (text == wanted or allow_text_context
                        and '\n\n'+wanted+'\n\n' in '\n\n'+text+'\n\n'))
        return bool(allow_text_context and isinstance(actual, list) and len(actual) > len(wanted)
                    and actual[:len(wanted)] == wanted and text_parts(actual[len(wanted):]) is not None)
    except (ValueError, TypeError):
        return False


def proof_matches(content, proof, *, allow_text_context=False):
    if proof.content_sha256:
        try:
            if content_digest(content) == proof.content_sha256:
                return True
            return bool(allow_text_context and isinstance(content, list) and proof.content_parts > 0
                        and len(content) > proof.content_parts
                        and content_digest(content[:proof.content_parts]) == proof.content_sha256
                        and text_parts(content[proof.content_parts:]) is not None)
        except (ValueError, TypeError):
            return False
    return matches(content, proof.own_text, allow_text_context=allow_text_context)
