# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Observe Hermes rendering and queue composition without changing their results."""
from collections import OrderedDict
from copy import copy
from functools import wraps
import inspect
import importlib
import sys
from types import SimpleNamespace
import weakref


def event_copy(event):
    return SimpleNamespace(source=copy(getattr(event, 'source', None)),
        **{key: getattr(event, key, None) for key in (
        'text', 'message_type', 'internal', 'message_id', 'reply_to_message_id', 'reply_to_text')},
        media_urls=list(getattr(event, 'media_urls', None) or []),
        metadata={key:value for key,value in (getattr(event, 'metadata', None) or {}).items()
                  if key in {'whatsapp_native_type','whatsapp_from_owner','forwarded','reaction',
                             'hermes_plugin_injection'}})


class _MergeHub:
    def __init__(self, module):
        self.module = module
        self.original = module.merge_pending_message_event
        self.listeners = weakref.WeakSet()

        @wraps(self.original)
        def merge(pending, key, event, *args, **kwargs):
            previous = pending.get(key)
            captured = []
            for observer in list(self.listeners):
                try:
                    captured.append((observer, observer.parts(previous) + observer.parts(event)))
                except Exception:
                    captured.append((observer, ()))
            result = self.original(pending, key, event, *args, **kwargs)
            current = pending.get(key)
            if previous is not None and current is previous and current is not event:
                for observer, parts in captured:
                    observer.remember(current, parts)
            return result

        self.wrapper = merge
        module.merge_pending_message_event = merge


class _NativeRenderHub:
    """Observe the final host-built message, after local images become data URLs."""
    def __init__(self, runner_type):
        self.runner_type = runner_type
        self.original = runner_type._native_image_run_message
        self.original_run = runner_type._run_conversation_with_approval
        self.listeners = weakref.WeakSet()

        @wraps(self.original)
        def render(turn, *args, **kwargs):
            result = self.original(turn, *args, **kwargs)
            for observer in list(self.listeners):
                observer.observe_native_render(turn, result)
            return result

        self.wrapper = render
        runner_type._native_image_run_message = render

        @wraps(self.original_run)
        def run(turn, *args, **kwargs):
            scopes = []
            for observer in list(self.listeners):
                authority = observer.authority()
                if authority is not None:
                    scopes.append((authority, authority.render_scope.set((True, None))))
            try:
                return self.original_run(turn, *args, **kwargs)
            finally:
                for authority, token in reversed(scopes):
                    authority.render_scope.reset(token)

        self.run_wrapper = run
        runner_type._run_conversation_with_approval = run


class HermesObserver:
    """The event object and host rendering seam supply provenance, never prompt text."""
    def __init__(self, authority):
        self.authority = weakref.ref(authority)
        self.groups = OrderedDict()
        self.renderers = []
        self.hub = None
        self.native_hub = None
        self.events = OrderedDict()
        self.attach_merge()

    def attach_merge(self):
        module = sys.modules.get('gateway.platforms.base')
        if module is None:
            try:
                module = importlib.import_module('gateway.platforms.base')
            except ImportError:
                return
        if module is None or not callable(getattr(module, 'merge_pending_message_event', None)):
            return
        hub = getattr(module, '_aevonix_owner_merge_observer', None)
        if hub is None or module.merge_pending_message_event is not hub.wrapper:
            hub = _MergeHub(module)
            module._aevonix_owner_merge_observer = hub
        hub.listeners.add(self)
        self.hub = hub

    def parts(self, event):
        if event is None:
            return ()
        saved = self.groups.get(id(event))
        return saved[1] if saved and saved[0]() is event else (event_copy(event),)

    def remember(self, event, parts):
        # Oversized or untraceable batches remain ineligible instead of dropping members.
        try:
            reference = weakref.ref(event)
        except TypeError:
            reference = lambda: event
        self.groups[id(event)] = (reference, parts if 0 < len(parts) <= 64 else ())
        while len(self.groups) > 4096:
            self.groups.popitem(last=False)

    def attach(self, gateway):
        self.attach_merge()
        self.attach_native_render()
        if gateway is None or any(row[0] is gateway for row in self.renderers):
            return
        for name, first_item in [('_prepare_profile_scoped_inbound_message_text', False),
                                 ('_hmwa_apply_message_timestamp', True)]:
            original = getattr(gateway, name, None)
            if not callable(original):
                continue
            try:
                signature = inspect.signature(original)
            except (ValueError, TypeError):
                continue
            existed = name in getattr(gateway, '__dict__', {})
            render = self._renderer(gateway, original, signature, first_item)
            setattr(gateway, name, render)
            self.renderers.append((gateway, name, original, render, existed))

    def attach_native_render(self):
        # Hermes can import TurnRunner only after ingress admission. Import its
        # native renderer here so the first image turn is observed as well.
        try:
            module = importlib.import_module('gateway.run_turn_runner')
        except ImportError:
            return
        runner_type = getattr(module, 'TurnRunner', None)
        if not all(callable(getattr(runner_type, name, None)) for name in
                   ('_native_image_run_message', '_run_conversation_with_approval')):
            return
        hub = getattr(runner_type, '_flashalign_native_render_observer', None)
        if hub is None or runner_type._native_image_run_message is not hub.wrapper:
            hub = _NativeRenderHub(runner_type)
            runner_type._flashalign_native_render_observer = hub
        hub.listeners.add(self)
        self.native_hub = hub

    @staticmethod
    def _native_key(gateway, source, message_id):
        return (id(gateway), str(getattr(getattr(source, 'platform', ''), 'value',
                                         getattr(source, 'platform', ''))),
                str(getattr(source, 'account_id', '') or 'default'),
                str(getattr(source, 'user_id', '')), str(getattr(source, 'chat_id', '')),
                str(message_id or ''))

    def remember_ingress(self, gateway, event):
        source = getattr(event, 'source', None)
        message_id = getattr(event, 'message_id', None) or getattr(source, 'message_id', None)
        if not message_id:
            return
        self.events[self._native_key(gateway, source, message_id)] = event_copy(event)
        while len(self.events) > 4096:
            self.events.popitem(last=False)

    def observe_native_render(self, turn, content):
        gateway, context = getattr(turn, '_runner', None), getattr(turn, '_ctx', None)
        if not any(row[0] is gateway for row in self.renderers):
            return
        # The raw inbound id is required; the reply anchor can name another message.
        key = self._native_key(gateway, getattr(context, 'source', None),
                               getattr(context, 'inbound_message_id', None))
        event = self.events.get(key)
        authority = self.authority()
        if authority is not None:
            active, _ = authority.render_scope.get()
            authority.render_scope.set((active, None))
        if event is None or not event.media_urls or authority is None:
            return
        try:
            authority.record_native_render(event=event, gateway=gateway, content=content)
        except Exception:
            authority.reject_render(event)

    def _renderer(self, gateway, original, signature, first_item):
        @wraps(original)
        def render(*args, **kwargs):
            try:
                event = signature.bind_partial(*args, **kwargs).arguments.get('event')
            except TypeError:
                event = None
            result = original(*args, **kwargs)

            def observed(value):
                authority = self.authority()
                if authority is not None and event is not None:
                    try:
                        content = value[0] if first_item else value
                        # A captionless image is still empty at this earlier seam.
                        # Only the later native renderer can bind it; absence of
                        # that final observation remains ineligible.
                        if content == '' and getattr(event, 'media_urls', None) and self.native_hub:
                            return value
                        authority.record_render(event=event, gateway=gateway, content=content)
                    except Exception:
                        authority.reject_render(event)
                return value

            if inspect.isawaitable(result):
                async def wait():
                    return observed(await result)
                return wait()
            return observed(result)

        return render

    def close(self):
        for gateway, name, original, wrapper, existed in self.renderers:
            if getattr(gateway, name, None) is wrapper:
                if existed:
                    setattr(gateway, name, original)
                else:
                    delattr(gateway, name)
        self.renderers.clear()
        if self.hub:
            self.hub.listeners.discard(self)
            if not self.hub.listeners and self.hub.module.merge_pending_message_event is self.hub.wrapper:
                self.hub.module.merge_pending_message_event = self.hub.original
                delattr(self.hub.module, '_aevonix_owner_merge_observer')
        self.groups.clear()
        self.events.clear()
        if self.native_hub:
            hub = self.native_hub
            hub.listeners.discard(self)
            if not hub.listeners and hub.runner_type._native_image_run_message is hub.wrapper:
                hub.runner_type._native_image_run_message = hub.original
                if hub.runner_type._run_conversation_with_approval is hub.run_wrapper:
                    hub.runner_type._run_conversation_with_approval = hub.original_run
                delattr(hub.runner_type, '_flashalign_native_render_observer')
