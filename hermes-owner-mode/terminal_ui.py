# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Portable FlashAlign terminal presentation; no deployment or model operations."""
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import select as selectors
import shutil
import signal
import sys
import time
import unicodedata


BG = (21, 22, 21)
IVORY = (255, 241, 221)
AMBER = (255, 184, 117)
CORAL = (245, 145, 135)
TEAL = (118, 220, 203)
MUTED = (151, 163, 153)
LINE = (62, 76, 66)
_RESET = '\x1b[0m'
_ESCAPES = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\|$)|[ -/]*[@-~])')
_FIELDS = {'model', 'profile', 'strength', 'access', 'mode', 'engine', 'harness', 'tools', 'status', 'context'}


def clean_text(value):
    """Remove terminal commands and control characters from displayed data."""
    value = _ESCAPES.sub('', str(value))
    return ''.join(c for c in value.expandtabs(4)
                   if c == '\n' or not unicodedata.category(c).startswith('C'))


def _single(value):
    return ' '.join(clean_text(value).splitlines())


def _width(value):
    return sum(0 if unicodedata.combining(c) else
               2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in value)


def _clip(value, width):
    value = _single(value)
    if width <= 0:
        return ''
    if _width(value) <= width:
        return value
    result, used = [], 0
    for char in value:
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + size > width - 1:
            break
        if size or result:
            result.append(char)
        used += size
    return ''.join(result) + '…'


def _pad(value, width):
    clipped = _clip(value, width)
    return clipped + ' ' * max(0, width - _width(clipped))


@lru_cache(maxsize=1)
def _logo():
    data = json.loads(Path(__file__).with_name('terminal-logo.json').read_text())
    w, h, pixels = data['width'], data['height'], data['pixels']
    if (type(w) is not int or type(h) is not int or not 1 <= w <= 96 or not 2 <= h <= 20
            or h % 2 or len(pixels) != h or any(len(row) != w for row in pixels)
            or any(not isinstance(rgb, list) or len(rgb) != 3
                   or any(type(c) is not int or not 0 <= c <= 255 for c in rgb)
                   for row in pixels for rgb in row)):
        raise ValueError('Invalid bundled terminal logo')
    return w, h, pixels


class TerminalUI:
    def __init__(self, output=None, color=True, animation=True):
        self.output = output or sys.stdout
        self.color = bool(color and self._tty(self.output) and 'NO_COLOR' not in os.environ
                          and os.environ.get('TERM') != 'dumb')
        self.animation = bool(animation)
        self._animated = False
        self._context = {}

    @staticmethod
    def _tty(stream):
        try:
            return stream.isatty()
        except (AttributeError, OSError):
            return False

    def _size(self):
        try:
            return os.get_terminal_size(self.output.fileno())
        except (AttributeError, OSError, ValueError):
            return shutil.get_terminal_size((80, 24))

    @property
    def interactive(self):
        size = self._size()
        return bool(self.color and self._tty(sys.stdin) and size.columns >= 60 and size.lines >= 16)

    @property
    def supports_picker(self):
        return self.interactive

    def ink(self, value, rgb):
        text = clean_text(value)
        if not self.color:
            return text
        if isinstance(rgb, str):
            parts = rgb.split(';')
            if len(parts) != 3 or any(not p.isdigit() or not 0 <= int(p) <= 255 for p in parts):
                raise ValueError('Invalid terminal color')
            rgb = tuple(map(int, parts))
        return '\x1b[38;2;' + ';'.join(map(str, rgb)) + 'm' + text + _RESET

    def line(self, text=''):
        print(clean_text(text), file=self.output)

    def context(self, **fields):
        for key, value in fields.items():
            if key not in _FIELDS:
                continue
            if value is None:
                self._context.pop(key, None)
            else:
                self._context[key] = _single(value)

    def _write(self, text):
        self.output.write(text)
        self.output.flush()

    @contextmanager
    def _cursor(self):
        handlers = {}
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    old = signal.signal(sig, interrupted)
                    handlers[sig] = old
                except ValueError:  # Rendering from a worker must not alter main-thread signals.
                    break
            self._write('\x1b[?25l')
            yield
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                self._write(_RESET + '\x1b[?25h')
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def _header(self, version, subtitle, width):
        logo_width, logo_height, pixels = _logo()
        margin = max(0, (width - logo_width) // 2)
        rows = []
        for y in range(0, logo_height, 2):
            parts = [' ' * margin]
            colors = None
            for x in range(logo_width):
                top, bottom = pixels[y][x], pixels[y + 1][x]
                current = (tuple(top), tuple(bottom))
                if current != colors:
                    parts.append('\x1b[38;2;' + ';'.join(map(str, top)) + ';48;2;' + ';'.join(map(str, bottom)) + 'm')
                    colors = current
                parts.append('▀')
            rows.append(''.join(parts) + _RESET)
        caption = _clip(f'{subtitle} {version}  /  AEVONIX RESEARCH', width)
        tagline = 'ReAlign models instantly, on demand.'
        return [self.ink('─' * width, LINE), *rows,
                self.ink(' ' * max(0, (width - _width(tagline)) // 2) + tagline, IVORY),
                self.ink(' ' * max(0, (width - _width(caption)) // 2) + caption, MUTED)]

    def banner(self, version, subtitle='Setup'):
        size = self._size()
        if not self.color or size.columns < 102 or size.lines < 20:
            self.line('FLASHALIGN / AEVONIX RESEARCH')
            self.line('ReAlign models instantly, on demand.')
            self.line(f'{_single(subtitle)} {_single(version)}  |  Direct engine / Hermes / OpenClaw')
            self.line()
            return
        width = min(108, size.columns - 2)
        try:
            header = self._header(version, subtitle, width)
        except (OSError, ValueError, KeyError, TypeError):
            self.line('FLASHALIGN / AEVONIX RESEARCH')
            self.line('ReAlign models instantly, on demand.')
            self.line(f'{_single(subtitle)} {_single(version)}')
            return
        self._write('\n'.join(header) + '\n')
        animate = (self.animation and not self._animated and self._tty(sys.stdin)
                   and not any(k in os.environ for k in ('NO_ANIM', 'CI', 'REDUCED_MOTION')))
        self._animated = True
        if animate:
            with self._cursor():
                start = time.monotonic()
                for index in range(15):
                    progress = (index + 1) / 15
                    colors = (AMBER, CORAL, TEAL)
                    cells = []
                    head = round(progress * width)
                    previous, run = None, ''
                    for x in range(width):
                        color = colors[min(2, x * 3 // width)] if head - 14 <= x < head else LINE
                        if previous is not None and previous != color:
                            cells.append(self.ink(run, previous))
                            run = ''
                        previous, run = color, run + '━'
                    if run:
                        cells.append(self.ink(run, previous))
                    self._write('\r' + ''.join(cells))
                    remaining = start + (index + 1) * .05 - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                self._write('\r' + self.ink('─' * width, LINE) + '\n')
        else:
            self._write(self.ink('─' * width, LINE) + '\n')
        self.line()

    def step(self, number, title):
        self.line()
        self._write(self.ink(f'[{number}/6] {_single(title)}', AMBER) + '\n')
        if self.color:
            self._write(self.ink('─' * min(108, max(1, self._size().columns - 2)), LINE) + '\n')

    def result(self, text, ok=True):
        self._write(self.ink('OK' if ok else 'ACTION NEEDED', TEAL if ok else CORAL) + '  ')
        self.line(text)

    def _context_fields(self):
        order = ('model', 'engine', 'profile', 'harness', 'access', 'strength', 'tools', 'status', 'context', 'mode')
        return [(key, self._context[key]) for key in order if key in self._context]

    def _choice_lines(self, options, active, width, sidebar=False):
        """Return fixed-width boxed choices and, where space permits, live context."""
        left_width = (width * 3 // 5) if sidebar else width
        left = []
        for index, (label, description) in enumerate(options):
            tone = TEAL if index == active else LINE
            selected = '●' if index == active else '○'
            left.append(self.ink('┌' + '─' * (left_width - 2) + '┐', tone))
            left.append(self.ink('│', tone)
                        + self.ink(_pad(f' {selected} {index + 1}. {label}', left_width - 2), TEAL if index == active else IVORY)
                        + self.ink('│', tone))
            left.append(self.ink('│', tone)
                        + self.ink(_pad('   ' + description, left_width - 2), MUTED)
                        + self.ink('│', tone))
            left.append(self.ink('└' + '─' * (left_width - 2) + '┘', tone))
        if not sidebar:
            return left
        right_width = width - left_width - 4
        right = [self.ink(_pad('CONFIGURATION', right_width), MUTED), ' ' * right_width]
        for key, value in self._context_fields()[:4]:
            right.append(self.ink(_pad(key.upper(), right_width), MUTED))
            right.append(self.ink(_pad(value, right_width), TEAL if key == 'profile' else IVORY))
        empty_left, empty_right = ' ' * left_width, ' ' * right_width
        return [(left[i] if i < len(left) else empty_left)
                + '  ' + self.ink('│', LINE) + ' '
                + (right[i] if i < len(right) else empty_right)
                for i in range(max(len(left), len(right)))]

    def select(self, prompt, options, default=0):
        if not options or not 0 <= default < len(options):
            raise ValueError('A selection needs choices and a valid default')
        options = [(_single(label), _single(description)) for label, description in options]
        if not self.interactive or len(options) * 4 + 4 > self._size().lines:
            for index, (label, description) in enumerate(options, 1):
                self.line(f'  {index}. {label}' + (f': {description}' if description else ''))
            while True:
                answer = input(_single(prompt)).strip()
                if not answer:
                    return default
                if answer.isdigit() and 1 <= int(answer) <= len(options):
                    return int(answer) - 1
                match = next((i for i, (label, _) in enumerate(options) if answer.casefold() == label.casefold()), None)
                if match is not None:
                    return match
                if answer.casefold() in ('q', 'quit', 'exit'):
                    raise KeyboardInterrupt
                self.line('Choose a listed number or label.')
        import termios
        import tty
        width = min(108, self._size().columns - 2)
        sidebar = width >= 98 and bool(self._context) and self._size().lines >= 18
        self._write(self.ink(_clip(prompt, width), IVORY) + '\n')
        if not sidebar:
            for key, value in self._context_fields()[:2]:
                self._write(self.ink(_pad(key.upper(), 10), MUTED)
                            + self.ink(_clip(value, width - 10), IVORY) + '\n')
        self._write(self.ink('─' * width, LINE) + '\n')
        self._write(self.ink(_clip('↑/↓ Move   Enter Select   Number Choose   Esc Cancel', width), MUTED) + '\n')
        fd, state = sys.stdin.fileno(), None
        current = default
        height = len(self._choice_lines(options, current, width, sidebar))
        self._write('\n' * height + f'\x1b[{height}A\r\x1b[s')
        try:
            state = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            with self._cursor():
                while True:
                    self._write('\x1b[u' + '\n'.join(self._choice_lines(options, current, width, sidebar)))
                    if self._size().columns < 60:
                        raise KeyboardInterrupt
                    key = os.read(fd, 1)
                    if key == b'\x1b':
                        if not selectors.select([fd], [], [], .08)[0]:
                            raise KeyboardInterrupt
                        rest = os.read(fd, 1)
                        if rest in (b'[', b'O') and selectors.select([fd], [], [], .08)[0]:
                            rest += os.read(fd, 1)
                        if rest in (b'[A', b'OA'):
                            current = (current - 1) % len(options)
                        elif rest in (b'[B', b'OB'):
                            current = (current + 1) % len(options)
                    elif key in (b'\r', b'\n'):
                        break
                    elif key in (b'q', b'Q', b'\x03', b'\x04', b''):
                        raise KeyboardInterrupt
                    elif key in b'123456789' and key:
                        choice = int(key) - 1
                        if choice < len(options):
                            current = choice
                            break
        finally:
            try:
                if state is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, state)
            finally:
                self._write('\x1b[u' + f'\x1b[{height}B\r' + _RESET + '\x1b[?25h')
        self.result(options[current][0])
        return current
