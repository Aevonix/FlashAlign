# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Terminal presentation does not change decisions or trust display data."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if 'owner_mode' not in sys.modules:
    spec = importlib.util.spec_from_file_location('owner_mode', ROOT/'__init__.py',
                                                 submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['owner_mode'] = package
    spec.loader.exec_module(package)
from owner_mode.terminal_ui import TerminalUI, clean_text, _clip, _width


class TTY(io.StringIO):
    def isatty(self):
        return True


class Presentation(unittest.TestCase):
    def test_redirected_banner_stays_plain_and_does_not_read_artwork(self):
        output = io.StringIO()
        with mock.patch('owner_mode.terminal_ui._logo', side_effect=AssertionError('not needed')):
            TerminalUI(output=output).banner('1.0.0')
        self.assertNotIn('\x1b', output.getvalue())
        self.assertIn('FLASHALIGN / AEVONIX RESEARCH', output.getvalue())
        self.assertIn('ReAlign models instantly, on demand.', output.getvalue())

    def test_empty_no_color_is_an_opt_out(self):
        output = TTY()
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color','NO_COLOR':''}, clear=True):
            ui = TerminalUI(output=output)
            ui.banner('1.0.0')
            ui.result('Done')
        self.assertFalse(ui.color)
        self.assertNotIn('\x1b', output.getvalue())

    def test_narrow_terminal_uses_readable_header(self):
        output = TTY()
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color'}, clear=True):
            ui = TerminalUI(output=output)
            with mock.patch.object(ui, '_size', return_value=os.terminal_size((70,24))):
                ui.banner('1.0.0')
        self.assertIn('FLASHALIGN / AEVONIX RESEARCH', output.getvalue())
        self.assertNotIn('▀', output.getvalue())

    def test_graphic_header_uses_sharp_bundled_art(self):
        output = TTY()
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color'}, clear=True):
            ui = TerminalUI(output=output, animation=False)
            with mock.patch.object(ui, '_size', return_value=os.terminal_size((112,42))):
                ui.banner('1.0.0')
        self.assertIn('▀', output.getvalue())
        self.assertIn('AEVONIX RESEARCH', output.getvalue())
        self.assertNotIn('SAMPLE', output.getvalue())
        self.assertNotIn('Qwen', output.getvalue())

    def test_ansi_osc_bidi_and_controls_are_removed_but_lines_are_preserved(self):
        value = 'first\x1b[31mRED\x1b[0m\x1b]8;;https://invalid\x07link\x1b]8;;\x1b\\\n\tlast\u202e\x00'
        self.assertEqual(clean_text(value), 'firstREDlink\n    last')

    def test_clipping_counts_wide_and_combining_characters(self):
        for width in range(1,9):
            self.assertLessEqual(_width(_clip('界界e\u0301\x1b[31mverylong',width)),width)
        self.assertNotIn('\x1b',_clip('a\x1b[2Jb',20))

    def test_context_accepts_display_fields_and_ignores_credentials(self):
        ui = TerminalUI(output=io.StringIO())
        ui.context(model='publisher/model',profile='profile\x1b[2J',strength=1,
                   api_key='SECRET',contact='PRIVATE')
        self.assertEqual(ui._context, {'model':'publisher/model','profile':'profile','strength':'1'})
        ui.context(model=None)
        self.assertNotIn('model',ui._context)

    def test_plain_selection_accepts_numbers_labels_and_default(self):
        options=[('Verified contacts','Approved identities'),('Always On','Every routed request')]
        for answer, expected in [('2',1),('always on',1),('',0)]:
            with self.subTest(answer=answer),mock.patch('builtins.input',return_value=answer):
                self.assertEqual(TerminalUI(output=io.StringIO()).select('Access: ',options),expected)

    def test_plain_invalid_selection_reprompts_and_cancel_raises(self):
        ui=TerminalUI(output=io.StringIO())
        with mock.patch('builtins.input',side_effect=['unknown','2']):
            self.assertEqual(ui.select('Access: ',[('One',''),('Two','')]),1)
        with mock.patch('builtins.input',return_value='q'),self.assertRaises(KeyboardInterrupt):
            ui.select('Access: ',[('One','')])

    def test_boxed_choices_fit_their_context_sidebar(self):
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color'},clear=True):
            ui=TerminalUI(output=TTY(),animation=False)
            ui.context(model='界'*80,profile='owner\x1b[2J',engine='TensorFold',harness='Hermes')
            rows=ui._choice_lines([('Verified contacts','Approved identities'),('Always On','Every routed request')],0,108,True)
            self.assertIn('┌',clean_text(rows[0]))
            self.assertTrue(any('CONFIGURATION' in clean_text(row) for row in rows))
            self.assertTrue(all(_width(clean_text(row))==108 for row in rows))
            self.assertFalse(any('\x1b[2J' in row for row in rows))

    def test_opt_out_disables_animation_even_when_set_empty(self):
        for name in ('NO_ANIM','CI','REDUCED_MOTION'):
            with self.subTest(name=name),mock.patch.dict(os.environ, {'TERM':'xterm-256color',name:''},clear=True):
                ui=TerminalUI(output=TTY())
                with mock.patch.object(ui,'_size',return_value=os.terminal_size((112,42))),mock.patch('owner_mode.terminal_ui.time.sleep') as sleep:
                    ui.banner('1.0.0')
                sleep.assert_not_called()

    def test_animation_runs_once_for_three_quarters_of_a_second(self):
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color'},clear=True),mock.patch('owner_mode.terminal_ui.sys.stdin',TTY()):
            output=TTY();ui=TerminalUI(output=output)
            with mock.patch.object(ui,'_size',return_value=os.terminal_size((112,42))),mock.patch('owner_mode.terminal_ui.time.monotonic',side_effect=[0]+[i*.05 for i in range(15)]),mock.patch('owner_mode.terminal_ui.time.sleep') as sleep:
                ui.banner('1.0.0');ui.banner('1.0.0')
            self.assertEqual(sleep.call_count,15)
            self.assertAlmostEqual(sum(c.args[0] for c in sleep.call_args_list),.75)
            self.assertEqual(output.getvalue().count('\x1b[?25l'),1)
            self.assertIn('\x1b[?25h',output.getvalue())

    def test_interrupted_animation_restores_cursor(self):
        with mock.patch.dict(os.environ, {'TERM':'xterm-256color'},clear=True),mock.patch('owner_mode.terminal_ui.sys.stdin',TTY()):
            output=TTY();ui=TerminalUI(output=output)
            with mock.patch.object(ui,'_size',return_value=os.terminal_size((112,42))),mock.patch('owner_mode.terminal_ui.time.sleep',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
                ui.banner('1.0.0')
            self.assertTrue(output.getvalue().endswith('\x1b[0m\x1b[?25h'))


if __name__ == '__main__':
    unittest.main()
