# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Interactive direct chat preserves history without imposing an output cap."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
from flashalign.terminal_cli import run


class TerminalCLI(unittest.TestCase):
    def test_direct_chat_history_reset_and_engine_output_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'client.json';config.write_text(json.dumps({'profile':'test'}))
            calls=[]
            class Client:
                def __enter__(self):return self
                def __exit__(self,*_):pass
                def chat(self,messages,**options):
                    calls.append((copy.deepcopy(messages),options))
                    return {'choices':[{'message':{'role':'assistant','content':'reply'}}]}
            with (mock.patch('flashalign.Client.from_config',return_value=Client()),
                  mock.patch('builtins.input',side_effect=['one','two','/reset','three','/exit']),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(run(['chat','--config',str(config),'--owner']),0)
            self.assertEqual([len(messages) for messages,_ in calls],[1,3,1])
            self.assertTrue(all(options=={'profile':'test','scale':1.} for _,options in calls))

    def test_native_terminal_rejects_noninteractive_input_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('sys.stdin.isatty',return_value=False):
                with self.assertRaisesRegex(ValueError,'interactive terminal'):
                    run(['terminal','--harness','hermes','--authorize','--home',directory])
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__=='__main__':unittest.main()
