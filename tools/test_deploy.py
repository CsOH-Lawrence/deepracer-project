import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import deploy


class DeployTests(unittest.TestCase):
    def test_model_name(self):
        self.assertEqual(deploy.validate_model('my-model-001'), 'my-model-001')
        for value in ('ab', '../old', 'x;touch pwned', 'UPPER', '-model', 'x' * 65, 'x\nfoo'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                deploy.validate_model(value)

    def test_loopback_port_ranges_preserved(self):
        data = {'services': {'robo': {'ports': ['${PORT}:8080', '5901-5920:5900']}, 'coach': {}}}
        actual = deploy.loopback_ports(data)
        self.assertEqual(actual['services']['robo']['ports'],
                         ['127.0.0.1:${PORT}:8080', '127.0.0.1:5901-5920:5900'])
        self.assertNotIn('ports', actual['services']['coach'])

    def test_unexpected_port_formats_rejected(self):
        for value in (8080, '8080', '0.0.0.0:8080:8080', {'target': 8080}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                deploy.loopback_ports({'services': {'robo': {'ports': [value]}}})

    def test_env_update_requires_exact_one_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.env'
            path.write_text('# header\nMODEL=old\nOTHER=keep\n')
            deploy.env_set(path, 'MODEL', 'new')
            self.assertEqual(path.read_text(), '# header\nMODEL=new\nOTHER=keep\n')
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                deploy.env_set(path, 'MISSING', 'new')
            self.assertEqual(before, path.read_bytes())
            path.write_text('MODEL=a\nMODEL=b\n')
            with self.assertRaises(RuntimeError):
                deploy.env_set(path, 'MODEL', 'new')

    def test_check_never_installs_or_prompts(self):
        with patch.object(deploy, 'preflight') as check, patch.object(deploy, 'setup') as setup, \
                patch('builtins.input', side_effect=AssertionError('must not prompt')):
            self.assertEqual(deploy.main(['--check']), 0)
        check.assert_called_once()
        setup.assert_not_called()

    def test_cancel_never_installs(self):
        with patch.object(deploy, 'preflight'), patch.object(deploy, 'setup') as setup, \
                patch('builtins.input', return_value='no'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(deploy.main([]), 1)
        setup.assert_not_called()

    def test_fail_closed_on_preflight_failure(self):
        with patch.object(deploy, 'preflight', side_effect=RuntimeError('existing model')), \
                patch.object(deploy, 'setup') as setup, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main([]), 1)
        setup.assert_not_called()

    def test_rechecks_after_confirmation(self):
        with patch.object(deploy, 'preflight', side_effect=[None, RuntimeError('state changed')]) as check, \
                patch.object(deploy, 'setup') as setup, patch('builtins.input', return_value='DEPLOY'), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main([]), 1)
        self.assertEqual(check.call_count, 2)
        setup.assert_not_called()

    def test_only_explicit_confirmation_installs(self):
        with patch.object(deploy, 'preflight'), patch.object(deploy, 'setup') as setup, \
                patch('builtins.input', return_value='DEPLOY'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(deploy.main(['--model', 'demo-model']), 0)
        self.assertEqual(setup.call_args.args[1], 'demo-model')

    def test_upstream_is_immutable_commit(self):
        self.assertRegex(deploy.REVISION, r'^[a-f0-9]{40}$')


if __name__ == '__main__':
    unittest.main()
