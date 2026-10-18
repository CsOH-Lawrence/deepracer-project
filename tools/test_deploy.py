import argparse
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
        self.assertEqual(setup.call_args.args[1:], ('demo-model', 'gpu'))

    def test_school_requires_cpu(self):
        with patch.object(deploy, 'preflight') as check, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main(['--school']), 1)
        check.assert_not_called()

    def test_cpu_check_passes_correct_mode_without_install(self):
        with patch.object(deploy, 'preflight') as check, patch.object(deploy, 'setup') as setup:
            self.assertEqual(deploy.main(['--arch', 'cpu', '--school', '--check']), 0)
        self.assertEqual(check.call_args.args[1:], ('cpu', True))
        setup.assert_not_called()

    def test_school_cannot_install_without_dedicated_vm_confirmation(self):
        with patch.object(deploy, 'preflight'), patch.object(deploy, 'setup') as setup, \
                patch('builtins.input', return_value='DEPLOY'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(deploy.main(['--arch', 'cpu', '--school']), 1)
        setup.assert_not_called()

    def test_school_install_requires_two_confirmations(self):
        with patch.object(deploy, 'preflight'), patch.object(deploy, 'setup') as setup, \
                patch('builtins.input', side_effect=['DEDICATED_VM', 'DEPLOY']), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(deploy.main(['--arch', 'cpu', '--school', '--model', 'school-pilot']), 0)
        self.assertEqual(setup.call_args.args[1:], ('school-pilot', 'cpu'))

    def test_docker_install_option_cannot_mutate_in_check_mode(self):
        with patch.object(deploy, 'preflight'), patch.object(deploy, 'prepare_docker') as prepare, \
                patch.object(deploy.shutil, 'which', return_value=None):
            self.assertEqual(deploy.main(['--arch', 'cpu', '--school', '--check', '--install-docker']), 0)
        prepare.assert_not_called()

    def test_docker_install_not_allowed_from_gpu_entry(self):
        with patch.object(deploy, 'preflight') as check, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main(['--install-docker']), 1)
        check.assert_not_called()

    def test_docker_preparation_requires_explicit_option(self):
        with patch.object(deploy, 'preflight', side_effect=RuntimeError('docker missing')), \
                patch.object(deploy, 'prepare_docker') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main(['--arch', 'cpu', '--school']), 1)
        prepare.assert_not_called()

    def test_cpu_preflight_does_not_require_nvidia(self):
        def fake_run(*args, **kwargs):
            if args[:2] == ('docker', 'info'):
                return '{"DefaultRuntime":"runc","Runtimes":{"runc":{}}}'
            if args[:3] == ('docker', 'network', 'ls'):
                return 'bridge\nhost\nnone'
            return ''
        with patch.object(deploy, 'host_checks'), patch.object(deploy, 'run', side_effect=fake_run) as runner, \
                patch.object(deploy.shutil, 'which', side_effect=lambda name: None if 'nvidia' in name else '/bin/' + name), \
                patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3)), \
                patch.object(deploy.socket, 'socket'), contextlib.redirect_stdout(io.StringIO()):
            deploy.preflight(Path('example-target'), 'cpu')
        self.assertFalse(any('nvidia' in str(call) for call in runner.call_args_list))

    def test_cpu_preflight_rejects_existing_container(self):
        def fake_run(*args, **kwargs):
            return '{"Runtimes":{}}' if args[:2] == ('docker', 'info') else 'old-container'
        with patch.object(deploy, 'host_checks'), patch.object(deploy, 'run', side_effect=fake_run), \
                patch.object(deploy.shutil, 'which', return_value='/bin/example'), \
                patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3)):
            with self.assertRaisesRegex(RuntimeError, '已有容器'):
                deploy.preflight(Path('example-target'), 'cpu')

    def test_cpu_setup_uses_cpu_images_and_never_starts_training(self):
        import json
        # Integration simulation: real temporary files, no apt/network/Docker/systemd.
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'training'
            (target / 'defaults').mkdir(parents=True)
            (target / 'docker').mkdir()
            (target / 'defaults/dependencies.json').write_text('{"containers":{"simapp":"6.0.6"}}')
            (target / 'system.env').write_text('DR_CAMERA_KVS_ENABLE=True\nDR_WORKERS=2\nDR_QUIET_ACTIVATE=False\n')
            (target / 'run.env').write_text('DR_WORLD_NAME=other\nDR_LOCAL_S3_MODEL_PREFIX=old\nDR_TRAIN_REVERSE_DIRECTION=True\n')
            def fake_run(*args, **kwargs):
                if args[:3] == ('git', 'rev-parse', 'HEAD'):
                    return deploy.REVISION
                if args[:3] == ('docker', 'inspect', 's3-minio-1'):
                    return 'true'
                if args[:2] == ('id', '-un'):
                    return 'teacher'
                if len(args) > 2 and args[1] == '-c' and 'yaml.safe_load' in str(args[2]):
                    return json.dumps({'services': {'minio': {'ports': ['9000:9000']}}})
                return ''
            # Windows temporary paths do not fit the Linux systemd path guard.
            original_search = deploy.re.search
            def search(pattern, value, *args):
                return None if pattern == r'[^A-Za-z0-9_./-]' else original_search(pattern, value, *args)
            response = io.BytesIO(b'{"model":"school-demo","rows":[]}')
            with patch.object(deploy, 'run', side_effect=fake_run) as runner, \
                    patch.object(deploy.re, 'search', side_effect=search), \
                    patch('urllib.request.urlopen', return_value=response), contextlib.redirect_stdout(io.StringIO()):
                deploy.setup(target, 'school-demo', 'cpu')
            calls = [tuple(str(arg) for arg in call.args) for call in runner.call_args_list]
            self.assertIn(('docker', 'pull', 'awsdeepracercommunity/deepracer-simapp:6.0.6-cpu'), calls)
            self.assertTrue(any(call[:4] == ('bash', 'bin/init.sh', '-a', 'cpu') for call in calls))
            self.assertFalse(any('--gpus' in call or deploy.CUDA in call for call in calls))
            self.assertFalse(any('dr-start-training' in ' '.join(call) for call in calls))
            document = json.loads((target / 'docker/docker-compose-local.yml').read_text())
            self.assertEqual(document['services']['minio']['ports'], ['127.0.0.1:9000:9000'])
            self.assertEqual(document['services']['minio']['mem_limit'], '2g')
            self.assertIn('DR_LOCAL_S3_MODEL_PREFIX=school-demo', (target / 'run.env').read_text())

    def test_upstream_is_immutable_commit(self):
        self.assertRegex(deploy.REVISION, r'^[a-f0-9]{40}$')


if __name__ == '__main__':
    unittest.main()
