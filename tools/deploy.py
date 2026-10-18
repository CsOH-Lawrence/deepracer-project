"""Interactive fresh-VM setup. Never starts training or overwrites an installation."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

UPSTREAM = 'https://github.com/aws-deepracer-community/deepracer-for-cloud.git'
REVISION = '245bd7f0fe0810043744d988edcae78c03617fee'
CUDA = 'nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04'
SOURCE = Path(__file__).resolve().parents[1]


def run(*args, cwd=None, capture=False):
    if not capture:
        print('[执行] ' + ' '.join(map(str, args)), flush=True)
    result = subprocess.run(list(map(str, args)), cwd=cwd, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ''


def fail(message):
    raise RuntimeError(message)


def env_set(path, key, value):
    text = path.read_text(encoding='utf-8')
    pattern = re.compile(r'^' + re.escape(key) + r'=.*$', re.M)
    if len(pattern.findall(text)) != 1:
        fail(f'预期恰好一个 {key}，停止修改 {path.name}')
    path.write_text(pattern.sub(lambda _: key + '=' + value, text), encoding='utf-8')


def loopback_ports(document):
    """Only transform Compose port declarations, preserving ranges/variables."""
    for service in document.get('services', {}).values():
        result = []
        for port in service.get('ports', []):
            if not isinstance(port, str) or port.count(':') != 1:
                fail('遇到未知端口模板，拒绝猜测公网绑定')
            result.append('127.0.0.1:' + port)
        if 'ports' in service:
            service['ports'] = result
    return document


def validate_model(value):
    if not re.fullmatch(r'[a-z][a-z0-9-]{2,63}', value):
        raise argparse.ArgumentTypeError('模型名须为 3–64 位小写字母、数字、横线，以字母开头')
    return value


def preflight(target):
    if sys.platform != 'linux' or not Path('/etc/os-release').exists():
        fail('只支持 Ubuntu 22.04 服务器；Windows .bat 不能直接在这里运行')
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    if release.get('ID', '').strip('"') != 'ubuntu' or release.get('VERSION_ID', '').strip('"') != '22.04':
        fail('v1 向导仅支持 Ubuntu 22.04；其他系统请使用手动指南')
    if os.geteuid() == 0:
        fail('请用有 sudo 权限的普通 SSH 用户运行，不要 sudo bash deploy.sh')
    if re.search(r'[^A-Za-z0-9_./-]', str(target)):
        fail('v1 安装路径只能包含英文字母、数字、下划线、横线、点和斜线')
    if target.exists() or target.is_symlink():
        fail(f'{target} 已存在。为保护旧模型，不覆盖、不初始化；请使用续训流程')
    if any(c.isspace() for c in str(target)):
        fail('上游不支持带空格的安装路径')
    if Path('/.dockerenv').exists():
        fail('检测到容器环境；请选择供应商 GPU 虚拟机')
    if shutil.which('systemd-detect-virt'):
        result = subprocess.run(['systemd-detect-virt', '--container', '--quiet'])
        if result.returncode == 0:
            fail('检测到容器环境；不尝试绕过宿主机限制')
    for command in ('docker', 'nvidia-smi', 'sudo', 'systemctl'):
        if not shutil.which(command):
            fail(f'缺少 {command}；请选择已预装 Docker / NVIDIA Toolkit 的 GPU 虚拟机')
    if shutil.disk_usage(Path.home()).free < 50 * 1024 ** 3:
        fail('安装盘可用空间不足 50 GiB，请先扩容；向导不会删除现有文件')
    info = json.loads(run('docker', 'info', '--format', '{{json .}}', capture=True))
    if 'nvidia' not in info.get('Runtimes', {}):
        fail('Docker 未配置 NVIDIA runtime；请先按官方 Toolkit 文档配置')
    if info.get('DefaultRuntime') != 'nvidia':
        fail('默认 runtime 不是 nvidia。请先按指南第3步配置；向导不会重启已有 Docker')
    if run('docker', 'ps', '-aq', capture=True):
        fail('检测到已有容器，包括停止的容器。为避免干扰其他项目，向导仅接受空白 Docker 主机')
    networks = run('docker', 'network', 'ls', '--format', '{{.Name}}', capture=True).splitlines()
    if 'sagemaker-local' in networks:
        fail('已有 DeepRacer 网络，停止：不要在旧环境再次初始化')
    unit = Path('/etc/systemd/system/drfc-metrics-dashboard.service')
    if unit.exists():
        fail('已有看板 systemd 服务，停止以免覆盖')
    for port in (3000, 9000, 9001):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                fail(f'端口 {port} 已占用，不停止原服务')
    run('docker', 'compose', 'version')
    run('nvidia-smi', '-L')
    print('基础检查通过：Ubuntu 22.04 / Docker / NVIDIA / 磁盘 / 无旧安装', flush=True)


def setup(target, model):
    # Interactive sudo and apt: no downloaded shell installers, driver replacement,
    # disk partitioning, docker restart, firewall edits or credential output.
    run('sudo', '-v')
    run('sudo', 'apt-get', 'update')
    run('sudo', 'apt-get', 'install', '-y', 'git', 'jq', 'awscli', 'python3-venv')
    run('docker', 'run', '--rm', 'hello-world')
    run('docker', 'run', '--rm', '--gpus', 'all', CUDA, 'nvidia-smi')
    run('git', 'clone', UPSTREAM, target)
    run('git', 'checkout', '--detach', REVISION, cwd=target)
    if run('git', 'rev-parse', 'HEAD', cwd=target, capture=True) != REVISION:
        fail('上游版本不匹配')
    run('python3', '-m', 'venv', target / '.venv')
    python = target / '.venv/bin/python'
    run(python, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', '-r', 'requirements.txt', cwd=target)
    run(python, '-m', 'pip', 'check')
    dependencies = json.loads((target / 'defaults/dependencies.json').read_text())
    simapp = 'awsdeepracercommunity/deepracer-simapp:' + dependencies['containers']['simapp'] + '-gpu'
    # Pull every image explicitly; upstream init alone does not reliably fail fast.
    for image in (simapp, 'minio/minio:latest'):
        run('docker', 'pull', image)
        run('docker', 'image', 'inspect', image, '--format', '{{json .RepoDigests}}')
    if run('docker', 'ps', '-aq', capture=True) or 'sagemaker-local' in run(
            'docker', 'network', 'ls', '--format', '{{.Name}}', capture=True).splitlines():
        fail('下载期间检测到其他容器或训练网络，停止初始化以避免干扰')
    # All prerequisites are complete before the one-time init overwrites templates.
    run('bash', 'bin/init.sh', '-a', 'gpu', '-c', 'local', '-s', 'compose', cwd=target)
    run('docker', 'image', 'inspect', simapp, '--format', '{{.Id}}')
    env_set(target / 'system.env', 'DR_CAMERA_KVS_ENABLE', 'False')
    env_set(target / 'system.env', 'DR_WORKERS', '1')
    env_set(target / 'system.env', 'DR_QUIET_ACTIVATE', 'True')
    env_set(target / 'run.env', 'DR_WORLD_NAME', 'reinvent_base')
    env_set(target / 'run.env', 'DR_LOCAL_S3_MODEL_PREFIX', model)
    env_set(target / 'run.env', 'DR_TRAIN_REVERSE_DIRECTION', 'False')
    # Edit only the fresh clone; PyYAML is already an upstream requirement.
    for name in ('docker-compose-local.yml', 'docker-compose-training.yml', 'docker-compose-eval.yml'):
        path = target / 'docker' / name
        converted = run(python, '-c',
                        'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))', path, capture=True)
        document = loopback_ports(json.loads(converted))
        if name == 'docker-compose-local.yml':
            document['services']['minio']['restart'] = 'unless-stopped'
        # JSON is a YAML subset accepted by Docker Compose.
        path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    # Isolate upstream's older shell assumptions from this Python process.
    run('bash', '-c', 'source bin/activate.sh && dr-upload-custom-files', cwd=target)
    if run('docker', 'inspect', 's3-minio-1', '--format', '{{.State.Running}}', capture=True) != 'true':
        fail('MinIO 未运行；未创建看板服务，不会开始训练')
    run(python, '-c',
        "import boto3; from botocore.config import Config; "
        "s=boto3.Session(profile_name='minio').client('s3',endpoint_url='http://127.0.0.1:9000',"
        "config=Config(connect_timeout=5,read_timeout=5)); "
        "[s.head_object(Bucket='bucket',Key='custom_files/'+n) for n in "
        "('reward_function.py','model_metadata.json','hyperparameters.json')]; "
        "print('三份训练配置已在私有MinIO验证')")
    dashboard = target / 'local-metrics-dashboard'
    dashboard.mkdir()
    for name in ('server.py', 'index.html'):
        shutil.copy2(SOURCE / 'metrics-dashboard' / name, dashboard / name)
    # A normal home path has no systemd specifiers; reject unsupported paths safely.
    if re.search(r'[^A-Za-z0-9_./-]', str(target)):
        fail('安装路径包含 systemd 模板不支持的字符，停止安装看板服务')
    username = run('id', '-un', capture=True)
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', username):
        fail('无法安全生成 systemd 用户字段')
    unit = dashboard / 'drfc-metrics-dashboard.service'
    unit.write_text(
        '[Unit]\nDescription=DeepRacer read-only metrics\nAfter=network.target docker.service\n'
        '[Service]\nType=simple\nUser=' + username + '\nWorkingDirectory=' + str(dashboard) + '\n'
        'ExecStart=' + str(python) + ' ' + str(dashboard / 'server.py') +
        ' --model-prefix ' + model + '\nRestart=on-failure\nRestartSec=5\n'
        '[Install]\nWantedBy=multi-user.target\n', encoding='utf-8')
    run('sudo', 'install', '-m', '644', unit, '/etc/systemd/system/drfc-metrics-dashboard.service')
    run('sudo', 'systemctl', 'daemon-reload')
    run('sudo', 'systemctl', 'enable', '--now', 'drfc-metrics-dashboard.service')
    # Health + metrics verify S3 credentials and connectivity, not just process state.
    import urllib.request
    verified = False
    for _ in range(10):
        try:
            with urllib.request.urlopen('http://127.0.0.1:3000/api/metrics', timeout=5) as response:
                metrics = json.load(response)
            verified = metrics.get('model') == model and 'rows' in metrics
            if verified:
                break
        except (OSError, ValueError):
            pass
        time.sleep(1)
    if not verified:
        fail('看板 API 未通过检查，请读取服务日志；未开始训练')
    print('\n部署检查通过，尚未开始训练。默认奖励/动作空间，没有替你提速。')
    print('模型：' + model + '\n上游版本：' + REVISION)
    print('看板只监听 127.0.0.1:3000，请用 SSH 隧道访问。')
    print('确认平台自动关机已设置后，另行在服务器执行：')
    print('cd ' + str(target) + '\nsource bin/activate.sh\ndr-start-training -q')
    print('本机隧道：ssh -N -T -L 127.0.0.1:13791:127.0.0.1:3000 USER@YOUR_SERVER')
    print('摄像头须在训练启动后查实际端口。没有修改 DNS、防火墙或公网认证。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只读检查，不安装、不拉镜像、不创建配置')
    parser.add_argument('--model', type=validate_model,
                        default='my-first-model-' + time.strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args(argv)
    target = Path.home() / 'deepracer-for-cloud'
    try:
        preflight(target)
        if args.check:
            return 0
        print('\n将安装 apt/Python 项目依赖、固定版本训练平台、MinIO 和只读看板。')
        print('可能下载数 GB 镜像；会创建 Docker 网络和 systemd 看板服务。')
        print('不会安装/重装 Docker 或驱动，不重启 Docker、不分区、不清理旧文件、不开始训练。')
        print('目标：' + str(target))
        if input('请先设置平台自动关机；确认已设置并接受上述操作后输入 DEPLOY：').strip() != 'DEPLOY':
            print('已取消，未安装。')
            return 1
        # Recheck after human input to catch intervening container/port changes.
        preflight(target)
        setup(target, args.model)
        return 0
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        # Avoid echoing raw command output which could contain credentials.
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print('\n部署停止：' + message, file=sys.stderr)
        print('保留已下载文件和已有数据；不会自动删除或重试覆盖。未由本向导启动训练。', file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print('\n已中止；未自动清理文件或启动训练。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
