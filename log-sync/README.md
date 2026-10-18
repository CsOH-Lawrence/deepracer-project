# 本地日志同步（只读）

程序不会启动、停止或修改训练。通过 SSH 接收指定模型的 Docker 日志，并访问远端 loopback 看板保存指标快照。它需要已有 Docker 读取权限，但不会修改 Docker socket 权限。

## Windows 设置

在本目录执行：

```powershell
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item config.example.json config.json
```

编辑仅保存在本机、已被 Git 忽略的 `config.json`：设置主机、用户名、模型前缀、保存目录及已验证的 SSH known_hosts 文件。通过云平台可信渠道核对主机密钥指纹后再使用；不要盲目信任网络上第一次返回的密钥。`known_hosts` 和 `log_directory` 相对配置文件所在目录解析。

运行 `./start-sync.ps1`，在终端提示处输入密码。密码仅留在内存供本次连接和重连使用，不写入命令行、配置或日志。

## 保存内容

- `containers/*.log`：选定控制器、仿真和模型训练容器 stdout/stderr；首次补拉仍保留的历史，之后持续追加，默认每 32 MiB 分卷。
- `metrics/latest.json` 和 `metrics/snapshots/*.json`：看板规范化后的完整指标快照；每 10 秒检查，仅内容变化时写新快照，不是原始 S3 文件的逐字副本。
- `status.json`、`state.json`：本地状态和续传位置；`sync-events.jsonl` 保存连接事件。

训练容器根据模型前缀筛选，优先读取环境变量或只读挂载的 SageMaker 输入配置。可用 `known_trainers` 映射作为明确的回退，但不要把其他项目容器加入。

## 断线与停止

重连从保存时间前两秒补拉，因此可能有少量重复行（at-least-once），不保证恰好一次。已被服务器删除或轮转丢弃的日志无法恢复。指标读取出错会重试；电脑休眠/断网期间无法实时接收。

按 Ctrl+C 或在输出目录创建 `STOP_SYNC` 可停止本地同步，不影响训练；再次运行前移走该标记。输出盘低于 1 GiB 时停止写入。程序不自动删历史，也不是 Windows 开机自启服务；重启电脑后需重新运行。

文本日志会遮盖已识别的凭据，但不能保证覆盖全部敏感字段，分享前必须再次检查。原始输出目录不应上传到公开仓库。

测试：在本目录运行 `.venv/Scripts/python.exe -m unittest -v test_sync_logs.py`。
