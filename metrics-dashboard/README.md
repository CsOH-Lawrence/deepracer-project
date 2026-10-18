# 只读奖励曲线看板

`server.py` 从指定模型的 `metrics/TrainingMetrics.json` 读取指标并规范化，`index.html` 使用原生 JavaScript/SVG 绘图，不需要前端构建。训练和评估分开显示；横轴按训练回合排序，避免仿真时钟重置把曲线错误连回起点。

## 云服务器运行

在训练环境所在服务器安装依赖，使用服务器已有的私有 `minio` AWS profile（不要复制凭据到本仓库）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py --profile minio --endpoint http://127.0.0.1:9000 --bucket bucket --model-prefix YOUR_MODEL_PREFIX --port 3000
```

地址仅绑定 `127.0.0.1`。已有模型名必须与真实 S3 前缀一致；找不到模型指标时显示等待，不会写入 S3 或启动训练。若 AWS profile 不可读，应使用拥有该 profile 的用户运行。

## 本机访问

将占位符换成自己的 SSH 用户和主机，先通过可信渠道核对 SSH 主机密钥：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 127.0.0.1:13791:127.0.0.1:3000 USER@YOUR_SERVER
```

浏览器打开 <http://127.0.0.1:13791/>。API 为 `/api/metrics`，健康检查为 `/health`。摄像头是仿真容器的另一个服务，需要查明实际端口再单独转发，不由看板实现。

## 指标说明

- 单回合奖励与最近 20 回合移动平均；同一训练回合的多个评估试次聚合成评估点。
- 平均完成率不是成功跑完圈数占比；“完整跑圈/训练回合”只统计训练部分。
- API 保留 `phase`、`status`、`progress`、`seconds` 等字段，可独立筛选完整评估圈。
- 数据刷新不代表训练一定仍在运行，应同时检查时间戳和容器状态。

测试：在本目录运行 `.venv/bin/python -m unittest -v test_server.py`。Windows 使用 `.venv/Scripts/python.exe`。
