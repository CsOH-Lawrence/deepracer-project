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

## 两个或更多并行仿真器

上游 deepracer-for-cloud 支持多个 Robomaker 仿真器给一个训练器采样。先在未运行训练时把 `system.env` 中的 `DR_WORKERS=1` 改为 `DR_WORKERS=2`，备份 `run.env`，再为续训建立**全新模型前缀**，不要覆盖旧前缀。20 回合一次参数更新、2 个仿真器时，可以把 `DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST` 设为 `0.10`，使每个仿真器的 10 个起点覆盖赛道。训练的停止与续训命令要以所用上游版本为准。切换前核对源模型已保存的 checkpoint、剩余磁盘和 GPU 内存；切换后验证两个仿真器都产生训练回合，并看到一次新参数更新。

看板启动时增加 `--workers 2`，会并发读取 `metrics/TrainingMetrics.json` 与 `metrics/TrainingMetrics_1.json`，分别使用 ETag 条件请求；未更新的文件复用本地解析结果。API 保持 `source` 为 worker 0 的字符串路径，并新增 `sources` 列表、`worker_modified`、`missing_workers`、`stale_workers`、`healthy_workers`。页面显示各仿真器回合数；缺失或超过 `--stale-seconds`（默认 120 秒）未写入时，状态为黄色 `degraded`。这个阈值只是写入新鲜度提醒，不足以证明训练进程已停止，可按训练/评估实际耗时调大。各仿真器的原始折线互不相连，最近 20 回合统计和移动平均则按两仿真合计的最近 20 条训练记录计算。横轴把每个仿真器的回合放到相邻位置，评估试次归入同一组；它不是实际发生时间。`--workers` 默认 1，旧的单仿真训练照常使用。

本次 RTX 3090、16 vCPU 机器上，短窗口**采样率**约从 9.8 提高到 19.9 步/秒；这不是学习速度或模型收敛速度。上游按 worker 分摊 `num_episodes_between_training`（例如总计 20 回合、2 worker 各 10 回合），每次参数更新汇入两个仿真器的数据，并不意味着参数更新次数翻倍。采样期间单仿真 CPU 约 3 核，两仿真各约 3–4 核，GPU 约 12%。这不是长期基准，仿真与评估时长会波动；增加仿真器也不能直接修复车辆左右摇晃。训练时的探索动作可能造成摇晃，应结合评估画面、动作空间与完成率判断。详见[上游多仿真器说明](https://aws-deepracer-community.github.io/deepracer-for-cloud/multi_worker.html)。

## 本机访问

将占位符换成自己的 SSH 用户和主机，先通过可信渠道核对 SSH 主机密钥：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 127.0.0.1:13791:127.0.0.1:3000 USER@YOUR_SERVER
```

浏览器打开 <http://127.0.0.1:13791/>。API 为 `/api/metrics`，健康检查为 `/health`。摄像头是仿真容器的另一个服务，需要查明实际端口再单独转发，不由看板实现。

## 指标说明

- 单回合奖励与最近 20 个训练回合（多仿真合计）移动平均；同一训练回合的多个评估试次聚合成评估点。
- 平均完成率不是成功跑完圈数占比；“完整跑圈/训练回合”只统计训练部分，且要求 `Lap complete` 与完成率 100% 同时成立。
- API 保留 `phase`、`status`、`progress`、`seconds` 等字段，可独立筛选完整评估圈。
- 数据刷新不代表训练一定仍在运行，应同时检查时间戳和容器状态。
- 服务端按 `--cache-seconds`（默认 8 秒）缓存，并对每个仿真器的 S3 文件分别使用 ETag 条件请求：文件没变时复用已解析的行，不重复下载。
  `--max-rows`（默认 20000）限制单次响应行数，截断时返回 `truncated: true` 与原始 `total_rows`，页面会注明只显示最近的部分。
  响应超过 2 KB 且客户端支持时使用 gzip。

测试：在本目录运行 `.venv/bin/python -m unittest -v test_server.py`。Windows 使用 `.venv/Scripts/python.exe`。
