# 全 LLM 实验：GPU 服务器运行手册

本版本在 CPU 上验证接口、轮次同步、1000／2000 个独立请求和实验流水线。
**尚未在 GPU 上测量真实模型行为、吞吐量或新的标度结果。** 论文旧数值不能直接
用于本版本。先完成模型行为与容量 pilot，再冻结配置、运行主实验。

## 1. 架构与计算量

N 个 agent 共享一个 vLLM 服务，每个 agent 有自己的资产、身份、收件箱、历史和
短记忆。每轮冻结所有观测，客户端以有限并发提交 N 个请求，全部返回并通过 schema
验证后，按固定顺序执行订单和消息。不会让先返回的 agent 提前改变别人的观测。

1000 个 agent、30 轮 = 每 episode 30,000 次生成；2000 个 = 60,000 次。
总调用量还要乘 α 网格、随机种子和实验条件。`plan` 会先打印准确预算。
连续批处理不意味着必须同时容纳 2000 条序列，也不需要 2000 份模型权重。

- `CONCURRENCY`：客户端同时在途请求数，初始用 32。
- `MAX_NUM_SEQS`：vLLM 同时调度序列数，初始用 256，按显存调整。
- `TP`：一份模型的 tensor parallel GPU 数；模型单卡放不下时使用。
- `DP`：模型副本数；有足够 GPU 时提高总吞吐。单机总 GPU 数为 `TP × DP`。
- `MAX_MODEL_LEN`：输入和输出的总 token 上限。实际输入长度随消息、记忆增长。

vLLM 官方资料：[OpenAI-compatible 服务](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)、
[结构化输出](https://docs.vllm.ai/en/stable/features/structured_outputs/)、
[数据并行](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)、
[可复现性](https://docs.vllm.ai/en/stable/usage/reproducibility/)。
本客户端使用 `structured_outputs.json`，不使用旧 `guided_json`。

## 2. 安装与启动服务

以下命令在仓库根目录执行。推荐 GPU 服务和实验客户端分别使用虚拟环境，避免
vLLM 的 CUDA／PyTorch 依赖与分析依赖冲突。GPU 环境的 Python、CUDA、驱动需
符合所选 vLLM 版本的[安装要求](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

```bash
python3 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip
# 如服务器已有经过验证的版本，安装该固定版本；首次安装后记录精确版本。
python -m pip install vllm
mkdir -p runs/provenance
python -m pip freeze > runs/provenance/server-requirements.txt
nvidia-smi > runs/provenance/gpu.txt
```

启动脚本要求 vLLM ≥ 0.12，具体模型与 flags 的兼容性仍需实际启动验证。
用本地已下载模型最方便；`SERVED_MODEL` 是客户端要请求的模型名。

```bash
export MODEL=/srv/models/your-model
export SERVED_MODEL=wolf-policy
export CUDA_VISIBLE_DEVICES=0
export SERVER_MANIFEST="$PWD/runs/provenance/server.json"
TP=1 DP=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=256 \
  scripts/serve_vllm.sh
```

脚本默认监听 `127.0.0.1:8000`，启用 prefix caching、chunked prefill，并输出服务
版本和启动参数。使用 Hugging Face 模型名时设置 `MODEL_REVISION` 为不可变 commit，
`TOKENIZER_REVISION` 默认与其一致。使用本地路径时另存模型、tokenizer 文件的
SHA256 清单，后续把它的哈希传给客户端的 `WEIGHTS_REVISION` 参数。

默认 `VLLM_BATCH_INVARIANT=1` 请求批次不变性；只在 vLLM 支持的模型与硬件上使用。
如目标组合不支持，显式设为 `0` 并保留记录。即使 seed 相同，也不要跨模型、版本、
硬件、服务拓扑宣称逐 token 一致。脚本记录配置，不能自行验证实际模型权重的真实性。

多卡例子：`CUDA_VISIBLE_DEVICES=0,1,2,3 TP=2 DP=2 ... scripts/serve_vllm.sh`。
先让单副本稳定运行，再比较吞吐。无需把 `MAX_NUM_SEQS` 设为 agent 总数。

## 3. 安装客户端并做 CPU 验收

另开终端，仍在仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot]'
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
pytest -q

python -m paper_experiments.runner plan --mock --stage smoke --families all \
  --sizes 20 --seeds 9001 --horizon 3 --manifest runs/qa/manifest.json
python -m paper_experiments.runner run --manifest runs/qa/manifest.json --output runs/qa
```

`--mock` 明确不调用模型，结果写入 `mock/`。默认 fixture 选择 hold，用于接口检查，
不用于证明交易活跃、传播或 collapse。真实数据写入 `real/`，分析默认拒绝 mock。
旧 `wolfbench run` 和 `paper_experiments_v3` 仍是旧运行路径。

## 4. 实测 1000／2000 请求与小规模行为

先导出 1000 个独立 agent 的真实初始观测；此步骤无模型调用：

```bash
python scripts/export_initial_requests.py --n-agents 1000 --output runs/load/initial-1000.jsonl
MODEL=wolf-policy NUM_AGENTS=1000 CONCURRENCY=32 ROUNDS=3 \
  REQUESTS_JSONL=runs/load/initial-1000.jsonl \
  BENCH_OUTPUT=runs/load/benchmark-1000 scripts/benchmark_vllm.sh

python scripts/export_initial_requests.py --n-agents 2000 --output runs/load/initial-2000.jsonl
MODEL=wolf-policy NUM_AGENTS=2000 CONCURRENCY=32 ROUNDS=3 \
  REQUESTS_JSONL=runs/load/initial-2000.jsonl \
  BENCH_OUTPUT=runs/load/benchmark-2000 scripts/benchmark_vllm.sh
```

benchmark **会真正调用模型**，保存每轮时长、请求／token 吞吐、延迟分位数、
请求和失败记录。这里重放初始观测，收件箱与记忆尚为空，适合启动与容量检查。
不能把这个速度直接外推到完整实验。随后从真实 pilot 的 `*.backend_audit.jsonl.gz`
解压并选取后期完整一轮，再运行 benchmark 测长上下文吞吐；不要把多个轮次的同一
agent 冒充为同一轮的不同 agent。每次 benchmark 使用新的输出目录。

开始小规模真实行为 pilot（替换权重版本占位值）：

```bash
EXECUTE=1 scripts/run_pilot.sh wolf-policy WEIGHTS_REVISION runs/pilot-small \
  --families p01 --sizes 20,50,100 --seeds 1001 --horizon 5
```

对于支持关闭 thinking 的模型，可在 pilot/main 命令后添加：
`--chat-template-kwargs '{"enable_thinking":false}'`。
benchmark 对应设置 `CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'`。
只有模型的 chat template 支持该选项时才使用。`--max-tokens` 默认 512；结合输出
长度和模型行为测试确定，并在完整 pilot 前固定。不要通过忽略截断输出继续运行。

查看实际生成的请求、决策、成交和消息轨迹：是否真有多种行为、消息是否被读取，
是否经常撞预算上限、是否出现拒绝／截断。检查干净 α=0 条件是否本身就普遍失效。
单纯“JSON 能解析”不足以确认实验有效。模型策略可能不发动攻击或不响应消息，
这需要如实记录，不得在代码中偷偷补回规则攻击。

## 5. Pilot、冻结网格、正文及附录

完整 family 与科学含义见 [实验注册表](paper_experiments/README.md)。最初优先 P01，
覆盖最终所有 N；等行为与性能稳定后再分批覆盖附录。

```bash
EXECUTE=1 scripts/run_pilot.sh wolf-policy WEIGHTS_REVISION runs/pilot-p01 \
  --families p01 --sizes 100,200,300,500,1000,2000 --seeds 1001-1003
```

脚本默认只创建 manifest 并打印预算，`EXECUTE=1` 才执行。每次改变模型、网格或条件
选择，使用新的输出根目录。初始 α 候选是 0,.01,.03,.06,.10,.20,.40；旧混合模型的
临界区间没有被预设为新模型答案。命令结束返回 `run_dir`，下面 `RUN_HASH` 必须用
实际目录名替换。

```bash
python -m paper_experiments.analysis runs/pilot-p01/real/RUN_HASH --output runs/pilot-tables
python -m paper_experiments.grids refine runs/pilot-p01/real/RUN_HASH --output runs/refined-grid.json
EXECUTE=1 scripts/run_pilot.sh wolf-policy WEIGHTS_REVISION runs/pilot-refined \
  --families p01 --sizes 100,200,300,500,1000,2000 \
  --grid runs/refined-grid.json --seeds 1004-1006
python -m paper_experiments.grids freeze runs/pilot-refined/real/RUN_HASH --output runs/frozen-grid.json
EXECUTE=1 scripts/run_full_llm_paper.sh wolf-policy WEIGHTS_REVISION \
  runs/frozen-grid.json runs/main-p01 --families p01 --seeds 1-12
```

其余 family 按同样流程单独 pilot→freeze→main。全部一起运行时，把两阶段的
`--families` 设为 `all`，并从相应 all-family pilot 冻结完整网格；不能拿只有 P01
的冻结文件直接声称其他条件也已 pilot。示例预算：全部宽网格 pilot（3 个 seed，
30 轮）约 3,363 episodes、5,828 万次生成；先查看当前 manifest 的准确预算。

`freeze` 默认要求临界区间已解析；确实没有转变的条件可显式 `--allow-censored`，
其含义是保留未解析状态。不能把缺失阈值填成拟合值。主实验使用独立 seed，且代码、
prompt、horizon 和采样参数应与 pilot 固定协议一致。改协议需重新 pilot。

P07 用同一冻结对照网格和 seed 分别运行每个权重固定的模型；每个服务使用独立输出
目录。主文／附录分析可读取多个完成目录，输出调用审计、边界与指数、bootstrap、
阈值敏感性、固定 K 分析、held-out response 预测等表。真实数据不足时保留缺失状态。

```bash
python -m paper_experiments.analysis runs/main-p01/real/RUN_HASH \
  --output runs/main-tables --bootstrap-draws 5000
```

## 6. 正式论文出图：拒绝旧数据与自动回退

**新实验结束后只使用下面的新出图入口。** 它直接读取指定 run 的 manifest、完成记录
和模型轨迹，不读取旧 CSV、旧图片、v3 outputs，也不会按文件修改时间猜哪个结果最新。
当前论文包含 teaser 与三张结果图；附录没有额外图片。对应关系固定在
`paper_experiments/paper_figure_catalog.json`。

**沿用原来的画图代码和样式。** 新入口只校验数据并适配字段，然后直接调用
`paper_experiments_v3/figures/make_paper_figures.py` 的原绘图函数：Fig. 2 双面板
（崩溃概率、episode severity）、Fig. 3 缩放拟合、Fig. 4 分组森林图。旧 CSV
读取函数不被调用。Teaser 保留原有静态美术和布局；其 A/B 曲线区域使用新数据
重新绘制。仓库中的 `teaser_layout.pdf` 已移除原曲线，并非旧结果 PDF。

```bash
python -m pip install -e '.[plot]'
scripts/build_paper_figures.sh wolf-policy WEIGHTS_REVISION runs/figures-release-01 \
  runs/main-p01/real/P01_RUN_HASH runs/main-p04/real/P04_RUN_HASH
```

如果一个完整主实验 run 同时包含 P01/P04，只传该目录即可。替换示例路径和 revision。
正式图要求：

- 真实 `vllm`、`main` 阶段、当前实验代码、冻结网格、同一模型及采样协议。
- revision 必须为 40 位权重 commit，或 64 位 SHA256（允许 `sha256:` 前缀）。
  这是记录格式约束，实际服务权重仍需与你保存的 server manifest 对照核实。
- P01 六个论文规模、P04 两个论文规模及全部已注册干预，完整 30 轮、至少 12 个配对
  主实验 seeds；每个 α 都有相同 seeds，包含 α=0。
- 传入的 manifest 全部 episodes 完成；每个图源 episode 的请求、输出、动作和日级
  失败记录相互一致；不会只选成功子集或把 mock／pilot 当正式数据。

任一条件不满足就报错，**不生成正式图包，也不复制旧图补位**。没有临界转变是允许
的实测结果，会明确显示未解析／删失；缺失实验数据则不允许出图。图中原来写死的
阈值数字、坐标范围及结果方向已改为由新数据决定；点、拟合和区间全部重新计算。
Teaser 的静态机制示意保留，A/B 结果面板与 Fig. 2/3 使用同一组新数据图形。

图包包含四张 PDF、四张 PNG、逐图源 CSV，以及 `figure_manifest.json`。manifest
保存具体实验 run、job IDs、模型 revision、数据和图片哈希。输出目录必须是新的，
不能覆盖一个已有图包。改变图样式只需重新生成图包；改变实验／prompt 后则必须使用
相应新实验数据，不能借旧结果生成新版本正式图。

安装到论文前先验证图包：

```bash
python -m paper_experiments.figure_bundle verify \
  --manifest runs/figures-release-01/figure_manifest.json
python -m paper_experiments.manuscript_figures install \
  --paper-dir /ABS/PAPER_SOURCE \
  --manifest runs/figures-release-01/figure_manifest.json
python -m paper_experiments.manuscript_figures check \
  --paper-dir /ABS/PAPER_SOURCE \
  --manifest runs/figures-release-01/figure_manifest.json
```

`install` 才会替换论文的四个 PDF；默认 `check` 只读。安装前检查实际 TeX 引用，
安装后逐文件核对哈希。旧 PDF、未知图、引用绕到其他目录、图文件或源 CSV 被手改
都会导致失败。脚本不改 captions、正文数值或表格；这些需要按新结果另行更新。
审查需要访问原实验记录，建议在 GPU 服务器上带着论文源码目录完成安装和验证，
再传送生成的论文材料。不要删除被图包引用的原 run 或改其路径。

现有旧稿还引用了缺失的 `AnonymousSubmission2027_v3_arxiv_supplement.tex`。检查器
会明确报这个缺失输入；正式安装前需修正该引用或提供实际 fragment，不能静默跳过。
本次没有改写旧稿，也没有把测试图片装进论文。

旧 v3 的历史数据 CLI 默认拒绝执行；仅显式 `--allow-historical-data` 才会读取历史
结果。这些图片不被新版论文检查器接受。两个入口复用同一套原绘图函数，但使用
各自显式指定的数据来源；正式入口不会回退到旧数据目录。

## 7. 断点续跑、多进程与错误处理

继续同一 manifest 时直接执行 `run`，不重新 plan：

```bash
python -m paper_experiments.runner run \
  --manifest runs/main-p01/main.manifest.json --output runs/main-p01
```

完成的 episode 校验结果／trace 哈希后跳过。未完成的 episode 整体重跑，不会把半轮
结果算完成。请求仅对暂时网络／服务错误重试，保持原 seed 和 payload；schema 失效、
截断、模型拒绝会中止，**没有自动规则 fallback**。更改 prompt、代码、模型配置或
输出完整性会阻止错误续跑。进程被强制杀死后可能留下 `.lock`，确认原进程已退出后
仅删除相应锁再重启。

多 worker 使用同一个 manifest，不重复创建，按完整 episode 分片：

```bash
# 两个终端分别运行 index 0 与 1，避免同一 index 同时启动两次。
python -m paper_experiments.runner run --manifest runs/main-p01/main.manifest.json \
  --output runs/main-p01 --num-shards 2 --shard-index 0
python -m paper_experiments.runner run --manifest runs/main-p01/main.manifest.json \
  --output runs/main-p01 --num-shards 2 --shard-index 1
```

同一服务上的 workers 共享推理容量，总并发相加；更多 worker 不一定更快。
先用单 worker 测试。跨多台服务器时，分别建立与各 endpoint 对应的 manifest，
保留模型及环境记录；不要篡改已冻结 manifest 的 endpoint。

## 8. 本次实现的研究边界

- 规则控制器只出现在显式 rule/mixed/factorial 对照中；新的数值对照使用相同可见输入，
  不宣称重现 v3 的原策略。Watts null 是单独标识的解析参照。
- N 仅计入参与交易与传播的主体。被动 pooled liquidity 不作为额外 LLM 人口；旧设定
  的攻击者初始资金为普通 agent 的 5 倍，本版保留并需在论文报告。
- 自然语言正文由模型生成、阅读；传播只走一跳，下一跳需要下一轮 agent 明确选择。
  数值 sentiment/intensity 同样由发送者输出，**未验证与文字语义一致**，应审计。
- S3/S4 接收上一轮可见 microstructure；基本面真值、攻击标签、true wash share 和
  real-versus-wash volume 不进入模型观测。私有估值是带噪声的信号。
- S4 精确实例化 K 个攻击者，对手候选采用对称环邻居，wash 需双方相反方向的互选意图。
  wash 沿用净零往返成交的环境语义，受资金／库存和名义预算约束。自由攻击时机下，
  S4 全 horizon 计算跌幅，且只能把先前轮次的 fake liquidity 与后续跌幅组合为失败；
  不套用旧规则攻击的固定日历窗口。新策略不设置旧防御实验的 grace period。
- 从众和 deliberation 是接收方 prompt 干预，注意力是实际收件箱截断；需要检查实际
  行为。`no_feedback` 冻结公开市场观测，资产结算仍按实际价格，不能据此宣称消除了
  所有经济反馈。`no_multihop` 禁止普通 agent 发消息，是 source-only 对照。
- `sender_return_annotation_off` 只删除一项收益注释，不能关闭模型隐式学习信任。
  信息指标中的 D_N／CMI 衡量**数值社会摘要**与实际净成交的关系，不包括全部语言语义；
  语言作用应通过 content ablation 及实际行为对照另行判断。LLM 没有已知 QRE action
  probabilities，相关理论概率／熵字段明确留空。
- full text／sentiment only／neutral text 使用同一投递规则和初始条件；策略反应不同
  后，实际会话轨迹可以分叉，因此不是逐消息完全配对的反事实重放。
- collapse 中的社会级联按攻击者来源和显式 reshare 链追踪。prompt 要求改写原消息
  也引用来源，但模型未引用来源的新 post 无法自动归因。因此这是显式来源曝光指标，
  不声称覆盖全部语义传播，也不把挑战消息自动判为有害。结果保存该指标定义和
  `message_semantics_validated=false`，需要结合文本轨迹、行为及语言消融解释。

代码和脚本能支持新的实证协议；是否出现临界转变、LLM 与规则是否不同、语言是否
改变机制，必须由真实实验决定。历史探索及先前论文附录依赖见
[归档说明](legacy_experiments/README.md) 与 [原稿实验映射](legacy_experiments/final_paper_experiment_catalog.json)。
