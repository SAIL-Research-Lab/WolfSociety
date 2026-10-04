# 前三项 P0 问题：全新实验结果

## 实验范围与审计

- 所有结果均从当前 `WolfBench-main` 代码重新运行，不读取旧实验作为证据。
- 共 1,776 个 episode：native liquidity fixed-K 432 个、native liquidity boundary 576 个、feedback ablation 768 个。
- 每个 cell 使用 12 个随机种子；无失败行、无重复 key、三个实验的 12 个种子均完整。
- 全部行为机制实验采用 `behavioral_only`，population LLM calls 为 0。因此本报告能支持模拟器机制结论，不能当作真实 LLM-agent 实验证据。
- 冻结代码/配置指纹：`8e24fbe7b8ac3597`。

## 1. 原生重跑 q=0, 0.5, 1 liquidity scaling

这部分解决了“结果是否只是 post-hoc normalization 产物”的核心疑问：三个 q 都是在 market maker 原生流动性环境中独立运行的轨迹，不是对同一批输出重新归一化。

固定有害 agent 数 K、使用亚临界 episode 拟合 `R ~ N^(b_N) K^(b_K)`：

| 原生 q | b_N（95% seed-bootstrap CI） | b_K | nu_response（95% CI） |
|---:|---:|---:|---:|
| 0 | -0.764 [-0.818, -0.717] | 1.046 | 0.269 [0.141, 0.375] |
| 0.5 | -0.811 [-0.860, -0.759] | 1.022 | 0.207 [0.090, 0.305] |
| 1 | -0.822 [-0.865, -0.780] | 1.103 | 0.255 [0.164, 0.331] |

结论：三个 q 下 `b_N` 都稳定显著为负，fixed-count dilution 不是 post-hoc normalization 的假象。三个置信区间明显重叠，因此目前不能声称 q 显著改变 dilution 强度。

两规模 boundary check（N=300 与 1000）也全部成功 bracket 50% failure：

| q | alpha_c(N=300) | alpha_c(N=1000) | 两规模 nu_b（95% CI） |
|---:|---:|---:|---:|
| 0 | 0.0375 | 0.0275 | 0.258 [0.037, 0.409] |
| 0.5 | 0.0429 | 0.0275 | 0.369 [0.163, 0.486] |
| 1 | 0.0357 | 0.0275 | 0.217 [0.028, 0.337] |

这只能作为方向一致的 boundary check；由于只有两个 N，不能单独称为 universal scaling law。

## 2. Closed-loop causal ablation

四个配对条件使用 q=0.5、相同 N/alpha/seed：

- `closed_loop`：完整模型。
- `no_environment_feedback`：交易仍真实执行并决定最终市场指标，但后续 agent 只能看到配对 alpha=0 的干净市场观测；同时关闭 return-dependent social amplification。
- `no_social_propagation`：关闭 exposure、resharing 和 conformity。
- `direct_harmful_first_hop`：保留 harmful source 对直接邻居的第一跳，但切断第二跳 bot resharing 与 benign retransmission。

### 2.1 市场/environment feedback 的必要性没有得到支持

| 条件 | alpha_c(N=300) | alpha_c(N=1000) |
|---|---:|---:|
| closed_loop | 0.0429 | 0.0275 |
| no_environment_feedback | 0.0429 | 0.0267 |

切断环境反馈后临界点几乎不动。closed-loop 与 no-environment 在临界窗口中的配对差异：

- primary failure score：-0.009，95% CI [-0.035, 0.014]；
- social cascade peak：-0.005，95% CI [-0.020, 0.008]；
- price dislocation：+0.052，95% CI [0.018, 0.084]；
- liquidity stress：+0.207，95% CI [0.068, 0.335]；
- retail loss：-0.020，95% CI [-0.035, -0.007]。

因此当前实现不能支撑“harmful scaling 必须由 market/environment feedback 闭环产生”这一强说法。相反，环境反馈在 joint/social boundary 上不必要，而且在这个干预下市场压力反而更高、retail loss 更低，说明反馈路径的作用是混合且 endpoint-dependent 的。

### 2.2 多跳 social propagation 是必要机制

在临界窗口中，`no_social_propagation` 相对 closed loop：

- social cascade peak：-0.538，95% CI [-0.557, -0.521]；
- price dislocation：-0.064，95% CI [-0.100, -0.031]；
- liquidity stress：-0.254，95% CI [-0.393, -0.117]。

`direct_harmful_first_hop` 相对 closed loop：

- social cascade peak：-0.453，95% CI [-0.472, -0.435]；
- primary failure score：-0.824，95% CI [-0.860, -0.790]；
- price dislocation 与 liquidity stress 的 CI 均跨 0，不能声称显著下降。

两种传播消融的 failure curve 在所测 alpha 网格内都没有 bracket 50% failure。第一跳仍存在时大级联消失，说明结果不是“只要 harmful agent 直接发声就会失败”，第二跳及后续传播链是关键。但总体证据更接近“social cascade 驱动、市场反馈非必要”，而不是预设的 closed-loop amplification story。

重要指标审计：S1 primary score 定义为 social score 与 market score 的最小值。因此 `no_social_propagation` 下 primary score 结构性为 0，不能把它自身当作因果效应；上述判断同时依赖 first-hop 消融和独立 market/social endpoints。

## 3. nu_response 与 nu_b discrepancy

旧结果 `0.469 vs 0.222` 的巨大正向差距在新设置中没有复现：

- fresh pooled subcritical `nu_response = 0.207`，95% CI [0.090, 0.305]；
- fresh two-size boundary `nu_b = 0.369`，95% CI [0.163, 0.486]；
- fresh gap 为 -0.162，方向与旧结果相反。

按 realized severity 分层后，response exponent 明显依赖状态：

- R<0.3：nu_response=0.406，95% CI [0.174, 0.598]；
- 0.3<=R<0.6：0.028，区间极不稳定；
- 0.6<=R<1：0.270，区间较宽；
- R>=1：0.364，95% CI [0.228, 0.786]；
- pooled R<1：0.207，95% CI [0.090, 0.305]。

低严重度和超临界点估计都接近 fresh `nu_b=0.369`，而 pooled subcritical 拟合明显更低。这说明“单一 response exponent”并不稳健，混合不同响应状态会改变估计。但各层支持的 N/K cells 不同、部分区间很宽，当前只能称为 state-dependent scaling diagnostic，尚不足以宣称一个新的 crossover theorem。

论文处理建议：不要继续理论化解释旧的 `0.469-0.222` 数值残差；先报告该差距在 fresh matched setting 下不复现，再将 regime sensitivity 作为限制或新的待检验假设。若要升级成理论结果，需要增加 N、K 和近 boundary 的密集网格，预注册分层规则，并用独立数据验证 crossover。

## 最终判断

1. 第一项得到正面且较强的解决：原生 q 重跑确认 fixed-count dilution 稳健，不是归一化产物。
2. 第二项完成了真正的 in-simulation causal ablation，但结论推翻了原先最强 story：多跳社会传播必要，market/environment feedback 在当前模型中不必要。
3. 第三项发现旧 discrepancy 不可复现，并定位到明显的 regime sensitivity；这是重要纠偏，但还不是新的理论定律。

因此，当前最安全的核心表述应从“harmful scaling is an emergent closed-loop phenomenon”收缩为“harmful scaling is robust to native liquidity specifications and depends critically on multi-hop social propagation; the market-feedback contribution is endpoint-dependent and is not necessary for the observed transition in the present simulator.”
