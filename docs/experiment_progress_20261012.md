# Relation 实验进展与启动记录（2026-10-12）

用户已授权安排并启动训练。本轮按 [审核计划](relation_audit_20261012.md) 先运行负例监督的有限对照；没有把尚未完成的新 sparse 协议直接投入训练。

用户进一步要求完整流水线、顺序、缩短和RL以及最终完整效果。后续按 [正式交付路线](relation_delivery_plan_20261012.md) 推进：当前64步只是P0，随后包括清洁实体重训、3-epoch正式SFT、顺序/输入输出长度对照、OOF/规则、完整DPO/在线GRPO及三seed最终验收；尚未执行的阶段明确列为planned，不改当前冻结队列。

**14:20优先级调整：** 用户要求五六小时内取得多方向证据。grounded64已训练并验收，原全量底座评测被主动中断且保留raw；missing64移到独立新目录继续，同参数不变。已部署 [六小时有限筛选](rapid_screen_20261012.md)，包含固定12篇评测、matched256顺序/短输出训练和受控DPO，20:20前汇总。原130条完整评测仍未完成，不能把本轮子集结果填入其成绩。

## 当前安排

| 阶段 | 状态 / 目的 |
|---|---|
| 迁移、数据和指标审核 | 已完成并提交 `0789e80`；原封存数据不变。 |
| V4 数据发布 | 从冻结 prepared records 以 unique-evidence-v1 重建；歧义证据维持 UNKNOWN。 |
| 独立训练环境 | 已完成；Python3.12 + torch2.8.0/cu128 + transformers4.56.2 + peft0.17.1；pip check及GPU BF16前后向通过，与现有 LF 环境隔离。 |
| 2-step smoke | 已用4条完全合成样例完成；权重确有更新，重新加载的权重指纹一致，首个prompt末token logits差为0。 |
| V4_grounded_empty | 已启动首个64-step试点：原文外干扰负例 + 构造文本中无支持关系的负例。实际优化进度见下方启动快照。 |
| 裸底座0-step tune | 同模型加载器、相同候选/模板/解码；作为新服务器基线。 |
| grounded tune / stress | 完整130条 tune 和64对压力样例；不省略失败样本。 |
| V4_missing_only | 第二个64-step对照：同输入和正目标，仅屏蔽原文内构造负例的 loss。 |
| missing tune / stress | 与第一组相同评测输入、预算、口径。 |
| sparse / 全负批 / 长度排序 / 实体 OOF / RL | 待本轮诊断和数据仲裁后逐项开展；本次队列不自动扩展到这些实验。 |

## 本轮控制变量

- 两组使用同一 Qwen3-8B 底座、seed42、LoRA r16/alpha32/dropout0.05、BF16、lr5e-5、batch1×梯度累计16、6144 tokens、64 optimizer steps。仅改变构造的可靠 grounded-empty decisions 是否参与 loss。
- 不输出字面 NONE 关系；本轮仍是 dense fixed-pair 协议，空关系用 `relations:[]`。这不等于已实现“只输出正关系”的 sparse 协议。
- 银标漏标 UNKNOWN 不变，普通未标注对不会自动变成负例。512 个构造样例仅来自训练来源组，不借用 tune/stress 词库。
- 记录各臂正/负/UNKNOWN、有效监督 tokens、实际采样曝光与耗时。同输入/同步数并不意味着同监督 token 预算。
- 新的唯一证据策略减少可用监督，不能把 V4 对旧 V2 的差异全归因于负例；本轮主要因果对照是 V4 两臂之间。

## 评测与执行约束

完整 tune130 包含69条 free_text、61条 ICD，分别报告。tune 使用 `--oracle --strict-identity`，不传旧 entity cache；它是关系阶段诊断，不是实时端到端性能。64对压力集共128 prompts，报告错连边及共同正关系保留，不能作为挑 checkpoint 的集合。148条回归暂不进入选模流程。

训练成功后按队列自动评测，保存原始生成、图输出和 TP/FP/FN/P/R/F1。旧压力集虽可对照，但与真实 NER 错误不同，不据此单独宣称医学幻觉已经解决。每臂只保存/评测预先指定的64-step checkpoint，不按训练 loss 挑选。

GPU 串行执行；新目录写入，输入和代码绑定 SHA256，进程持有设备锁；任一步失败即停止队列并记录错误。queue `status.json`、各阶段日志及训练 `run_manifest.json` 是运行事实来源。训练在服务器持久会话运行，关闭本地终端不终止作业。

部分监督需要本仓库 `generative.train`。不能把本轮 JSONL 直接导入普通 LLaMA-Factory SFT，否则 UNKNOWN mask 会丢失。环境依赖记录在 [requirements-relation.txt](../requirements-relation.txt)。

## 启动验证

本地代码检查：873 项测试通过。新数据每臂 1835 行（1323 条严格 base + 512 条构造），正例对5639、正边5659；missing/grounded 负例对2508/3908、UNKNOWN16683/15283。最大4932 tokens，真实 tokenizer/mask 检查通过。

- 数据聚合与哈希：[pilot-data-manifest.json](audits/20261012/pilot-data-manifest.json)。
- 冻结队列配置：[relation_pilot_20261012.json](../configs/experiments/relation_pilot_20261012.json)。路径按授权服务器目录布局填写，重定位时必须重新验证。
- 数据上传已获用户明确授权并完成。服务器端归档哈希、全部训练/评测输入及代码哈希验证通过，七阶段 dry-run 通过；私有医学文本和标注仅保留在本地私有工作目录与授权服务器，不随代码提交。
- smoke 验收：[pilot-smoke-summary.json](audits/20261012/pilot-smoke-summary.json)。2步训练耗时4.53秒，峰值分配显存约19.12GiB；这只是短合成样例的工程验证，不能估算长医学文本吞吐或质量。
- 正式试点已于北京时间 2026-10-12 13:24:55 启动，独立 tmux 会话为 `medicalner-relation-pilot`。启动前确认 GPU 无活动计算进程，空闲显存占用4MiB，无需重启 WebUI。
- 队列运行目录为 `/workspace/medicalner/20261012/pilot`；`status.json`、GPU占用与训练日志共同用于确认实际进度。运行时输入/代码保持冻结，GitHub 后续进度记录不会改动正在运行的代码。
- 启动快照（北京时间13:26:44）：`grounded_s42-train` 正在运行，已观察到首个优化步骤，loss=0.3721、grad_norm=1.3953；GPU利用率100%，当时显存31380MiB。首条日志的学习率为0是warmup起点，最终权重变化仍须由训练结束时的权重指纹验证，不能用此条loss代替验收。见 [pilot-start-summary.json](audits/20261012/pilot-start-summary.json)。
- 这是一份启动时快照，后续实时状态以服务器文件为准。两组完整训练及质量评测尚未结束，当前没有新的准确率/F1结论。
