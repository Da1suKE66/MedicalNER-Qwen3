# 关系标签 token 加权 SFT：隔离候选

本目录只准备候选；是否排入队列由主任务决定，不改既有10个任务。

完整1000行训练数据共有399,973个监督 token，关系标签占6.94%，实体名称和证据合计84.70%。这只是 token 数量统计，没有测得各部分实际交叉熵或梯度份额，不能据此认定复制任务导致主要误差。

候选从原最佳关系 SFT 父 adapter 重新加载，完整继续1个 epoch，LR2e-5、rank16、alpha32、dropout0.05、batch1、accum8、seed42、最大8192 token。关系标签字段内 token 权重为4，其他全部目标 token（含EOS）权重为1；prompt和padding为0。原文、实体、完整语义标签输出、所有目标和EOS不删减，不改JSON组装、解析或评分。预检最大输入5140 token，无截断。4倍权重后标签占总加权 token 质量22.97%。

损失按整个梯度累积组计算 `sum(weight * causal_CE) / sum(weight)`，保留长短样本的 token 归一化语义。10项本地CPU测试通过；主任务另在服务器CPU实际运行真实Torch与HF Trainer回归，三个不同长度microbatch、accum2、最后不足accum组均通过，SGD参数与显式参考最大差为0。本地审查环境没有Torch，没有声称本地重跑CUDA。

执行环境必须设置 `PYTHONPATH=/workspace/medicalner/20261019/code_eval_v6/agent`。部署清单见 `source_free_export_sha256.json`，参数见 `training_config.json`。使用服务器Python运行 `run_smoke_then_train.py --config training_config.json --deadline-unix <截止UTC秒数>`。包装器先核验代码、数据、父adapter文件/协议/base配置和manifest，再运行CPU预检与真实Torch数学回归。仅传脚本和汇总文件，训练数据从服务器已驻留的冻结1000行文件读取。

smoke只对全局最长样本做1个GPU更新，独立输出、accum1、warmup0，避免唯一一步学习率为0。正式训练另启进程，从原SFT父模型重新加载，warmup0.05，125个更新，所有1000行各曝光一次；smoke权重绝不作为正式父模型。初始tensor指纹须精确匹配原SFT，保存后核验内存与磁盘bitwise相等，并将adapter重新载入新槽核验参数；后者复用同一个未改动的base，不宣称重新加载了整个base。输出存在即拒绝覆盖。

开始至少需要3000秒余量，完整130文档评测按1800秒保守预算。新训练含smoke/loading暂估18–20分钟，尚未实测加权损失的耗时；smoke与正式训练共同硬超时1500秒，正式单段最多1320秒。超时或任何预检失败都记录失败，不能把部分训练记为成功。评测必须复用原基线实际实体输出，核对逐样本prompt、gold及评分实现；未完成全130评测前不得报告提升。

这是额外训练量、LR及损失权重同时改变的探索；缺少同父模型、同LR、同epoch的无加权续训对照，不能把任何增益单独归因于标签加权。完整数据仍含已知source188的一行问题监督，标签不平衡、未标注关系与实体漏抽也没有由本候选修复。继续使用的130文档是已反复查看的tune集，不是新的盲测集。若时间或检查不满足条件，应放弃本候选。
