# MedicalNER-Qwen3 实验台账

整理日期：2026-09-28。本台账汇总本地原始归档，整理本身不产生新训练或模型成绩。完整机器JSON保存在私有工作区，不随代码分支上传。随后主线程启动的新实验见[本轮进度](progress_20260928.md)；本文仍保留各来源的时间边界。

## 1. 时间边界、状态和计数

数值主归档来自 9 月 8 日历史阶段及 9 月 16 日有限队列结果，最近一次既有本地状态复核归档为 **9 月 23 日 16:09（北京时间）**。G0 固定实体补评测已在 9 月 15 日 11:48 报告中确认完成。台账任务没有独立核验 9 月 28 日远程状态。

主线程于 **9 月 28 日 03:44（北京时间）**核对四 GPU 空闲、旧 13 项终态与 9/16 一致、GitHub 分支未变。此为**主线程提供的信息**，不是台账子任务的独立远程核验。DPO no-hard 的残留 child=running 不覆盖父 queue=failed；有效状态为评测失败。新四项已于04:02左右完成GPU smoke并进入正式训练；最新状态以同目录REPORT.md及状态快照为准。

| 阶段/表 | 本台账行数或状态 | 不能怎样计数 |
| --- | --- | --- |
| 9/15有限队列 | 13逻辑项：11完整完成、1评测失败、1未启动；12项训练完成 | 11 done不是全项目总完成数 |
| 主比较表 | 18行＝16生成式逻辑项＋同一classifier的2种阈值策略 | classifier不是训练了2次；其中含历史G0/G1/G6 |
| 历史阶段表 | 8行（旧单模型缓存＋v1–v7）；v1–v7归档11个阶段训练checkpoint | 版本行/实体与关系checkpoint不混为同一单位 |
| 早期上下文记录 | 7行，部分仅有9/23概要证据 | 数据治理/格式检查不是新训练 |
| 附表 | 41生成式内部checkpoint、20分类阈值、63规则重评分、7压力测试、30历史缓存 | 互相引用，禁止相加成训练总数 |
| 9/28新计划 | 4项预启动登记快照；主线程后报04:02已启动，尚无训练结果，done=0 | 不并入旧13项，不将预算/smoke通过写成正式训练完成 |

全项目独立训练总数：**unknown**。本台账覆盖所引本地归档可追溯的配置、评测与准备项；不以旧概要或缓存文件数制造全项目总数，也不推断未归档实验不存在。

## 2. 评分与缺失值约定

`P=TP/(TP+FP)`，`R=TP/(TP+FN)`，`F1=2TP/(2TP+FP+FN)`。下列 P/R/F1 均为百分数，后接 TP/FP/FN；是文档去重有向三元组的 micro 汇总，不平均分片 F1。实体 ID/数组顺序不决定匹配；证据另算。所有分数基于教师银标，FP 不等于医学上已确认的幻觉。

| 层 | 实际范围 | 解释 |
| --- | --- | --- |
| 内部生成式 | 130记录中的69自由文本；573关系；oracle输入 | 用于F0.5等选模；14个已选生成式模型均未达P≥50%门槛，采用明确回退 |
| 固定预测实体 | 完整148记录中81自由文本；638关系；旧v4缓存 | 不是实时entity→relation完整链路 |
| Oracle | 同一81自由文本；638关系；标签实体 | 仍有grounding/候选损失，不是无条件relation上限 |
| Classifier内部 | 5714已覆盖候选对；383正关系决策 | 不含候选外FN，与生成式内部整图指标不等价 |
| ICD | 回归67记录；66关系；确定性分支 | 单列，不掺入自由文本改善宣传 |
| 压力测试 | 64 clean/noisy配对；按prompt pair ID+relation评分 | 不是实体名称图谱F1；clean47/noisy41，分母不同 |

`unknown` 表示未找到充分证据，绝不填0。`undefined` 表示数学分母为0；源文件原reported空集约定保留在JSON。旧v2的0/0/0却100%为无效成绩。登记步数、实际全程步数、选中步数、正文字符数、训练上下文上限、每批候选数、生成上限分别记录，不能互相替代。

## 3. 主比较表（18行）

每个指标格为“P/R/F1%；TP/FP/FN”。历史项与本轮项同表便于检索，但不暗示同预算。classifier两行只作历史对照，非当前生成式主线。

| 实验 | 阶段 | 总训/选中步 | 内部 | 固定预测实体 | Oracle | 完成边界 |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | 9/8历史 | 279/279 | 42.11/58.64/49.02；336/462/237 | 12.71/11.60/12.13；74/508/564 | 33.71/56.27/42.16；359/706/279 | 历史完成 |
| G1_data | 9/8历史 | 279/279 | 42.11/57.24/48.52；328/451/245 | 12.16/11.76/11.95；75/542/563 | 33.21/55.49/41.55；354/712/284 | 历史完成 |
| G2_short_batch | 9/15 | 279/279 | 28.88/49.74/36.54；285/702/288 | 9.59/11.13/10.30；71/669/567 | 23.03/44.51/30.36；284/949/354 | 完成 |
| G3_order | 9/15 | 279/279 | 27.24/54.10/36.24；310/828/263 | 7.58/10.34/8.75；66/805/572 | 19.79/50.47/28.43；322/1305/316 | 完成 |
| G4_multi | 9/15 | 279/279 | 37.25/54.80/44.35；314/529/259 | 10.61/11.44/11.01；73/615/565 | 32.00/57.68/41.16；368/782/270 | 完成 |
| G5_local | 9/15 | 0/unknown | unknown | unknown | unknown | 未启动 |
| G6_all | 9/8历史 | 933/311 | 48.88/41.71/45.01；239/250/334 | 17.79/9.09/12.03；58/268/580 | 45.07/42.95/43.98；274/334/364 | 历史完成 |
| G7_disease_comma | 9/15 | 279/279 | 43.06/62.83/51.10；360/476/213 | 11.67/12.07/11.86；77/583/561 | 33.39/60.50/43.03；386/770/252 | 完成 |
| N0 | 9/15 | 279/279 | 41.91/57.42/48.45；329/456/244 | 11.67/11.60/11.64；74/560/564 | 34.01/55.17/42.08；352/683/286 | 完成 |
| N10 | 9/15 | 279/279 | 39.60/55.85/46.34；320/488/253 | 10.86/11.13/10.99；71/583/567 | 32.11/56.27/40.89；359/759/279 | 完成 |
| N25 | 9/15 | 279/279 | 40.97/56.20/47.39；322/464/251 | 11.22/11.13/11.17；71/562/567 | 33.52/56.43/42.06；360/714/278 | 完成 |
| DPO_b005 | 9/15 | 64/32 | 22.32/64.05/33.11；367/1277/206 | 6.68/13.17/8.86；84/1174/554 | 19.06/64.42/29.42；411/1745/227 | 完成 |
| DPO_b010 | 9/15 | 64/32 | 16.87/65.79/26.85；377/1858/196 | 5.30/14.42/7.75；92/1645/546 | 14.60/67.24/23.99；429/2509/209 | 完成 |
| DPO_no_hard_b010 | 9/15 | 64/unknown | unknown | unknown | unknown | 训完/评测失败 |
| GRPO_e02 | 9/15 | 384/384 | 35.55/57.94/44.06；332/602/241 | 10.42/12.07/11.18；77/662/561 | 29.30/57.68/38.86；368/888/270 | 完成 |
| GRPO_e0 | 9/15 | 384/384 | 37.41/57.59/45.36；330/552/243 | 10.81/11.91/11.33；76/627/562 | 30.44/57.21/39.74；365/834/273 | 完成 |
| classifier_global | 9/7分类头 | 1282/1282 | 65.98/67.36/66.67；258/133/125 | 27.27/7.52/11.79；48/128/590 | 61.76/34.17/44.00；218/135/420 | 完成/历史对照 |
| classifier_type_thresholds | 9/7分类头 | 1282/1282 | 82.13/56.40/66.87；216/47/167 | 30.94/8.78/13.68；56/125/582 | unknown | 完成/历史对照 |

G5 的0步来自明确“未启动”，登记预算279步未执行。DPO no-hard 完整 tune32=396/4658/177，P/R/F1=7.84/69.11/14.08%，仅见附表，**不是选中模型或最终回归**；tune64 超时后的63/69篇、661批部分输出不凑成完整成绩。G0固定回归由9/15补完，沿用旧checkpoint279，不是新13项中的第14项。

ICD分层：上述有完整回归的生成式模型及classifier各条件均为 **54/2/12，P/R/F1=96.43/81.82/88.52%**。生成式内部ICD为57/11/26（61条），不能与67条回归ICD混用。新4项未出结果，不继承这些分数当新成绩。

### 3.1 数据版本、候选与上下文

SFT旧版与9/15均来自1345条DeepSeek命名教师银标（606 ICD＋739自由文本），内部训练/验证545/61来源组；神经自由文本train/tune为670/69条。旧Schema/Gemini858不是这一数据版。9/15全部实验未使用9/16的v2修复候选。训练输入SHA已读本地文件核对，完整SHA和来源路径在JSON。

| 实验 | 训练版本；行数；SHA前12位 | 候选/批 | 上下文策略；训练token上限；数据实测最长 | 推理生成上限 | 数据/参数来源 |
| --- | --- | --- | --- | --- | --- |
| G0 | generative_20260908/data/G0；1475；6d0ebe5b113b | 16 | full；6144；5032 | 16000 | manifest_old_G0, running_md |
| G1_data | generative_20260908/data/G1_data；1475；e7fb234f1258 | 16 | full；6144；unknown | 16000 | manifest_old_G1_data, running_md |
| G2_short_batch | data_final_v1/G2_short_batch；4971；8c79028ea8ff | 4 | full；6144；3296 | 16000 | queue_A_G_r2, manifest_G2_short_batch |
| G3_order | data_final_v1/G3_order；1475；1b7301aa96af | 16 | full；6144；5171 | 16000 | queue_A_G_r2, manifest_G3_order |
| G4_multi | data_final_v1/G4_multi；1475；85d3a928c787 | 16 | full；6144；5031 | 16000 | queue_A_G_r2, manifest_G4_multi |
| G5_local | data_final_v1/G5_local；1353；58fb8a40febd | 16 | local；6144；5031 | 16000 | queue_B_DPO_r3, manifest_G5_local |
| G6_all | generative_20260908/data/G6_all；4971；865aa53e481a | 4 | full；6144；3234 | 16000 | manifest_old_G6_all, running_md |
| G7_disease_comma | data_final_v1/G7_disease_comma；1496；9727630f038e | 16 | full；6144；5031 | 16000 | queue_D_noise_r2, manifest_G7_disease_comma |
| N0 | data_final_v1/G1_data；1475；f09be9dae22a | 16 | full；6144；5031 | 16000 | queue_D_noise_r2, manifest_G1_data |
| N10 | noise_final_v1/noise_1_of_10；1639；5c104afe4e3e | 16 | full；6144；5031 | 16000 | queue_D_noise_r2, manifest_G1_data |
| N25 | noise_final_v1/noise_1_of_4；1967；9431e26c6c16 | 16 | full；6144；5031 | 16000 | queue_D_noise_r2, manifest_G1_data |
| DPO_b005 | preferences_g1_final_v1/full.jsonl；512；94da46469c5a | unknown | full；6144；4969 | 16000 | queue_B_DPO_r3, manifest_G1_data, prefs |
| DPO_b010 | preferences_g1_final_v1/full.jsonl；512；94da46469c5a | unknown | full；6144；4969 | 16000 | queue_B_DPO_r3, manifest_G1_data, prefs |
| DPO_no_hard_b010 | preferences_g1_final_v1/no_hard_semantic.jsonl；512；39c0fb5f4fea | unknown | full；6144；5011 | 16000 | queue_B_DPO_r3, manifest_G1_data, prefs |
| GRPO_e02 | data_final_v1/G1_data；1475；f09be9dae22a | 16 | full；8192；unknown | 16000 | queue_C_GRPO_r3, manifest_G1_data |
| GRPO_e0 | data_final_v1/G1_data；1475；f09be9dae22a | 16 | full；8192；unknown | 16000 | queue_C_GRPO_r3, manifest_G1_data |
| classifier_global | pair_v8 / DeepSeek冻结银标未修词版；10246；89babe7b298d | 1 | 局部pair上下文；768；unknown | not_applicable | classifier_manifest |
| classifier_type_thresholds | pair_v8 / DeepSeek冻结银标未修词版；10246；89babe7b298d | 1 | 局部pair上下文；768；unknown | not_applicable | classifier_manifest |

SFT共同配置：Qwen3-8B、LoRA r16/alpha32、lr5e-5、有效batch16、seed42。DPO lr5e-6，beta=.05/.1；GRPO lr2e-6、KL=.04、G=4、rollout上限1536，两臂正式训练上限8192。RL继承历史G1的279步SFT，不是从零与279步SFT等成本比较，也不是DPO→GRPO串联。classifier为r8/alpha16、2epochs/1282步、长度768、局部距离≤2 spans/正文≤2200字符；无生成token预算。其余正文字符上限、历史实测prompt最长如无直接证据均为unknown，不从当前代码默认值反推。

N0在data_final_v1/G1_data上重新SFT，不是旧G1重命名。N10/N25分别保留1475干净行并追加164/492副本，训练总行1639/1967；约10%/25%指最终序列混合比例。G4的1475行按权重抽样，weighted_rows1662；历史G6为4971行、weighted_rows5171。G2正关系决策4781 vs N0 4779，分批也改变少量可保留监督。

| 实验 | 固定候选对；覆盖/638 | Oracle候选对；覆盖/638 | 评测完整性说明 |
| --- | --- | --- | --- |
| G0 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| G1_data | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| G2_short_batch | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| G3_order | 8050对；120/638 | 14272对；546/638 | 固定/Oracle各1批截断 |
| G4_multi | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| G5_local | unknown | unknown | 无完整回归 |
| G6_all | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| G7_disease_comma | 8050对；120/638 | 14344对；573/638 | Oracle批次926；与普通prompt不同 |
| N0 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| N10 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| N25 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| DPO_b005 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| DPO_b010 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| DPO_no_hard_b010 | unknown | unknown | 无完整回归 |
| GRPO_e02 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| GRPO_e0 | 8050对；120/638 | 14272对；546/638 | 完整回归；不代表原始协议零错 |
| classifier_global | 3988对；106/638 | 6328对；444/638 | 局部候选；构建/评测口径差异见D07 |
| classifier_type_thresholds | 3988对；106/638 | unknown | 局部候选；构建/评测口径差异见D07 |

普通固定候选8050对、仅覆盖120/638=18.81%；普通oracle14272对、覆盖546/638=85.58%。G7 oracle14344对、覆盖573/638=89.81%；920共用批中26个prompt变动，另增6批。G7的43.03%不能全归因模型。classifier完整评测候选3988/6328、覆盖106/444，不能拿其构建阶段408/638或生成式546/638混作一个上限。

### 3.2 真实曝光、问题与下一步价值

曝光是全程训练消耗；DPO总训64但选32，不能把64步全部token当选中模型前缀曝光。prompt曝光只是前向上下文，不代表进入SFT目标loss。旧G0/G1/G6没有在本台账找到可同口径引用的实际曝光账，保留unknown。

| 实验 | 序列/偏好/rollout曝光 | SFT实体对曝光 | prompt token | completion/rollout token | 预算/问题 | 下步价值 |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | unknown | unknown | unknown | unknown | G0/G1均279步、1475行；可比性较强但不证明严格同token曝光；与旧v4输出预算及训练不同；9/8固定回归中断，9/15同checkpoint补齐；不是本轮13项新增训练；不能沿用旧缺失状态 | 保留已补齐的三路基线与规则对照，供后续严格配对复验 |
| G1_data | unknown | unknown | unknown | unknown | G0/G1均279步、1475行；可比性较强但不证明严格同token曝光；与旧v4输出预算及训练不同；有限修词不等于v2语义审定；oracle未超过G0；N0是另一次训练而非此模型重命名 | 作为RL固定SFT起点；先补同加载器零步和审定负例 |
| G2_short_batch | 4464 | 17702 | 8963765 | 334567 | 279步但仅17702实体对曝光，N0为59088（29.96%）；数据正关系决策4781 vs4779，输入/监督集合也不完全相同，不是等token/pair单变量对照；279步只曝光17702对，N0为59088；不是等token/pair预算长度对照。 | 高价值是等pair/token预算复验，不能用当前结果断言缩短输出无效。 |
| G3_order | 4425 | 59088 | 12668856 | 1066551 | 279步且pair曝光59088与N0同；prompt token12668856 vs12847548、completion1066551 vs1066545，不严格同token；固定/oracle各1次截断；预测/oracle各1次截断；排序并未解决长输出问题。 | 中等：保留顺序消融，但先控制输出循环和重复种子；排序本身没有消除截断。 |
| G4_multi | 4425 | 60411 | 13019038 | 1095461 | 279步但重采样使pair曝光60411 vs N0 59088；不是等标签/token曝光；多标签输出已是共同能力；重加权而非首次支持多标签；新SFT本就支持多关系。 | 中等：建立真正多关系挑战集，比较重采样而非声称首次支持多标签。 |
| G5_local | unknown | unknown | unknown | unknown | DPO父队列失败，未启动；无本轮成绩。；DPO父队列失败，未启动；无本轮成绩。 | 待新授权且先审上下文覆盖；不能把预检当训练完成。 |
| G6_all | unknown | unknown | unknown | unknown | 不等预算：933总步/311选中步；G0/G1为279；同时改3因素；token/pair实际曝光unknown；组合差异无法归因到短批/顺序/重采样；不能因最高oracle就声称等预算胜出 | 保持组合历史结果；补等pair/token预算单项、去项及重复种子 |
| G7_disease_comma | 4448 | 59602 | 12910732 | 1076777 | 279步，oracle候选546→573，普通14272对→14344；共用920批中26prompt改变、另增6批；模型与输入改动混合；oracle候选覆盖546→573，同时prompt改变；不能把增益全归模型。 | 高：拆分候选/定位与模型效果，做同候选同prompt对照并审别名。 |
| N0 | 4425 | 59088 | 12847548 | 1066545 | 279步clean参考；与N10/N25步数同但epoch、序列/pair/token曝光不同；与9/8 G1为不同训练和文件SHA；不加合成噪声，不代表医学金标；固定实体候选覆盖低。 | 高：保留新轮基线，与旧G1分开；在审定v2上建立新对照。 |
| N10 | 4446 | 59537 | 12924523 | 1073589 | 279步匹配N0，但序列/实体对/token曝光不相同；噪声比例是最终序列比例，不是实体/token比例；错误实体误连少2/151，但固定回归F1较N0低；只为易合成噪声。 | 中等：真实错误实体审定与out-of-fold负例优先；合成误连减少需更大样本确认。 |
| N25 | 4462 | 59799 | 13006325 | 1071890 | 279步匹配N0，但序列/实体对/token曝光不相同；噪声比例是最终序列比例，不是实体/token比例；未超过N0，合成实体错分布不等于真实错误分布。 | 较低：未优于N0，先研究真实噪声分布，不直接继续提高比例。 |
| DPO_b005 | 1024 | not_applicable | 3898812 | 351770 | 两beta组同512偏好、64总步、1024偏好曝光及token；均选32步；继承279步SFT，不能与279步新SFT称等预算；增加召回并大幅增加银标FP；64步训练，但选中32步。 | 较低：先审偏好及过度连边，再考虑继续RL。 |
| DPO_b010 | 1024 | not_applicable | 3898812 | 351770 | 两beta组同512偏好、64总步、1024偏好曝光及token；均选32步；继承279步SFT，不能与279步新SFT称等预算；较beta .05更多过度连边；64步训练，但选中32步。 | 较低：beta加大使银标FP更多，不支持直接扩算力。 |
| DPO_no_hard_b010 | 1024 | not_applicable | 6271676 | 508582 | 512行/64总步虽与full同，prompt6271676 vs3898812、completion508582 vs351770不等；来源组/偏好分布也不同；64步训练完成，tune64 14400秒超时；无完整选模/回归/压力。 | 仅在新授权下恢复；先诊断重复输出，保留失败而非以tune32替代最终分数。 |
| GRPO_e02 | 6144 | not_applicable | unknown | 1489181 | 两臂同384步/6144rollout；completion1489181 vs1483091不严格等token；继承279步SFT，与纯SFT不等预算，缺零步加载对照；真实在线更新但未超过历史G1；没有同加载器零步对照。 | 先补同加载器零步与重复seed、改奖励可靠性，再判断RL收益。 |
| GRPO_e0 | 6144 | not_applicable | unknown | 1483091 | 两臂同384步/6144rollout；completion1489181 vs1483091不严格等token；继承279步SFT，与纯SFT不等预算，缺零步加载对照；比e0.2少54个oracle FP，但仍比历史G1多122；抗错误实体未改善。 | 先补同加载器零步与真实负例；去掉证据奖励未降低错误实体关联误连。 |
| classifier_global | unknown | unknown | unknown | unknown | 与生成式不同数据、候选、长度768、1282步、损失和输出形式；非等预算主线消融；内部候选precision不可冒充整图精度；分类别策略拒绝两种无合格阈值关系以牺牲召回；8个合成挑战4正例全漏；事后规则不算独立盲测 | 仅保留历史参照和错误归因；不据此擅自切换用户选定生成式主线。 |
| classifier_type_thresholds | unknown | unknown | unknown | unknown | 与生成式不同数据、候选、长度768、1282步、损失和输出形式；非等预算主线消融；内部候选precision不可冒充整图精度；分类别策略拒绝两种无合格阈值关系以牺牲召回；8个合成挑战4正例全漏；事后规则不算独立盲测 | 仅保留历史参照和错误归因；不据此擅自切换用户选定生成式主线。 |

GRPO两组各6144在线rollout，共12288次、2972272 completion token，非接口/离线伪rollout；但loss接近0不等于精度接近100%。最新e0 oracle=365/834/273，相比e0.2少3TP、少54FP，F1仅+0.88百分点；相对历史G1多11TP、多122FP，F1−1.81百分点。两组均缺同加载器零步与重复种子。

## 4. 新4项登记快照：尚无训练结果（done=0）

**最新主线程通知（用户转达，04:02北京时间）**：四项均已启动正式训练，2步GPU smoke全部通过，weights_changed=true，独立保存重载logit差0.0。本任务没有独立远程核验，也不等待新结果。下表的“准备”是上一消息冻结的预启动数据/参数快照，不代表04:02之后仍未启动；详细launch状态由主线程维护。smoke通过不作为正式训练完成或P/R/F1结果。

来源：continuation_plan、controlled_manifest、controlled_matrix以及用户转达的主线程更新。manifest明确`new_model_results_available=false`、真实分词通过、部分监督不适用于现有RL。计划开始前记621 tests；主线程后续报告633 tests通过，时间/范围分开，本台账没有执行这两套测试。

| 实验 | 训练行/正标签 | 登记步/保存间隔 | 候选；上下文上限；实测最长 | 干预与匹配条件 | 内部/固定/Oracle | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| V2_base | 1333/4785 | 252/84 | 16；full/6144；4932 | 修复v2基础/UNKNOWN掩码；base/noise同1333行/4785正标签/252计划步；正例证据曝光一致；新增负监督、输入token不同 | unknown / unknown / unknown | 准备，done=0 |
| V2_noise10 | 1333/4785 | 252/84 | 16；full/6144；4945 | 135行注噪版本替换原行；不追加重复正例；增加270个合成negative；base/noise同1333行/4785正标签/252计划步；正例证据曝光一致；新增负监督、输入token不同 | unknown / unknown / unknown | 准备，done=0 |
| V2_full_common | 1322/4606 | 249/83 | 16；full/6144；4932 | 按local有监督行/完整目标恢复full可见原文；屏蔽full-only正例；full/local同1322行/4606标签/249计划步；目标与mask逐行相同，但输入token不相同 | unknown / unknown / unknown | 准备，done=0 |
| V2_G5_local_common | 1322/4606 | 249/83 | 16；local/6144；4893 | 与full_common逐行相同实体/候选/completion/loss mask/顺序，只改变可见原文；full/local同1322行/4606标签/249计划步；目标与mask逐行相同，但输入token不相同 | unknown / unknown / unknown | 准备，done=0 |

V2_noise10是替换135/1333行（约10.13%），不是旧N10的追加混合；增加270个合成negative，正例/证据曝光与base一致。full/local共同对照的实体表、候选表、completion、loss mask、顺序与行数逐行相同；不是仅凭最终标签总数相同宣称匹配。训练标签4785→4606少179，内部482→472少10，代价必须保留。四项只有各自配对内计划步数相同，跨配对252/249不同。

| 实验 | 数据单遍prompt token | 单遍completion token | 监督/屏蔽completion | 说明 |
| --- | --- | --- | --- | --- |
| V2_base | 4036892 | 343416 | 157371/186045 | 这是预检数据单遍计数，不是实际训练曝光 |
| V2_noise10 | 4060288 | 346926 | 160785/186141 | 这是预检数据单遍计数，不是实际训练曝光 |
| V2_full_common | 4001150 | 336669 | 150183/186486 | 这是预检数据单遍计数，不是实际训练曝光 |
| V2_G5_local_common | 3199574 | 336669 | 150183/186486 | 这是预检数据单遍计数，不是实际训练曝光 |

新4项保留旧实体缓存，未来仍为relation条件评测；不能继承历史G5失败的身份，也不能把新局部窗口准备好写成旧G5 done。后续smoke、launch、训练步数、选模与回归由主线程另行更新。

## 5. 规则消融（63个重评分行，不是63次训练）

规则继承对应模型的数据、选中步数、候选和上下文，不做额外训练。下表含G0/G1/G6三轮内部验证及固定/oracle、classifier事后类型规则和旧v4窄规则。新13项最终聚合未给逐规则计数，记unknown；不据此断言远程从未评过规则。

规则名：`all_rules`全部启用；`different_disease_in_evidence`证据疾病差异；`enumerated_siblings`并列项；`symptom_disease_type_mismatch`症状疾病类型约束。独立规则与all_rules不是可简单相加的效果。每行P/R/F1、TP/FP/FN均重算；删除TP/FP由归档保留，旧v4一行为报告删除数推导。

| 模型/评测 | 规则 | 规则后P/R/F1%；TP/FP/FN | 删TP/FP | F1变化（百分点） |
| --- | --- | --- | --- | --- |
| classifier_global/predicted | type_rules_posthoc | 28.40/7.52/11.90；48/121/590 | 0/7 | 0.10 |
| classifier_type_thresholds/predicted | type_rules_posthoc | 32.75/8.78/13.84；56/115/582 | 0/10 | 0.17 |
| G0/oracle | all_rules | 35.17/56.11/43.24；358/660/280 | 1/46 | 1.08 |
| G0/oracle | different_disease_in_evidence | 34.69/56.11/42.87；358/674/280 | 1/32 | 0.71 |
| G0/oracle | enumerated_siblings | 33.71/56.27/42.16；359/706/279 | 0/0 | 0.00 |
| G0/oracle | symptom_disease_type_mismatch | 34.16/56.27/42.51；359/692/279 | 0/14 | 0.35 |
| G0/predicted | all_rules | 14.29/11.60/12.80；74/444/564 | 0/64 | 0.67 |
| G0/predicted | different_disease_in_evidence | 13.58/11.60/12.51；74/471/564 | 0/37 | 0.38 |
| G0/predicted | enumerated_siblings | 12.71/11.60/12.13；74/508/564 | 0/0 | 0.00 |
| G0/predicted | symptom_disease_type_mismatch | 13.33/11.60/12.41；74/481/564 | 0/27 | 0.27 |
| G0/tune_186 | all_rules | 41.82/56.20/47.95；322/448/251 | 0/23 | 0.81 |
| G0/tune_186 | different_disease_in_evidence | 41.71/56.20/47.88；322/450/251 | 0/21 | 0.74 |
| G0/tune_186 | enumerated_siblings | 40.61/56.20/47.14；322/471/251 | 0/0 | 0.00 |
| G0/tune_186 | symptom_disease_type_mismatch | 40.71/56.20/47.21；322/469/251 | 0/2 | 0.07 |
| G0/tune_279 | all_rules | 43.30/58.64/49.81；336/440/237 | 0/22 | 0.80 |
| G0/tune_279 | different_disease_in_evidence | 43.19/58.64/49.74；336/442/237 | 0/20 | 0.73 |
| G0/tune_279 | enumerated_siblings | 42.11/58.64/49.02；336/462/237 | 0/0 | 0.00 |
| G0/tune_279 | symptom_disease_type_mismatch | 42.21/58.64/49.09；336/460/237 | 0/2 | 0.07 |
| G0/tune_93 | all_rules | 33.21/45.38/38.35；260/523/313 | 0/42 | 1.15 |
| G0/tune_93 | different_disease_in_evidence | 32.75/45.38/38.04；260/534/313 | 0/31 | 0.84 |
| G0/tune_93 | enumerated_siblings | 31.52/45.38/37.20；260/565/313 | 0/0 | 0.00 |
| G0/tune_93 | symptom_disease_type_mismatch | 31.94/45.38/37.49；260/554/313 | 0/11 | 0.29 |
| G1_data/oracle | all_rules | 34.85/55.33/42.76；353/660/285 | 1/52 | 1.21 |
| G1_data/oracle | different_disease_in_evidence | 34.27/55.33/42.33；353/677/285 | 1/35 | 0.78 |
| G1_data/oracle | enumerated_siblings | 33.21/55.49/41.55；354/712/284 | 0/0 | 0.00 |
| G1_data/oracle | symptom_disease_type_mismatch | 33.75/55.49/41.97；354/695/284 | 0/17 | 0.42 |
| G1_data/predicted | all_rules | 13.54/11.76/12.58；75/479/563 | 0/63 | 0.63 |
| G1_data/predicted | different_disease_in_evidence | 13.00/11.76/12.35；75/502/563 | 0/40 | 0.39 |
| G1_data/predicted | enumerated_siblings | 12.16/11.76/11.95；75/542/563 | 0/0 | 0.00 |
| G1_data/predicted | symptom_disease_type_mismatch | 12.63/11.76/12.18；75/519/563 | 0/23 | 0.22 |
| G1_data/tune_186 | all_rules | 40.86/56.20/47.32；322/466/251 | 0/21 | 0.72 |
| G1_data/tune_186 | different_disease_in_evidence | 40.81/56.20/47.28；322/467/251 | 0/20 | 0.68 |
| G1_data/tune_186 | enumerated_siblings | 39.80/56.20/46.60；322/487/251 | 0/0 | 0.00 |
| G1_data/tune_186 | symptom_disease_type_mismatch | 39.85/56.20/46.63；322/486/251 | 0/1 | 0.03 |
| G1_data/tune_279 | all_rules | 43.33/57.24/49.32；328/429/245 | 0/22 | 0.80 |
| G1_data/tune_279 | different_disease_in_evidence | 43.21/57.24/49.25；328/431/245 | 0/20 | 0.73 |
| G1_data/tune_279 | enumerated_siblings | 42.11/57.24/48.52；328/451/245 | 0/0 | 0.00 |
| G1_data/tune_279 | symptom_disease_type_mismatch | 42.21/57.24/48.59；328/449/245 | 0/2 | 0.07 |
| G1_data/tune_93 | all_rules | 32.79/45.90/38.25；263/539/310 | 0/46 | 1.24 |
| G1_data/tune_93 | different_disease_in_evidence | 32.23/45.90/37.87；263/553/310 | 0/32 | 0.85 |
| G1_data/tune_93 | enumerated_siblings | 31.01/45.90/37.02；263/585/310 | 0/0 | 0.00 |
| G1_data/tune_93 | symptom_disease_type_mismatch | 31.53/45.90/37.38；263/571/310 | 0/14 | 0.37 |
| G6_all/oracle | all_rules | 45.79/42.63/44.16；272/322/366 | 2/12 | 0.18 |
| G6_all/oracle | different_disease_in_evidence | 45.56/42.63/44.05；272/325/366 | 2/9 | 0.07 |
| G6_all/oracle | enumerated_siblings | 45.07/42.95/43.98；274/334/364 | 0/0 | 0.00 |
| G6_all/oracle | symptom_disease_type_mismatch | 45.29/42.95/44.09；274/331/364 | 0/3 | 0.11 |
| G6_all/predicted | all_rules | 18.71/9.09/12.24；58/252/580 | 0/16 | 0.20 |
| G6_all/predicted | different_disease_in_evidence | 18.35/9.09/12.16；58/258/580 | 0/10 | 0.13 |
| G6_all/predicted | enumerated_siblings | 17.79/9.09/12.03；58/268/580 | 0/0 | 0.00 |
| G6_all/predicted | symptom_disease_type_mismatch | 18.13/9.09/12.11；58/262/580 | 0/6 | 0.08 |
| G6_all/tune_311 | all_rules | 49.69/41.71/45.35；239/242/334 | 0/8 | 0.34 |
| G6_all/tune_311 | different_disease_in_evidence | 49.18/41.71/45.14；239/247/334 | 0/3 | 0.13 |
| G6_all/tune_311 | enumerated_siblings | 48.88/41.71/45.01；239/250/334 | 0/0 | 0.00 |
| G6_all/tune_311 | symptom_disease_type_mismatch | 49.38/41.71/45.22；239/245/334 | 0/5 | 0.21 |
| G6_all/tune_622 | all_rules | 32.42/63.87/43.01；366/763/207 | 1/36 | 0.80 |
| G6_all/tune_622 | different_disease_in_evidence | 32.08/63.87/42.71；366/775/207 | 1/24 | 0.50 |
| G6_all/tune_622 | enumerated_siblings | 31.48/64.05/42.21；367/799/206 | 0/0 | 0.00 |
| G6_all/tune_622 | symptom_disease_type_mismatch | 31.80/64.05/42.50；367/787/206 | 0/12 | 0.29 |
| G6_all/tune_933 | all_rules | 40.60/59.51/48.27；341/499/232 | 0/20 | 0.67 |
| G6_all/tune_933 | different_disease_in_evidence | 40.31/59.51/48.06；341/505/232 | 0/14 | 0.47 |
| G6_all/tune_933 | enumerated_siblings | 39.65/59.51/47.59；341/519/232 | 0/0 | 0.00 |
| G6_all/tune_933 | symptom_disease_type_mismatch | 39.93/59.51/47.79；341/513/232 | 0/6 | 0.20 |
| old_v4/predicted | narrow_rules | 5.61/0.94/1.61；6/101/632 | 0/2 | 0.00 |

固定实体规则后：G0 **12.80%**（74/444/564，删64FP/0TP）、G1 **12.58%**、G6 **12.24%**；classifier分类别＋规则 **13.84%**仅历史对照。Oracle规则后G0 **43.24%**、G1 **42.76%**、G6 **44.16%**，分别误删1/1/2TP。全部数字并列，预算与事后规则限制不因分数较高而取消。

## 6. 完整64对错误实体压力测试（7行）

| 实验 | clean P/R/F1%；TP/FP/FN | noisy P/R/F1%；TP/FP/FN | 错误实体关联误连pair | 共同41条TP clean→noisy | 异常/截断clean,noisy |
| --- | --- | --- | --- | --- | --- |
| N0 | 36.47/65.96/46.97；31/54/16 | 36.36/68.29/47.46；28/49/13 | 6/151 | 29→28 | 0,0 / 0,0 |
| N10 | 35.37/61.70/44.96；29/53/18 | 34.57/68.29/45.90；28/53/13 | 4/151 | 27→28 | 0,0 / 0,0 |
| N25 | 40.28/61.70/48.74；29/43/18 | 36.00/65.85/46.55；27/48/14 | 5/151 | 28→27 | 0,0 / 0,0 |
| DPO_b005 | 17.65/70.21/28.21；33/154/14 | 17.92/75.61/28.97；31/142/10 | 15/151 | 31→31 | 2,1 / 0,0 |
| DPO_b010 | 14.98/72.34/24.82；34/193/13 | 13.91/78.05/23.62；32/198/9 | 21/151 | 32→32 | 5,1 / 0,0 |
| GRPO_e02 | 33.68/68.09/45.07；32/63/15 | 29.70/73.17/42.25；30/71/11 | 8/151 | 30→30 | 0,0 / 0,0 |
| GRPO_e0 | 35.16/68.09/46.38；32/59/15 | 32.94/68.29/44.44；28/57/13 | 8/151 | 30→28 | 0,0 / 0,0 |

N10较N0错误关联误连少2/151，但固定回归F1下降且样本小；N25不更好。GRPO两臂误连均8/151；去证据奖励未改善该指标。clean与noisy分母47/41不同，禁止凭noisy F1更高宣称抗噪提升；共同41条和错误关联151对才是更直接的配对观察。都是易合成噪声，不代表真实NER错误分布。

## 7. 历史阶段（8行；不属于9/15队列）

v1–v4分别训练entity/relation，v5–v7只训练relation并沿用entity v4；归档共11个阶段checkpoint。步数是各模块总更新，是否另行选模unknown。下面“主缓存”是其自身历史流水线/小样本诊断，不填进新固定实体或oracle栏；该三路不足的字段在JSON保留unknown。

| 历史阶段 | 数据版/训练行 | entity/relation步 | 候选/批；训练上下文 | 主缓存范围；P/R/F1；TP/FP/FN | 自由文本历史流水线F1 | 完成/问题 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old_single_json | 历史stage old_single_json（精确训练文件版本证据不足）；unknown | unknown/unknown | unknown；unknown | 148条/overall_148_mixed；undefined/0.00/0.00；0/0/704 | 0.00 | 缓存完整但relation空；弱失败基线，原模型训练履历unknown | 仅保留失败基线，不外推联合生成方法上限 |
| old_v1 | 历史stage v1（精确训练文件版本证据不足）；unknown | 1247/458 | unknown；unknown | 10条/small_sample_diagnostic；undefined/0.00/0.00；0/0/61 | unknown | 训练完成；仅10例诊断，无可比全量成功回归 | 保留训练事实，不能当全量效果 |
| old_v2 | 历史stage v2（精确训练文件版本证据不足）；unknown | 1247/458 | unknown；unknown | 10条/small_sample_diagnostic；100.00/1.64/3.23；1/0/60 | unknown | 训练完成；10例有效，全量148缓存评分作废 | 修正评测失败口径，禁止采用旧100% |
| old_v3 | 历史stage v3（精确训练文件版本证据不足）；unknown | 391/282 | unknown；unknown | 148条/overall_148_mixed；39.86/7.81/13.06；55/83/649 | 0.00 | 训练完成且148缓存；自由文本关系0；ICD占主导 | 用于实体/ICD系统演进对照 |
| old_v4 | 历史stage train_relations_v4；DeepSeek output_only source；1135 | 391/284 | 32；unknown | 148条/overall_148_mixed；36.36/8.52/13.81；60/105/644 | 1.61 | 训练完成、148回归；实体改善强，relation低 | 保留后续固定实体缓存基线 |
| old_v5 | 历史stage v5（精确训练文件版本证据不足）；unknown | reuse_v4/417 | 8；unknown | 3条/small_sample_diagnostic；2.44/7.14/3.64；1/40/13 | unknown | relation训练完成，仅3例；不优于3例v4 | 仅探索证据，不外推泛化 |
| old_v6 | 历史stage v6（精确训练文件版本证据不足）；unknown | reuse_v4/417 | 8；unknown | 3条/small_sample_diagnostic；5.00/14.29/7.41；2/38/12 | unknown | relation训练完成，仅3例；多变量同时改变 | 需独立等预算复验，非主线提分结论 |
| old_v7 | 历史stage v7（精确训练文件版本证据不足）；unknown | reuse_v4/417 | 8；unknown | 3条/small_sample_diagnostic；5.71/14.29/8.16；2/33/12 | unknown | relation训练完成，仅3例，FP仍高 | 保留负结果，优先现生成式基线 |

历史v3→v4整体entity=205/949/774→459/284/520，F1 **19.22→53.31%**；自由文本entity **6.75→46.38%**；ICD relation=55/83/11→54/2/12，F1 **53.92→88.52%**。这是组合workflow成效，不是chunk单变量。旧v4全量生成上限512；3例4096/16000都1/32/13（F1 4.26%），与512完整回归实体缓存也不完全相同，不能混成同预算对照。

### 7.1 早期档案脉络（7行，证据不足保留unknown）

这些条目只有resume_md的本地综述级证据；本次不联网补取历史GitHub文件，不反推缺失TP/FP/FN。完整训练配置或原始日志缺失，不宣称已经穷尽早期独立训练总数。

| 条目 | 数据/上下文 | 步数 | 状态及其他结果 | 内部/固定/Oracle P/R/F1 | 下一步价值 |
| --- | --- | --- | --- | --- | --- |
| early_202604_joint_format | Gemini/Schema历史858来源；非DeepSeek1345；context=unknown | unknown | 有工程/5例格式检查记录；训练履历与P/R/F1证据不足；858来源产物；初期5例格式检查不是准确率基准 | unknown / unknown / unknown | 保留数据谱系，不计作医学准确率训练胜例 |
| early_202605_2048_audit | 历史858记录；context=2048 | unknown | 长度审计完成，不是模型评测；522/858超长 | unknown / unknown / unknown | 仅用于长度问题归因，不计独立模型训练 |
| early_202605_16384_rechunk | 858来源→862 structure/860 CoT导出；不是1722独立来源；context=16384 | unknown | 长度检查完成，训练效果unknown；两类导出0超长；最长14933/12726 | unknown / unknown / unknown | 有长度工程收益；不可归因成单变量准确率收益 |
| early_202607_schema | 858条Schema修复候选，非后续1345银标；context=unknown | unknown | 数据治理；不是新模型精度完成项；8590实体/6702关系；直接/共同父节点划分检查0交叉 | unknown / unknown / unknown | 保留来源契约，不宣称医学标签全正确 |
| early_202608_output_objective | 精确训练输入SHA unknown；context=unknown | unknown | 概要可追溯，逐配置训练结果unknown；output-only 25例JSON有效23/25；放大输出仍循环；JSON有效率不作关系F1 | unknown / unknown / unknown | 如需完整早期单次训练追踪，应补原始配置/数值日志，不推断预算 |
| early_202608_score_diagnostic | 16例混合诊断与25例复核非同一评测规模；context=unknown | unknown | 评分诊断完成；不能并入主线整图排名；约73.9%的局部诊断不能替代25例关系F1约17.7%；TPFPFN unknown，无法算式复核 | unknown / unknown / unknown | 保留评分契约边界；需原日志才能细拆 |
| early_202608_relations_v2 | 精确训练输入SHA unknown；context=unknown | 819 | 概要称训练完成；只有8例dev分数，三路完整评测unknown；8例dev relation F1=6.12%；TP/FP/FN unknown，保留报告数而非复算 | unknown / unknown / unknown | 列为历史小样本训练结果，不能与81条固定/oracle排名 |

### 7.2 全部30份历史缓存清单（缓存≠训练）

| 缓存名 | 样本/范围 | Relation P/R/F1%；TP/FP/FN | Entity F1% | 有效性/约定 |
| --- | --- | --- | --- | --- |
| dev_metrics.json | 148/overall_148_mixed | undefined/0.00/0.00；0/0/704 | 12.81 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_old_free_text.json | 81/free_text | undefined/0.00/0.00；0/0/638 | 0.00 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_old_full_recheck.json | 148/overall_148_mixed | undefined/0.00/0.00；0/0/704 | 12.81 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_old_structured_icd.json | 67/structured_icd | undefined/0.00/0.00；0/0/66 | 61.75 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_v1_10.json | 10/small_sample_diagnostic | undefined/0.00/0.00；0/0/61 | 2.47 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_v2.json | 148/overall_148_mixed | undefined/undefined/undefined；0/0/0 | undefined | 原报告0/0/0却P/R/F1=1：旧scorer跳过无效预测，成绩作废；本台账保留原数但不选用 |
| dev_metrics_v2_10.json | 10/small_sample_diagnostic | 100.00/1.64/3.23；1/0/60 | 12.17 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v3_10.json | 10/small_sample_diagnostic | 100.00/1.64/3.23；1/0/60 | 8.51 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v3_fast_10.json | 10/small_sample_diagnostic | 100.00/1.64/3.23；1/0/60 | 8.51 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v3_fast_full_recheck.json | 148/overall_148_mixed | 39.86/7.81/13.06；55/83/649 | 19.22 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v3_free_text.json | 81/free_text | undefined/0.00/0.00；0/0/638 | 6.75 | 原P=1属于空预测约定；公式P未定义，F1=0，不能称精度100% |
| dev_metrics_v3_full.json | 148/overall_148_mixed | 39.86/7.81/13.06；55/83/649 | 19.22 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v3_structured_icd.json | 67/structured_icd | 39.86/83.33/53.92；55/83/11 | 81.69 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_10.json | 10/small_sample_diagnostic | 50.00/1.64/3.17；1/1/60 | 40.39 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_512_3.json | 3/small_sample_diagnostic | 100.00/7.14/13.33；1/0/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_bigchunk_3.json | 3/small_sample_diagnostic | 100.00/7.14/13.33；1/0/13 | 68.00 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_cap1024_3.json | 3/small_sample_diagnostic | 33.33/7.14/11.76；1/2/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_full_bigchunk_jsonstop_512.json | 148/overall_148_mixed | 36.36/8.52/13.81；60/105/644 | 53.31 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_jsonstop2_3.json | 3/small_sample_diagnostic | 100.00/7.14/13.33；1/0/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_jsonstop_3.json | 3/small_sample_diagnostic | 100.00/7.14/13.33；1/0/13 | 13.79 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_max16000_3.json | 3/small_sample_diagnostic | 3.03/7.14/4.26；1/32/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_max4096_3.json | 3/small_sample_diagnostic | 3.03/7.14/4.26；1/32/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_e1r1_midchunk_3.json | 3/small_sample_diagnostic | 100.00/7.14/13.33；1/0/13 | 40.00 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_free_text_bigchunk_jsonstop_512.json | 81/free_text | 5.50/0.94/1.61；6/103/632 | 46.38 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v4_structured_icd_bigchunk_jsonstop_512.json | 67/structured_icd | 96.43/81.82/88.52；54/2/12 | 90.11 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v5_b8_3.json | 3/small_sample_diagnostic | 2.44/7.14/3.64；1/40/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v5_b8_no_ss_3.json | 3/small_sample_diagnostic | 3.70/7.14/4.88；1/26/13 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v6_order_b8_neg3_3.json | 3/small_sample_diagnostic | 5.00/14.29/7.41；2/38/12 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v6_order_b8_neg3_no_ss_3.json | 3/small_sample_diagnostic | 5.56/14.29/8.00；2/34/12 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |
| dev_metrics_v7_importance_b8_neg3_3.json | 3/small_sample_diagnostic | 5.71/14.29/8.16；2/33/12 | 76.60 | 缓存包含分层/重复复核/参数开关，不能按文件数统计训练数 |

## 8. 选模与阈值留痕

### 8.1 生成式内部检查点（41行）

均为69条内部自由文本oracle图；星号“选中”仅表示登记规则选中，不表示过精确率门槛。G6最终933步F1较高也不能覆盖其F0.5规则选出的311步。

| 模型 | 步数 | P/R/F1%；TP/FP/FN | 选中/备注 |
| --- | --- | --- | --- |
| DPO_b005 | 32 | 22.32/64.05/33.11；367/1277/206 | 选中 |
| DPO_b005 | 64 | 16.61/65.10/26.47；373/1872/200 | 未选中 |
| DPO_b010 | 32 | 16.87/65.79/26.85；377/1858/196 | 选中 |
| DPO_b010 | 64 | 13.10/67.54/21.94；387/2568/186 | 未选中 |
| DPO_no_hard_b010 | 32 | 7.84/69.11/14.07；396/4658/177 | 完整tune32，但未完成选模 |
| G0 | 93 | 31.52/45.38/37.20；260/565/313 | 未选中 |
| G0 | 186 | 40.61/56.20/47.14；322/471/251 | 未选中 |
| G0 | 279 | 42.11/58.64/49.02；336/462/237 | 选中 |
| G1_data | 93 | 31.01/45.90/37.02；263/585/310 | 未选中 |
| G1_data | 186 | 39.80/56.20/46.60；322/487/251 | 未选中 |
| G1_data | 279 | 42.11/57.24/48.52；328/451/245 | 选中 |
| G2_short_batch | 93 | 29.15/22.69/25.52；130/316/443 | 未选中 |
| G2_short_batch | 186 | 26.16/49.04/34.12；281/793/292 | 未选中 |
| G2_short_batch | 279 | 28.88/49.74/36.54；285/702/288 | 选中 |
| G3_order | 93 | 19.33/43.63/26.80；250/1043/323 | 未选中 |
| G3_order | 186 | 22.08/57.42/31.90；329/1161/244 | 未选中 |
| G3_order | 279 | 27.24/54.10/36.24；310/828/263 | 选中 |
| G4_multi | 93 | 29.59/45.55/35.88；261/621/312 | 未选中 |
| G4_multi | 186 | 35.67/56.02/43.58；321/579/252 | 未选中 |
| G4_multi | 279 | 37.25/54.80/44.35；314/529/259 | 选中 |
| G6_all | 311 | 48.88/41.71/45.01；239/250/334 | 选中 |
| G6_all | 622 | 31.48/64.05/42.21；367/799/206 | 未选中 |
| G6_all | 933 | 39.65/59.51/47.59；341/519/232 | 未选中 |
| G7_disease_comma | 93 | 43.96/34.90/38.91；200/255/373 | 未选中 |
| G7_disease_comma | 186 | 36.17/63.00/45.96；361/637/212 | 未选中 |
| G7_disease_comma | 279 | 43.06/62.83/51.10；360/476/213 | 选中 |
| GRPO_e0 | 128 | 35.55/57.77/44.02；331/600/242 | 未选中 |
| GRPO_e0 | 256 | 36.91/57.59/44.99；330/564/243 | 未选中 |
| GRPO_e0 | 384 | 37.41/57.59/45.36；330/552/243 | 选中 |
| GRPO_e02 | 128 | 34.88/57.77/43.50；331/618/242 | 未选中 |
| GRPO_e02 | 256 | 35.48/57.77/43.96；331/602/242 | 未选中 |
| GRPO_e02 | 384 | 35.55/57.94/44.06；332/602/241 | 选中 |
| N0 | 93 | 31.41/44.68/36.89；256/559/317 | 未选中 |
| N0 | 186 | 39.90/55.15/46.30；316/476/257 | 未选中 |
| N0 | 279 | 41.91/57.42/48.45；329/456/244 | 选中 |
| N10 | 93 | 29.15/49.39/36.66；283/688/290 | 未选中 |
| N10 | 186 | 38.73/53.40/44.90；306/484/267 | 未选中 |
| N10 | 279 | 39.60/55.85/46.34；320/488/253 | 选中 |
| N25 | 93 | 36.55/37.70/37.11；216/375/357 | 未选中 |
| N25 | 186 | 39.46/55.85/46.24；320/491/253 | 未选中 |
| N25 | 279 | 40.97/56.20/47.39；322/464/251 | 选中 |

### 8.2 历史classifier阈值扫描（20行，同一次训练）

以下为每epoch10个全局阈值，内部候选5714对。epoch1/2 BCE=0.153565/0.049294，选epoch2；epoch2分类别校准216/47/167、P/R/F1=82.13/56.40/66.87%，另在主表保留，不是新增训练。

| epoch/step | 阈值 | P/R/F1%；TP/FP/FN | 当轮选择 |
| --- | --- | --- | --- |
| 1/641 | 0.15 | 30.82/89.82/45.90；344/772/39 | 否 |
| 1/641 | 0.25 | 36.15/88.25/51.29；338/597/45 | 否 |
| 1/641 | 0.35 | 40.05/85.64/54.58；328/491/55 | 否 |
| 1/641 | 0.45 | 43.19/82.77/56.76；317/417/66 | 否 |
| 1/641 | 0.5 | 45.18/80.68/57.92；309/375/74 | 否 |
| 1/641 | 0.55 | 46.68/78.85/58.64；302/345/81 | 否 |
| 1/641 | 0.65 | 50.80/74.41/60.38；285/276/98 | 否 |
| 1/641 | 0.75 | 56.68/68.67/62.10；263/201/120 | 否 |
| 1/641 | 0.85 | 65.11/61.88/63.45；237/127/146 | 是 |
| 1/641 | 0.95 | 68.70/41.25/51.55；158/72/225 | 否 |
| 2/1282 | 0.15 | 36.24/89.03/51.51；341/600/42 | 否 |
| 2/1282 | 0.25 | 42.55/87.21/57.19；334/451/49 | 否 |
| 2/1282 | 0.35 | 47.60/85.38/61.12；327/360/56 | 否 |
| 2/1282 | 0.45 | 51.36/83.55/63.62；320/303/63 | 否 |
| 2/1282 | 0.5 | 53.57/82.25/64.88；315/273/68 | 否 |
| 2/1282 | 0.55 | 54.75/81.20/65.40；311/257/72 | 否 |
| 2/1282 | 0.65 | 57.47/78.33/66.30；300/222/83 | 否 |
| 2/1282 | 0.75 | 62.45/74.67/68.01；286/172/97 | 否 |
| 2/1282 | 0.85 | 65.98/67.36/66.67；258/133/125 | 是 |
| 2/1282 | 0.95 | 67.43/53.52/59.68；205/99/178 | 否 |

## 9. 算式审计、冲突与未完成项

从6个归档JSON的数值投影递归检查 **1135个TP/FP/FN对象**（含重复副本、per-class与阈值扫点，不是实验次数），其中404个有至少一个零分母。有效非零分母P/R/F1与报告值在1e-10容差内全部一致；没有发现需要选择较高分来“解决”的数值冲突。本次是计数算式复核，不是重跑原始预测评分。历史无效成绩、口径/状态/时间冲突如下保留。

| 编号 | 差异/来源 | 处理 |
| --- | --- | --- |
| D01 | 9/8最终队列G0固定预测实体仍缺失；9/15补完148条，81自由文本74/508/564，P/R/F1=12.71/11.60/12.13%（finite_md, g0, stage_md） | 两时点同时保留；历史G0改列本地归档已完整评测，不增加9/15的13项计数 |
| D02 | 旧概要列G1+规则12.58%为生成式规则后最高（9/8当时无G0固定）；G0补齐后all_rules=74/444/564，F1=12.80%（history_md, resume_md, g0） | 不静默沿用或覆盖原排名；扩大比较范围后G0规则分数须加入，同数据预算/独立盲测仍未证明。并列数值，不挑高当主成绩 |
| D03 | DPO no-hard旧child事件running；父queue已failed且tune64超时（round, resume_evidence, user_main_thread_update） | effective overall=failed/evaluation_failed；训练complete64；不计运行，不把tune32作为选中模型 |
| D04 | 训练manifest status=training_complete_evaluation_pending；11项最终作业status=complete且回归齐全（round） | 训练阶段状态与最终作业状态分字段，不用旧训练字段抹去后续评测完成 |
| D05 | 旧v2全量0/0/0却P=R=F1=1；旧空预测P=1与当前scorer空集0约定不同（history, audit_md） | 三零成绩无效、公式分母0为undefined；原reported保留，禁止参与排名；其他0TP/0FP/正FN的F1重算0 |
| D06 | 分类别阈值内部P=82.13%（5714候选/383正决策）；同模型固定81自由文本P=30.94%、F1=13.68%（classifier, classifier_md） | 不同分母/候选空间不是算式冲突；历史分类头非生成式主线 |
| D07 | classifier数据构建dev_oracle covered_positive=408、output_pairs=6292；classifier评测candidate_ceiling oracle covered=444、candidate_pairs=6328（classifier, classifier_manifest, audit_md） | 构建过滤与完整候选计数口径分开；本审计未逐对重建，不取较高值覆盖408；也不与生成式546/638混用 |
| D08 | 旧G5预检4779→4245少534标签；旧G5未训；v2局部共同监督4785→4606少179训练标签、482→472少10tune标签（history_md, round_md, repair_md, controlled_manifest） | 不同数据版；不能相互覆盖或推断G5已完成；新4项全为prepared、done0 |
| D09 | 早期GRPO长度6144预检失败；正式两组max_length8192，prompt上限4752+rollout1536=6288（token_preflight, queue_C_GRPO_r3） | 保留早期失败，正式配置用登记8192；普通回归16000是生成上限，不是训练上下文 |
| D10 | 计划记录开工前621 tests；主线程后续转达633 tests与新受控包真实分词通过（continuation_plan, controlled_manifest, user_main_thread_update） | 区分阶段与归属；本台账只复核归档与算式，未冒称亲自运行621/633 tests |

| 未完成研究 | 状态 | 问题/边界 |
| --- | --- | --- |
| G5_20260915 | not_started | 旧13项之一；新V2_G5_local_common不是补上旧G5训练结果 |
| DPO_nohard_20260915 | evaluation_failed | 总训64；tune32完整但未选模；tune64 14400秒超时，63/69篇、661批部分输出、3次16000截断；父queue failed优先于残留child running |
| G6_leave_one_out_and_label_ablation | no_completed_results_in_cited_archive | 有矩阵配置不等于执行；不可把G6组合代替这些单项 |
| PPO | no_training_result_in_cited_archive | DPO/GRPO实现不等于PPO完成 |
| repeat_seeds | not_completed_in_cited_archive | 现有主要对照仅seed42 |
| v2_semantic_adjudication | incomplete | 264复核入口非264确认错误；机械修复/测试通过不等于人工医学金标 |
| independent_blind_gold_and_live_pipeline | not_completed_in_cited_archive | 新4项仍沿用历史实体缓存；实时entity→relation和独立盲测待做 |
| same_loader_zero_step_RL | missing | GRPO缺同加载器零步，不能把继承/加载变化算RL收益 |

## 10. 下一步建议（本任务未执行）

| 优先级 | 建议 | 原因与验收 |
| --- | --- | --- |
| 1 | 新4项仅按主线程计划推进并保留准备状态 | V2_base/noise比较固定正例/证据，full/local共同监督更接近单变量；尚无训练结果，不可预报提升；验收：2步smoke及保存重载→登记预算→130内部选模→148固定/oracle分层→完整TPFPFN与真实曝光 |
| 2 | 优先来源/实体/候选契约与真实错误实体 | 旧固定缓存候选仅120/638=18.81%，生成式oracle546/638=85.58%；relation调参无法恢复没进候选的边；验收：来源隔离out-of-fold实体预测、人工审定真实负例、规范名与mention双层、最终实时全链路回归 |
| 3 | 保留银标与独立盲测边界 | 264项未审；重复查看历史回归；NONE未必真负例，UNKNOWN改变监督分布；验收：独立人工关系金标与分层错误审定；不回改旧gold抬分 |
| 4 | 补预算与重复性，拆开G7候选输入增益 | G6总步933/G0G1 279；G2只有17702对曝光/N0 59088；G7覆盖+27且26共用prompt变化另增6批；验收：同pair/监督集合与token/compute实测，重复seed与区间；同候选同prompt比较 |
| 5 | RL暂不以扩算力作为下一优先级 | DPO/GRPO均未超过历史G1固定F1；GRPO缺同加载器零步；新UNKNOWN数据不兼容旧奖励器；验收：先奖励/偏好语义审定、零步加载对照与真实负例；旧no-hard恢复另行授权 |

## 11. 数据来源与复核方式

JSON逐项保留source ID、源文件SHA256、指标evidence路径、reported值与recomputed值。训练输入只读取/比较哈希，未复制训练医学文本。来源说明：round/finite/history分别为9/16终态、9/8晚间终态、9/8较早完整历史聚合；g0为后续恢复证据；resume为9/23归档；continuation/controlled为新准备项。`user_main_thread_update`是用户转达而非文件。

| 来源ID | 本地文件 | SHA256前12位 |
| --- | --- | --- |
| round | [work/round_20260915/FINAL_RESULTS_20260916.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/FINAL_RESULTS_20260916.json>) | 9b9708d65f53 |
| round_md | [work/round_20260915/FINAL_RESULTS_20260916.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/FINAL_RESULTS_20260916.md>) | dfcf1aed9f5d |
| finite | [work/generative_20260908/FINAL_FINITE_QUEUE_AGGREGATES_2130.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/FINAL_FINITE_QUEUE_AGGREGATES_2130.json>) | ac6f77fcf6a9 |
| finite_md | [work/generative_20260908/FINAL_FINITE_QUEUE_REPORT_2130.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/FINAL_FINITE_QUEUE_REPORT_2130.md>) | b4d6afa13d31 |
| g0 | [work/round_20260915/G0_RECOVERED_FULL_METRICS.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/G0_RECOVERED_FULL_METRICS.json>) | fec2108222b2 |
| stage_md | [work/round_20260915/REPORT.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/REPORT.md>) | 281f683862af |
| history | [work/generative_20260908/ALL_EXPERIMENTS_AGGREGATES_1940.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/ALL_EXPERIMENTS_AGGREGATES_1940.json>) | 242564a63d60 |
| history_md | [MedicalNER-Qwen3/agent/reports/20260908/ALL_EXPERIMENTS_SUMMARY_1940.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260908/ALL_EXPERIMENTS_SUMMARY_1940.md>) | d7977f727258 |
| classifier | [MedicalNER-Qwen3/agent/reports/20260907/metrics.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260907/metrics.json>) | 14c4d530ff12 |
| classifier_md | [MedicalNER-Qwen3/agent/reports/20260907/RESULTS.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260907/RESULTS.md>) | d1cb654361e5 |
| class_training | [MedicalNER-Qwen3/agent/reports/20260907/training_history.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260907/training_history.json>) | 891b672e5c8f |
| classifier_manifest | [work/audit_20260907/training/run_manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/audit_20260907/training/run_manifest.json>) | 253067879b57 |
| audit_md | [MedicalNER-Qwen3/agent/reports/20260907/AUDIT.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260907/AUDIT.md>) | 6fbfeedf8482 |
| legacy_audit | [work/audit_20260907/audit_legacy_repro.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/audit_20260907/audit_legacy_repro.json>) | 97855513ac5a |
| resume_evidence | [work/resume_audit_20260923/evidence.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/resume_audit_20260923/evidence.json>) | 478162af5c5b |
| resume_md | [work/resume_audit_20260923/PROGRESS_AND_METRICS.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/resume_audit_20260923/PROGRESS_AND_METRICS.md>) | 8d66a2164744 |
| running_md | [MedicalNER-Qwen3/agent/reports/20260908/RUNNING.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/reports/20260908/RUNNING.md>) | 1b3d24eb73ee |
| repair_md | [work/data_repair_20260915_v2/DELIVERY_20260916.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/data_repair_20260915_v2/DELIVERY_20260916.md>) | 6f9696e82abb |
| matrix | [MedicalNER-Qwen3/agent/generative/experiments.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/agent/generative/experiments.json>) | 8dec62da79b1 |
| token_preflight | [work/round_20260915/TOKEN_PREFLIGHT_V1.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/TOKEN_PREFLIGHT_V1.json>) | 45cd30fae220 |
| prefs | [work/round_20260915/preferences_g1_final_v1/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/preferences_g1_final_v1/manifest.json>) | 881d8dcd755b |
| manifest_old_G0 | [work/generative_20260908/data/G0/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/data/G0/manifest.json>) | 037afcf93bdf |
| manifest_old_G1_data | [work/generative_20260908/data/G1_data/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/data/G1_data/manifest.json>) | 9b1f09d413f3 |
| manifest_old_G6_all | [work/generative_20260908/data/G6_all/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/generative_20260908/data/G6_all/manifest.json>) | 411056559bfd |
| manifest_G1_data | [work/round_20260915/data_final_v1/G1_data/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G1_data/manifest.json>) | 1b4ed5ed0deb |
| manifest_G2_short_batch | [work/round_20260915/data_final_v1/G2_short_batch/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G2_short_batch/manifest.json>) | 7614e44d0f49 |
| manifest_G3_order | [work/round_20260915/data_final_v1/G3_order/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G3_order/manifest.json>) | 113ecf79b9b1 |
| manifest_G4_multi | [work/round_20260915/data_final_v1/G4_multi/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G4_multi/manifest.json>) | 3d9d48202466 |
| manifest_G5_local | [work/round_20260915/data_final_v1/G5_local/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G5_local/manifest.json>) | a5140de88d2d |
| manifest_G7_disease_comma | [work/round_20260915/data_final_v1/G7_disease_comma/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/data_final_v1/G7_disease_comma/manifest.json>) | fd143ddbb065 |
| noise_clean | [work/round_20260915/noise_final_v1/clean/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/noise_final_v1/clean/manifest.json>) | ea165f767250 |
| noise_noise_1_of_10 | [work/round_20260915/noise_final_v1/noise_1_of_10/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/noise_final_v1/noise_1_of_10/manifest.json>) | 4dc2baa123b1 |
| noise_noise_1_of_4 | [work/round_20260915/noise_final_v1/noise_1_of_4/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/noise_final_v1/noise_1_of_4/manifest.json>) | 4f6d69548866 |
| queue_A_G_r2 | [work/round_20260915/deployment_v1/A_G_r2.queue.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/deployment_v1/A_G_r2.queue.json>) | ecc083415f5a |
| queue_B_DPO_r3 | [work/round_20260915/deployment_v1/B_DPO_r3.queue.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/deployment_v1/B_DPO_r3.queue.json>) | 50d8d51450c1 |
| queue_C_GRPO_r3 | [work/round_20260915/deployment_v1/C_GRPO_r3.queue.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/deployment_v1/C_GRPO_r3.queue.json>) | 13f24a58fba0 |
| queue_D_noise_r2 | [work/round_20260915/deployment_v1/D_noise_r2.queue.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260915/deployment_v1/D_noise_r2.queue.json>) | 4bc8217e08cd |
| continuation_plan | [MedicalNER-Qwen3/docs/continuation_20260928.md](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/MedicalNER-Qwen3/docs/continuation_20260928.md>) | 5aa6fa173e11 |
| controlled_manifest | [work/round_20260928/controlled_v1/manifest.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260928/controlled_v1/manifest.json>) | 19efa44a1197 |
| controlled_matrix | [work/round_20260928/controlled_v1/experiments.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260928/controlled_v1/experiments.json>) | 3688d4494cae |
| existing_v2_validation | [work/round_20260928/validation_existing_v2.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260928/validation_existing_v2.json>) | e42b3fd91031 |

复核清单：旧13项状态不变；新4项done0且三路指标unknown；主表行数18；全部指标从计数计算；历史缓存无效项不参与排名；每一数据版本单列；21份本地训练/准备输入文件哈希与manifest相符；写入前62份引用文件/输入SHA复核全部不变；另核对360处指标计数引用与原文件一致。系统git不可用（缺Developer Tools），不声称git工作树干净；以文件哈希和只新增两文件作为本任务边界。

本台账未公开医学原文、样例prompt、逐样本实体图、权重或密钥。完整机器版：[EXPERIMENT_LEDGER.json](</Users/mu6oku/Documents/Codex/2026-08-25/wo-xi/work/round_20260928/EXPERIMENT_LEDGER.json>)。
