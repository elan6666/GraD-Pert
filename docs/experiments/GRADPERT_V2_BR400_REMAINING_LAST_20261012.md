# BR400 剩余三臂：关闭逐轮验证、只评估 last

BR400 epoch4 last 已独立双卡评估完成，592条件及18个指标验收见 GRADPERT_V2_BR400_LAST_ONLY_SWITCH_20261012.md。旧BR/F1/N1/GM1 bestlast队列按用户停止保留，BR不重跑，旧队列剩余ID不复用。

本轮顺序 F1 → N1 → GM1；每组训练全部4轮后双卡评估last，真正完成后才启动下一组。共同Jurkat cap40/seed1/global400（micro100×world2×accum2）、eval cellbatch300/GPU、GenePT映射重计算、冻结300-control/真实表达/DEG/表达见过分组/优化器更新协议保持；唯一共同协议变动为native `validation_mode=disabled`、monitor/monitor_mode=none，不执行joint/prediction/validationPearson、不生成或测试best。

| 实验 | 相对BR的科学改动 |
|---|---|
| F1 | 普通MLP隐藏激活改为SwiGLU，原宽度和其余机制保持 |
| N1 | Transformer FFN输入采用RMSNorm；其他机制保持 |
| GM1 | 图编码器也启用4路mHC；Cell/Response原mHC保持 |

实现为旧generator/runner的显式新family `br400-remaining-last-1`；历史bestlast默认和原config保持。capacity_probe输出原生300-control推理中已有的control/基因顺序审计hash，独立于是否调用joint验证；eval capacity叶节点严格绑定新family/新config/runtime/source/queue，不伪造旧family。原生训练/模型/loss/metric代码不改。

所有三臂先各10个非零学习率完整更新、checkpoint恢复、300×5000推理及双卡batch300输出/RNG/state parity工程检查。该检查不是长期稳定性或科学结果；主验收后开始正式4轮。每臂最后18Pearson和有效条件数、checkpoint/source/config/data/controls/reference/environment身份、全run zeroPKL缺一不能进入下一臂。资源保持已授权owned90%与每卡物理free2GiB、精确UID1003小任务共用，不能触碰foreign。

目前实现/CPU检查阶段，正式训练尚未启动；代码发布、服务器CPU验收、实际进程与监督ACK分别记录，不以launch替代科学完成。BR基线有joint验证、剩余臂无验证的差异明确保留；不把配置名字一致当作完全同一执行过程，不以测试结果调参。

## 发布前验证

本地定向CPU检查104通过（36.39秒、1项既有Torch弃用warning）；最后新增绑定5通过、历史/新dispatch phase2通过，Ruff/format/diff通过。原native src无改动，模型/数据/损失/Teacher/center规则保持。证据 `.byte-os/evidence/br400-best-last-20261012/remaining-local-validation.json`。服务器同源码复核与真实双卡预检仍待执行，不能称正式训练已开始。
