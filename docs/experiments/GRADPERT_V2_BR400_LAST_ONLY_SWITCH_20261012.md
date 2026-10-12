# BR400第4轮last独立双卡评估切换

用户要求停止第4轮验证直接评估、双卡最大batch，并接受评估结束再训练。首次快照验证359/413，epoch4尚未保存、last仍epoch3；旧版没有安全取消验证与先保存接口，未强杀丢权重。验证在安全操作前自然完成，TRAIN_COMPLETE与epoch4last校验通过。主仅SIGTERM精确旧controller及owned descendants；foreign未触碰。原queue留KeyboardInterrupt终态，配套USER_LAST_ONLY_SWITCH_REQUEST/STOPPED收据记录用户停止，不能把它说成整个旧bestlast队列成功或训练失败。

训练来源6a8c612bd7733d2f03b5c041aeff63c5f6ec8339；BR run nadig_jurkat-seed1-20261012T061448Z-cba17ce53a6b426cb8367b767d453cd5。last epoch4 SHA25685ca4a88f392dc29cca651eab5801e252d96d823bbe3209b15498842c6671975，configSHAd7877c03839645901f930817d5258638ca8738c650bd23fc42e5da33415e6f50；第4轮validationjoint38.659387267718564。checkpoint完整保留。F1/N1/GM1尚未启动，当前只独立评估BRlast。

新CPU守护controller与原生scientific evaluation来源分开封存。科学评估仍用同一6a8c612不可变源码与原config、冻结300controls/真值/split/reference/DEG/exposure。GPU0,1各负责不同条件；cellbatch300是每worker每条件的最高有用批次，超过300不会增加该条件的并发。四组在该源上已有300vs128输出/RNG/state一致性和两卡容量通过；本次BR用相同推理配置加载真实epoch4。Student onlyCUDA，Teacher/optimizerCPU；owned90%、physicalfree2GiB、UID1003精确ScButterfly/TotalVI单任务≤1GiB/GPU。

输出独立新run/queue ID，无旧canonical fit last-test/best-test覆盖、无伪造旧COMPLETE。验收18Pearson（overall/seen/unseen×TxPert/TriShift/Systema×all/DEG）、有效条件、全部身份哈希、true native COMPLETE及整个eval/train目录zeroPKL。评估完成前不训练下一组。后续新训练采用已确认的validation_mode=none、last-only，训练分项loss/速度/进度保留；需新配置/源码/ID，不恢复旧bestlast队列。

## 控制器验证

独立last-only守护controller28项CPU检查通过（3.06秒、1项既有Torch弃用warning），Ruff/format/diff通过，read-onlyreviewREADY。真实原生plan resolver和shards merge互调覆盖；旧postfit-best partial原有JSON集合/hash封存不变，freshoutput与原run/queue/source隔离，wholezeroPKL；eval原源码6a8c612、newcontroller另记版本。两卡GPU lease与5秒资源guard保持90%/2GiB/已授权精确foreign限制。本地检查不能代替真实双卡eval完成；待新运行receipt。

## 双卡实际评估完成（2026-10-12 21:18北京时间）

主会话独立核验 native COMPLETE、controller COMPLETE、last epoch4/checkpoint SHA、科学来源6a8c612、config、两卡296+296条件和全训练/评估目录zeroPKL。科学评估成功；旧bestlast队列仍属于用户停止，未追认其完整成功。GPU0/1 worker墙时2271.20/2358.94秒，总评估约39.3分钟（并行墙时，非两卡耗时相加）；每卡batch300、峰值allocated8.02GiB、reserved10.39GiB。

小结果收据：[BR400-last-epoch4-summary.json](../../.byte-os/evidence/br400-best-last-20261012/BR400-last-epoch4-summary.json)，SHA256 `5d838875dceb9bd6e42b658900867997474008125d88ef9f88d911a8b68cc064`。MSE 0.0054789308249717315。训练/科学评估SHA `6a8c612bd7733d2f03b5c041aeff63c5f6ec8339`；独立控制器SHA `7c40df452a4b85e814df9e9c9edffdc28a0414d0`。

| 基因范围 | TxPert全／DEG | TriShift全／DEG | Systema全／DEG | 有效条件全／DEG |
|---|---|---|---|---|
| 全部 | 0.228570 / 0.383056 | 0.177556 / 0.345992 | 0.095442 / 0.213948 | 592 / 590 |
| 见过表达（4775） | 0.421572 / 0.466754 | 0.313974 / 0.421965 | 0.234951 / 0.266813 | 592 / 590 |
| 未见表达（225） | 0.129746 / 0.172539 | 0.120777 / 0.119170 | -0.004309 / 0.041480 | 592 / 147 |

后续F1→N1→GM1使用新的disabled-validation/last-only配置，保持四轮/global400；未见DEG仅147条件，缺失不得填零。结果不用于选择架构/超参数。
