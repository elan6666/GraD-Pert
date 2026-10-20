# PT2 中等分析端点修复（2026-10-20）

预改动基线：cde649bd85aef016a11f6265cb096946b4c09a50，开始前已核验干净且 GitHub main 同SHA。
训练/末轮评估保持 f1799a089365f1722e73b676b3a338237c370c86。恢复run `nadig_jurkat-seed1-20261020T041003Z-7c5d08f09bd747ea84b62ef446c0cb87` 已完成6轮900更新及last18，但原分析attempt3失败，证据不覆盖。

## 原因与修复范围

训练在第2轮保存pretrain角色及独立pretrain-checkpoint.json，后续epoch journal重建未保留该角色，实际checkpoint仍在。分析改为验证独立收据的run identity、声明轮数、文件边界和SHA256；journal若也声明角色，两者必须完全一致。不能选择last代替epoch2。

恢复容器只继承前三轮history，早期诊断文件仍在父run。报告允许经RECOVERY_MIGRATION.json查找缺失的继承epoch（含初始epoch0），核验父子source/data/config、run ID、连续继承轮数及对应history前缀，按原文件哈希记录绝对来源路径；不写入或改写已完成run。

PT2修复在新的独立输出根执行，继续要求两个不同checkpoint、同一份物理中等采样manifest、0更新/state unchanged、reference/fast原1e-8门及完整输出hash。保留原queue与failed分析目录。报告与端点解析使用新发布分析源码；原生模型通过PYTHONPATH导入f1799a0，数值分析脚本和src必须与训练版本逐文件无差异。分析控制器commit与训练commit分别记录。

## 验证

本地65项相关测试通过（PT2端点、父诊断来源、完整报告、无SSL缺省项、中等observer），ruff及format/diff检查通过。新增测试拒绝错误identity、收据/journal冲突、错误epoch、内容hash变化及父history变化。服务器还将以真实保留的epoch2/epoch6分别做CPU端点与saved报告预检，之后仅补中等分析，不重训、不重做已完成last测试、不扩全量。
