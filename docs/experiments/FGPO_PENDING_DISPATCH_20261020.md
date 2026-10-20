# 新基线待运行队列：精确选臂派发

PT2双端点分析已独立验收，下一组S12-L4。已完成B0、VH等不重跑。

新增编排选项只作用于FGPO family的待运行臂：S12-L4→CG1→U24→MR1→P1→C1→O1→K2-fixed-prefix。选择必须非空、唯一、遵守已授权顺序；B0/VH、未知臂、重复或重排均拒绝。完整原manifest及配置保持不变，所选范围写入新queue并纳入其哈希；所有训练/评估/分析与source/data校验仍执行。未提供选项时历史行为不变。执行阶段不能再用命令行更改queue范围。

先只准备S12-L4单臂的新queue，4epoch/global320(80×2×累积2)，无逐轮验证、last18、同清单中等分析。重新执行10步训练/恢复及两卡eval128/reference128预检，核对混合4Local的source/config身份，再由主验收后启动。新run ID，不接着旧十臂queue运行，不给历史预检重标当前SHA。保持原资源门、allocator、数据/表达剔除、训练语义；视图比例和权重不因选臂改变。

## 首次派发证据与评估入口修复

r1在控制器导入时退出，虚拟环境的editable包仍绑定基础checkout，缺train_entry；没有CUDA预检或优化更新。r2显式PYTHONPATH绑定已发布src后通过训练/恢复10步，但eval入口仍按完整十臂拒绝单臂queue。两次失败目录保留。评估绑定改为复用同一manifest验证与ordered pending选择器，继续严格匹配每张卡、batch/reference128、config/source/publication、queue与10步；不放宽任何数值或资源门。新增真实validate_eval_binding回归覆盖两GPU，修复前复现ten-arm错误，修复后通过，并拒绝only_arms与实际rows不一致。

后续新run明确设置PYTHONPATH到对应不可变发布src，保留原allocator；不修改虚拟环境的全局editable安装或活动checkout。新版本重做预检，旧通过的10步收据保留原SHA，不重标新版本。
