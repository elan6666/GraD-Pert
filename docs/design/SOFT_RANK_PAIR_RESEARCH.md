# R12 原始软秩与正对余弦实验

恢复 2026-10-22 `round3-design-20261022/DESIGN_V2_CN.md` 的数学设计。
新的正式标识是 **R12-SRBAL、R12-SRAGG**，不复用历史未运行的 R3-BAL/R3-AGG。
R11 完成后再运行本对；本文件不表示任何 GPU 校准或训练已经通过。

两臂都使用原 RI 结构，从头 seed1、2 epoch、300 次成功 optimizer 更新。
全局 batch320=双卡×micro80×累积2；EMA 按原 .99→1 计划。
原 MSE、SSL1 condition、SSL2 DINO/iBOT 权重均1，SSL1 node/spread、KoLeo、AU、CR均关闭。
无 R9/R11 warm-start、无额外 baseline 重训、无逐轮 validation/best，test延期。

| 臂 | 归一化有效秩目标 | rank 权重 | cosine 权重 | softness | warmup |
|---|---:|---:|---:|---:|---:|
| R12-SRBAL | .20 | .02 | .10 | .25 | 30 completed updates |
| R12-SRAGG | .35 | .04 | .20 | .25 | 30 completed updates |

作用点为两个 Student Global Response CLS：最后 RMSNorm 后、投影头前。
两分支均反传；不改变主干表示、不增加显式负对或模型前向。
每个 Global 使用跨设备、跨累积的完整唯一 `(control,p)` 总体，单独计算：

\[
C_v=(Z_v-\bar Z_v)^\top(Z_v-\bar Z_v)/N,\quad
p_{vk}=\lambda_k(C_v)/\operatorname{tr}C_v,\quad
\rho_v=\exp[-\sum_k p_{vk}\log p_{vk}]/\min(D,N-1).
\]

\[
L_r=\operatorname{mean}_v\left[.25\operatorname{softplus}
((\log\rho_*-\log\rho_v)/.25)\right],\quad
L_c=\operatorname{mean}_i[1-\cos(Z_{0i},Z_{1i})].
\]

\[
L=L_{RI}+\min(1,t/30)(\lambda_rL_r+\lambda_cL_c).
\]

`t`是本次更新开始前已经成功完成的 optimizer 更新数；第一次更新辅助权重0，
第31次全权重。谱分支不逐行L2、不加协方差对角 jitter，不把两个Global拼成2N样本。
协方差、对称化、特征值和熵使用FP64；只截断尺度相关容差内的负舍入误差，
`0log0`按连续极限处理。非有限、N<2、零行范数、trace≤1e-24显式失败。
极小但均匀噪声也可提高rho，必须同时报告trace、绝对/相对方差及均值能量。

## 无更新真实校准

只在服务器使用原始 RI 配置。`scripts/v2/calibrate_spectral_pair.py` 复用已验证的 AU
驱动：固定8个完整训练batch，每批320唯一输入；分别在初始化与既有 RI-last观察。
已有末轮仅用于校准，绝不是两臂训练初始化。两个端点与两个强度使用相同ordered
control/truth/condition/context IDs和相同校准随机种子，不读取validation或test。

先记录完整CLS总体，再按同一前向随机状态重放微批VJP；每微批重放误差≤2e-6。
每批保存原joint、MSE、BAL/AGG原始rank、原始cos，以及各固定权重后的rank/cos、
auxiliary之和、total的Response/graph梯度范数、点积、余弦和夹角。
加权梯度显示warmup终点尺度1，第一次权重0另列，不能声称它是首步实际贡献。
保存rho、trace、绝对/相对方差、均值能量、实际N/D和最大可达秩。
所有Student/Teacher/center/optimizer/梯度/pending/mode和Python/NumPy/Torch/CUDA RNG
在观察前后hash相等；不调用optimizer、EMA或center commit。

命令沿用共享驱动参数：`--mode census|gradients --config ORIGINAL_RI.yaml --data-root
/data/yilangliu/... --output /data/yilangliu/... --publication ... --publication-sha256 ...
--source-commit ... --run-id ...`。梯度模式再提供已封存的`--shared-policy`及其hash；
RI-last需checkpoint、checkpoint-sha256、原training-manifest及其hash四项。
可用`--stop-after 1`先工程验证，再以同一不可变计划`--resume`完成8批。
每个真实GPU进程必须设置`PYTORCH_ALLOC_CONF=expandable_segments:True`。

`verify_endpoints(init, ri_last, source_commit=...)`独立校验16个批次receipt、checksum
index、两端点完整标记、相同总体及RNG，返回的`scientific_complete`恒为false。
控制器fit门要求runtime中的`spectral_pair_calibration`包含
`init`、`ri_last`两个服务器根目录及`init_complete_sha256`、`ri_last_complete_sha256`。
正式source必须为当前clean published commit；原RI训练source另行保留，允许并披露跨版本。
原系数保持，若真实证据需要调整，必须新设计修订及版本，不能训练中偷偷自适应调参。

## 工程与研究验收

每臂通过真实native10-update、完整总体320、first-update-zero、warmup及checkpoint
全状态/RNG精确恢复；双GPU各自eval128/reference128严格2e-6。
10-step是短预检，不是持续容量或训练成功证据。

完成每臂后输出同一medium1335条件/5317行、deep128/506与fixed32条件/128行面板，
保留真实pair/context数量、H/KL与原始SSL/MSE及实际warmup加权rank/cos损失。
逐条件或固定小面板不具备完整320总体时，新损失标NA；inactive与unobserved分开。
最终完整445条件validation18（all/seen/unseen × full/DEG × 三种Pearson）及MSE，zeroPKL。
自动中文报告、图和全表后由主会话更新实验平台；test仍延期。

主比较复用历史R3-RI；RS-B0、RS-RI仅作背景，原source/checkpoint/评估身份不可改写。
这是跨版本单seed筛选，零新项parity与DDP/fullbatch等价证明不保证训练轨迹相同。
不能仅用秩提高或CE降低判定任务改善，且本联合对不能分离rank与cos各自贡献。
