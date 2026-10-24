# R7：保留快速 Teacher，单独检验图蒸馏梯度尺度

授权：沿用用户持续表征研究、每臂2epoch、同一中等诊断和监督会话的授权；不重跑基线，不使用test选择。前置已发布干净版本为8a78218424802f1fab01be2b8b134d120c91dd71。

## 依据与单因素

R6快速EMA相对R5增强Teacher真实原型JS和固定control条件区分，但验证MSE较R5高7.62%、较RI高10.50%。同一32条件shared-graph探针SSL1梯度均值约增21.93倍，SSL2增4.50倍；均值夹角不是普遍负值，不能称梯度冲突已证实。现有探针排除完整人口A/CR，先用原生calibrate_alignment_uniformity.py的mechanism-v1对R5/R6各一个相同global320训练population作零更新核查。单批是有界观察，不冒称原八批校准完成；保留PARTIAL状态。

唯一候选R7-S1D05，父臂R6-EMA99：仅model.parameters.lambda1从1改.05，source=project_preregistered，reference指向本计划。按固定状态的梯度线性尺度.0120669/.264641≈.0456，选择预登记近似.05，不扫网格，不依据validation或test挑权重。lambda2=1为user_locked，保持。所有Teacher schedule(.99常数)、温度/center、DINO+iBOT、A=.05、U=0、CR=4e-5、epsilon²=.05、warmup30、pre-L2空间、架构/RI残差/GELU/GenePT/Softplus、views/RNG、data/split/ordered manifests、seed1、global320=80×2×2及2epoch保持。代码src/gradpert不改。

## 启动前判据

原8a源码在服务器只读诊断，checkpoint/config/训练identity分别完整保留。原始state/RNG/Teacher/center/optimizer零更新、重放误差≤2e-6，population320/有序ID/采样/view语义一致。完整人口若未保留SSL1梯度明显放大（预登记判据R6/R5 graph norm>2），不启动此候选，保留失败证据后重审机制；不偷偷调另一个系数。只有本次完整人口核查通过、配置/生成器测试通过、独立审查和clean publish后，新的source/config/run进入原生10step+restore、双卡eval128/reference128预检。主核验所有门后由同一controller启动训练。

## 实施范围与交付

实现所有权：scripts/v2/generate_representation_round7.py、configs/v2/representation_round7_20261024/**、tests/v2/test_representation_round7.py；仅必要注册到现有run_br400_group、eval_batch_capacity、run_fgpo_medium、run_representation_panel、run_representation_postfit。复用representation_family和已有runner，避免复制运行逻辑。历史R1-R6配置/manifest与模型src不修改。生成器必须验证唯一lambda1差异、lambda2锁、预算、父commit/config/checkpoint及中等panel身份；篡改后重算hash不能绕过。

每轮报告训练原始loss与加权贡献分开，joint变化因权重不可直接排序；445条件validation18及NA、MSE、相同medium506和crossed128人口；SSL1/SSL2 Teacher H/JS/Student匹配KL、错配gap、GG/GL/LL、raw/projector/token和同条件不同control分析，保留精确来源。预计会继续有不确定性，不能只凭熵下降或秩升高称问题解决。

正式长作业由既有监督会话20分钟检查和主回交唤醒负责；终态主核验并推进下一步。平台同ID幂等追加该新臂，不覆盖R6。工程候选与该科学干预隔离，不同时换kernel/性能数学。
