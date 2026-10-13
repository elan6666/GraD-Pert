# 空闲GPU1提前推进GEARS

用户授权：另一张卡没人用则推进GEARS。主实时确认GPU1无CUDA进程，GPU0只有scDFM full，1500/3988更新；监督已暂停并交回主。预改动基线943dd0f7be950671431e939e8d5ed5dc26261957与GitHub main一致。

现顺序CPUcontroller持有双卡锁，未知兄弟任务会触发保护。新的独立发布调度器只冻结/退休精确CPUcontroller，保留独立session的scDFM PID/启动tick/cmd哈希、原训练过程和产物。接管双卡锁后采用精确adopted worker树及当前子进程树计算owned资源，未知用户/UID仍拒绝，1GiB/2GiB/90%限额不变。GEARS使用原密封row的source9607981、配置/runtime、尚未启动的run ID；唯一运输改动是CUDA_VISIBLE_DEVICES=1（逻辑cuda:0）。GEARS先十步恢复/300control预检，再原四轮训练和last18指标；scDFM继续原3988更新和评估。

两个模型真实终态独立核验后，在旧队列写明确completion-controller/schedule身份的验证转发收据；旧queue.json绝不变，已有native依赖无须重启/改hash。迁移代码仅新脚本+测试+文档；src/benchmarks Python逐文件哈希必须与旧源码完全相同。新调度commit与旧科学训练/评估commit分别记录。

验收：75项迁移/资源门检查通过；迁移仅发送CPUcontroller信号，任何失败不触碰scDFM worker；PID重用、非空GPU1、科学代码变化、GEARS已启动、旧terminal覆盖均拒绝。发布后服务器隐藏CUDA重测；安全迁移收据、worker PID/tick保持、实际日志/步数继续，GEARS GPU1真实启动，之后唯一监督20min接管。GEARS或调度失败保留收据且不自动停止adopted scDFM，交主修复；不启动BR/Scouter。
