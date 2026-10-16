# 官方 DINOv2 中等规模表征参考分析

只读已完成：$images张独立图像、$classes类；深度子集$deep_images张，每图固定128个patch。未训练、未微调、0次更新，状态前后hash一致。CPU $threads线程，分析耗时$minutes分钟（不含下载/绘图）。

## 最终层统计

|表示|统计行数|有效秩|有效秩/上限|相对方差|
|---|---:|---:|---:|---:|
$table

有效秩使用中心化协方差谱的熵指数；相对方差为中心化能量/总能量，完全复用GraD-Pert中等分析的定义。混合patch行不是独立图像，不能把行数当作独立样本数。单图patch秩另列，避免把跨图差异误读成图内局部差异。

## 图与读法

1. ![逐层统计](layer_statistics.png)
横轴Transformer层，左图有效秩（方向数），中图除以min(384,N−1)的比例，右图无量纲相对方差。CLS包含全部图像，混合patch只含深度子集。虚线是block原输出；实线是官方get_intermediate_layers的同一final LayerNorm应用结果，不代表每层各有一个新训练的输出norm。比较实/虚线可见归一化对统计的影响。
2. ![单图局部秩](within_image_patch_rank.png)
横轴层，纵轴每张图内部128个patch的有效秩。每个箱体以图像为样本，中线中位数、箱体25%–75%、须1.5IQR；不显示离群点。它与混合patch秩回答不同问题。
3. ![相似度分组](cosine_groups.png)
纵轴平均余弦（−1到1）；分开同图Global/Local、同类不同图和不同类。跨图组对全体有序非自身图像对精确平均；同图组每图一对等权。没有把patch当独立图像，也没有把视觉类别等同于扰动条件。

## 可比与不可比

这里只对齐统计公式、CLS/局部token分开、同一输入/同类/异类分组，以及中等深度子集506×128的规模。图像类别不等于扰动条件；宽度384不同于GraD-Pert256；DINOv2是大规模预训练，而本项目是Jurkat从头学习，因此数值不是模型合格线。图像裁剪比例作用于像素面积，不能解释成基因裁剪比例。输入仅几何裁剪，没有重放原训练颜色扰动，不称为训练目标复现。

原型熵、边际熵、JS、Student–Teacher CE/KL、梯度冲突、扰动预测指标均标为**不可用**。公开权重仅主干，没有原SSL投影头及目标统计；对特征Softmax或用下游分类头替代，都会回答另一个问题。

## 数据与版本

官方实现：[DINOv2固定源码](https://github.com/facebookresearch/dinov2/tree/$upstream)；[公开模型明确为backbone only](https://github.com/facebookresearch/dinov2/blob/$upstream/README.md#pretrained-models)。图像：[fastai Imagenette](https://github.com/fastai/imagenette)，固定validation中按类别均衡hash采样，不依赖模型输出；仅10类是数据域限制。
分析源码：$native；官方源码：$upstream；权重SHA256：$weights；manifest SHA256：$manifest。官方模型strict load；官方中间层/forward_features核对最大绝对差$parity。

图与数据全表：[layer_statistics.csv](layer_statistics.csv)、[class_statistics.csv](class_statistics.csv)、[within_image_patch_statistics.csv](within_image_patch_statistics.csv)、[cosine_groups.csv](cosine_groups.csv)、[same_image_views.csv](same_image_views.csv)、[summary.json](summary.json)、[manifest.json](manifest.json)、[crop_manifest.json](crop_manifest.json)、[COMPLETE.json](COMPLETE.json)。原图、权重和server_only_features.npz只留服务器。