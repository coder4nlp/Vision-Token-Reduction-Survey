# ​​研究背景​​
**问题**：多模态大语言模型（MLLMs）如BLIP、LLaVA、QwenVL和InternVL等在视觉-语言任务中表现出色，但其计算资源需求和时间成本较高，主要原因是视觉token数量庞大。

**难点**：视觉token的数量通常比文本token多出十倍以上，导致计算复杂度呈平方级增长，严重影响训练效率。
**相关工作**：现有方法主要集中在优化视觉投影阶段或LLM阶段的token压缩，但缺乏联合优化策略，未能充分利用模型加速的潜力。
# ​​研究方法​​
提出了三种视觉token压缩方法：Projector Visual Token Compressor (PVTC)、Layer-wise Visual Token Compressor (LVTC) 和 Resolution Visual Token Compressor (RVTC)。

PVTC通过局部和全局查询的点对区域交叉注意力机制，有效压缩视觉token，同时保留局部细节和全局语义。具体来说，PVTC将视觉嵌入划分为非重叠网格，利用像素shuffle操作和MLP层生成局部嵌入，并通过CLSToken生成全局查询，最终通过元素相加结合局部和全局特征。

LVTC在LLM的浅层压缩token，在深层扩展token，通过引入高分辨率投影和多投影器策略，增强视觉信息的利用。LVTC在初始层使用低分辨率投影压缩token，在第k层通过上采样恢复token数量，并在中间层通过多投影器注入视觉信息。

RVTC基于图像像素计数或边缘长度动态调整视觉token数量，优化高分辨率图像切片，减少计算资源浪费。RVTC采用面积匹配和边缘长度匹配策略，最大化利用可用像素，减少训练时间。
# ​​实验设计​​

在12个基准测试上评估InternVL-X的性能，包括TextVQA、DocVQA、ChartQA、InfoVQA、GQA、VQAv2、VizWiz、MMB、MMMU、POPE、SEED和IMG。
实验结果表明，InternVL-X在2B和8B模型上分别平均提升了2.85%和3.02%的性能，使用不到20%的token数量，超过了LLaVA-NeXT的平均性能2.34%。
在高分辨率评估中，InternVL-X-8B在所有4个文本导向VQA任务中均取得了最佳结果，特别是在DocVQA任务中表现突出。
# ​​结果与分析​​
Ablation实验验证了各模块的有效性。PVTC单独提升了2.25%的性能，LVTC在结合PVTC后进一步提升了性能，尽管训练时间有所增加，但整体效率和性能显著提高。RVTC虽然略微降低了模型性能，但显著减少了训练时间。
具体数据表明，PVTC在256 token压缩比下，性能提升了2.25%，LVTC在结合HR投影和多投影器后，性能提升了2.32%。RVTC在面积匹配和边缘长度匹配策略下，训练时间分别减少了49.62%和58.51%，性能保持在97.77%和99.04%。
# ​​总体结论​​
InternVL-X通过PVTC、LVTC和RVTC三种模块显著提升了模型的有效性和训练优化，实现了在性能和效率上的双重提升。
论文展示了如何通过联合优化视觉token压缩技术，显著提高多模态大语言模型的性能和效率，为未来的研究提供了重要的参考。