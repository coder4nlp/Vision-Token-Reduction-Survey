# Vision-Token-Reduction-Survey
> A Comprehensive Survey and Resource Collection of Vision Token Reduction Techniques for Multimodal Large Models

# 📌 Project Overview
With the explosive growth of multimodal large models (such as LLaVA, Flamingo, BLIP-2, Qwen-VL), **efficient reduction of visual tokens** has become a key technology for reducing computational costs and enhancing inference speed. This repository systematically collects, analyzes, and compares the cutting-edge methods and advancements in the field of visual token compression.
# Introduction
The current multimodal large model consists of a visual encoder, a connector, and a large language model structure. In MLLMs, more visual tokens provide richer visual information and sigificantly improve the model performance. However, due to the n-squared complexity of the transformer, a large number of visual tokens will result in significant computational and memory consumption.

**Core Value**: Enable researchers to quickly grasp the progress in the field.

# 🗂️ Repository structure
```
Vision-Token-Reduction-Survey
├── papers_summaries/ 
├── methods_comparison/ 
├── datasets/ 
├── tech_reports_blogposts/ 
├── resources/ 
├── CONTRIBUTING.md
└── README.md 
```

| Paper Title | One-sentence Abstract | Training-Free|Date | Conference    |
|-|-|-|-|-|
|*GreedyPrune: Retenting Critical Visual Token Set for Large Vision Language Models* |We propose GreedyPrune, a training-free visual token pruning method that jointly optimizes semantic saliency and visual diversity through a combinatorial optimization framework, achieving state-of-the-art accuracy and reduced inference latency across multimodal tasks and models. |&#10004;|202506|arXiv (preprint)|
|*SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration* [[PDF]](https://www.arxiv.org/pdf/2506.12723)| We propose SP-VLA, a unified framework for accelerating Vision-Language-Action (VLA) models through joint model scheduling and token pruning, effectively reducing both temporal redundancy in sequential action generation and spatial redundancy in visual input while maintaining high accuracy, achieving up to 1.5× acceleration with less than 3% accuracy drop across multiple tasks.|&#10007;|202506|arXiv (preprint)|
|*Diversity-Guided MLP Reduction for Efficient Large Vision Transformers*|This paper proposes a Diversity-Guided MLP Reduction (DGMR) method to significantly compress large vision transformers by pruning redundant neurons in MLP modules while preserving weight diversity, achieving over 57.0% parameter and FLOPs reduction with near-lossless performance across multiple state-of-the-art models, including a 71.5% reduction for EVA-CLIP-E without performance degradation.|&#10007;|202506|arXiv (preprint)|
|*Learning Compact Vision Tokens for Efficient Large Multimodal Models* [[PDF]](https://arxiv.org/pdf/2506.07138) [[Github]](https://github.com/visresearch/LLaVA-STF)| This paper proposes a Spatial Token Fusion (STF) method and a Multi-Block Token Fusion (MBTF) module to reduce vision token sequences and enhance multi-granularity feature representation, achieving significant inference acceleration with minimal performance loss in large multimodal models.|&#10007;|202506|arXiv (preprint)|
|*Token Transforming: A Unified and Training-Free Token Compression Framework for Vision Transformer Acceleration* [[PDF](https://arxiv.org/pdf/2506.05709)]| This paper proposes a many-to-many Token Transforming framework for vision transformers, unifying existing token reduction methods into an explicit matrix transformation form, which minimizes information loss and enables training-free acceleration, achieving significant FLOPs reduction, inference speedup, and improved performance across various vision tasks such as segmentation, object detection, depth estimation, and language model generation.|&#10004;|202506|arXiv (preprint)|
|*Top-Down Compression: Revisit Efficient Vision Token Projection for Visual Instruction Tuning* [[PDF]](https://arxiv.org/pdf/2505.11945)|This paper introduces LLaVA-Meteor, a novel visual instruction tuning framework that achieves significant visual token compression (75%–95%) and improved efficiency while maintaining or enhancing performance across 12 vision-language benchmarks through a Top-Down Compression paradigm, Flash Global Fusion module, and Visual-Native Selection mechanism.|&#10007;|202505|arXiv (preprint)|
| *VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models* [[PDF]](https://arxiv.org/pdf/2505.22654v1)| This work proposes VScan, a two-stage visual token reduction framework for large vision-language models (LVLMs), achieving significant inference acceleration (2.91× speedup in prefilling, 10× FLOPs reduction) with minimal performance loss (95.4% retention) through complementary global/local token merging and intermediate-layer pruning.|&#10004; | 202505| arXiv (preprint)    | 
|*PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models.* [[PDF]](https://arxiv.org/pdf/2504.08966) [[Github]](https://github.com/orailix/PACT/tree/main)|We introduce PACT, a method that reduces inference time and memory usage in visual language models by pruning irrelevant tokens and merging visually redundant ones early in the model using a novel importance metric and Distance Bounded Density Peak Clustering. |&#10004; |202504|CVPR 2025|
| *Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs* [[PDF]](https://arxiv.org/pdf/2409.10994) [[Github]](https://github.com/FreedomIntelligence/TRIM/)| TRIM (Token Reduction using CLIP Metric) enhances Multimodal Large Language Models (MLLMs) efficiency by reducing image tokens without performance loss, validated across 12 datasets, advancing sustainable high-performance model development.  |&#10007; |202409| COLING 2025| 
| *TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model* [[PDF]](https://arxiv.org/pdf/2503.18278)| We propose TopV, a training-free token pruning method for Vision-Language Models that formulates pruning as an optimization problem using a visual-aware cost function, achieving efficient inference with reduced memory and computational cost while maintaining performance.|&#10004; | 202503 | CVPR2025 |
| *DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models* [[PDF]](https://arxiv.org/pdf/2503.02175)[[Github]](https://github.com/vbdi/divprune)| We propose DivPrune, a token pruning method for Large Multimodal Models that formulates pruning as a Max-Min Diversity Problem to maximize diversity among selected visual tokens, achieving state-of-the-art accuracy with reduced latency and memory usage across 16 image- and video-language datasets.|&#10004; |202503|CVPR 2025|
|*InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression* [[PDF]](https://arxiv.org/pdf/2503.21307)[[Github]](https://github.com/ludc506/InternVL-X)| We propose InternVL-X, a vision-language model that improves performance and efficiency through three visual token compression techniques—PVTC, LVTC, and RVTC—enabling state-of-the-art results with significantly reduced computational cost by using 20% or fewer visual tokens.| &#10007;|202503| arXiv (preprint)|
|*An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models* [[PDF]](https://arxiv.org/pdf/2403.06764) [[Github]](https://github.com/pkunlp-icler/FastV)|We propose FastV, a plug-and-play method for optimizing computational efficiency in Large Vision-Language Models (LVLMs) by learning adaptive attention patterns and pruning visual tokens, achieving significant reductions in FLOPs (e.g., 45% for LLaVA-1.5-13B) while maintaining strong performance across image and video understanding tasks, making it highly suitable for edge deployment and commercial applications.|&#10004;|202403|ECCV 2024 (Oral)|
# How LMMs work ?
> An LMM typically processes a pair of inputs, denoted as $(T,V)$, where T is the text input and $V$ is the visual input such as image or video.The text input is mapped to $N$ textual tokens $E_t=\{t_1, \dots, t_N\}$ using a text encoder.Similarly, the visual input is processed by a corresponding vision encoder. Specifically, it takes visual information $V$ as input and outputs image features, that are further converted to $M$ (generally $M \gg N$)vision tokens $E_v=\{v_1,\dots, v_M\}$ using a projector layer.


The textual tokens and visual tokens are then combined
to be fed to an LLM to generate the prediction in an autoregressive manner.  Specifically, $\hat N$ output tokens $Y=\{y_1,\dots, y_{\hat N}\}$ are generated as follows:

$$
P(y_1,\dots,y_{\hat N}| E_t,E_v)=\prod P(y_i| y< i, E_t,E_v)
$$



# TFLOP ratio
 TFLOP ratio is the TFLOP of the model with pruned tokens relative to the original model’s TFLOP with no pruning. 

 $$
 \frac {K \times  (4\mu d^2-2\mu^2d +2\mu dm)+(T-K)\times (4 \widetilde \mu d^2-2\widetilde \mu dm)}{T\times (4-\mu d^2-2 \mu ^2d+2 \mu dm)}
 $$

where $T$ is the total transformer-based decoder layers. $\mu = N+M$ is the total sequence length before pruning,$\widetilde \mu=N+M$is the sequence length after pruning. $d$ is the hidden state size of the layer, and m is the intermediate size of feed-forward network module.

# Comparison of performance and speed of different methods
## Performance comparisons on LLaVA-1.5-7B
<table>
	<thead>
		<tr>
			<th>Method</th>
			<th>Venue</th>
			<th>GQA</th>
			<th>MMB</th>
			<th>MMB<sup>CN</sup></th>
			<th>MME</th>
			<th>POPE</th>
			<th>SQA<sup>IMG</sup></th>
			<th>VQA<sup>V2</sup></th>
			<th>VQA<sup>Text</sup></th>
			<th>VizWiz</th>
			<th>Average</th>
			<th></th>
		</tr>
		<tr>
			<th colspan="12">Upper Bound, 576 Tokens (100%), 3,817 TFLOPs</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>LLaVA-1.5-7B</td>
			<td>61.9</td>
			<td>64.7</td>
			<td>58.1</td>
			<td>1862</td>
			<td>85.9</td>
			<td>69.5</td>
			<td>78.5</td>
			<td>58.2</td>
			<td>50.0</td>
			<td>100.0%</td>
			<td></td>
		</tr>
		<tr>
			<th colspan="12">Retain 192 Tokens in Average (↓ 66.6%), ~1,253 TFLOPs</th>
		</tr>
		<tr>
			<td>ToMe [7]</td>
			<td>54.3</td>
			<td>60.5</td>
			<td>-</td>
			<td>1563</td>
			<td>72.4</td>
			<td>65.2</td>
			<td>68.0</td>
			<td>52.1</td>
			<td>-</td>
			<td>88.5%</td>
			<td></td>
		</tr>
		<tr>
			<td>FastV [12]</td>
			<td>52.7</td>
			<td>61.2</td>
			<td>57.0</td>
			<td>1612</td>
			<td>64.8</td>
			<td>67.3</td>
			<td>67.1</td>
			<td>52.5</td>
			<td>50.8</td>
			<td>90.4%</td>
			<td></td>
		</tr>
		<tr>
			<td>SparseVLM [69]</td>
			<td>57.6</td>
			<td>62.5</td>
			<td>53.7</td>
			<td>1721</td>
			<td>83.6</td>
			<td>69.1</td>
			<td>75.6</td>
			<td>56.1</td>
			<td>50.5</td>
			<td>96.1%</td>
			<td></td>
		</tr>
		<tr>
			<td>PyramidDrop [60]</td>
			<td>57.3</td>
			<td>63.3</td>
			<td>56.8</td>
			<td>1797</td>
			<td>82.3</td>
			<td>69.0</td>
			<td>75.1</td>
			<td>56.5</td>
			<td>51.1</td>
			<td>97.2%</td>
			<td></td>
		</tr>
		<tr>
			<td>VisionZip</td>
			<td>59.3</td>
			<td>63.0</td>
			<td>-</td>
			<td>1783</td>
			<td>85.3</td>
			<td>68.9</td>
			<td>77.4</td>
			<td>57.3</td>
			<td>-</td>
			<td>97.8%</td>
			<td></td>
		</tr>
		<tr>
			<td>VScan (Ours)</td>
			<td>60.6</td>
			<td>63.9</td>
			<td>57.4</td>
			<td>1806</td>
			<td>86.2</td>
			<td>68.6</td>
			<td>77.8</td>
			<td>57.7</td>
			<td>50.4</td>
			<td>99.0%</td>
			<td></td>
		</tr>
		<tr>
			<th colspan="12">Retain 128 Tokens in Average (↓ 77.8%), ~833 TFLOPs</th>
		</tr>
		<tr>
			<td>ToMe</td>
			<td>52.4</td>
			<td>53.3</td>
			<td>-</td>
			<td>1343</td>
			<td>62.8</td>
			<td>59.6</td>
			<td>63.0</td>
			<td>49.1</td>
			<td>-</td>
			<td>80.4%</td>
			<td></td>
		</tr>
		<tr>
			<td>FastV</td>
			<td>49.6</td>
			<td>56.1</td>
			<td>56.4</td>
			<td>1490</td>
			<td>59.6</td>
			<td>60.2</td>
			<td>61.8</td>
			<td>50.6</td>
			<td>51.3</td>
			<td>85.4%</td>
			<td></td>
		</tr>
		<tr>
			<td>SparseVLM</td>
			<td>56.0</td>
			<td>60.0</td>
			<td>51.1</td>
			<td>1696</td>
			<td>80.5</td>
			<td>67.1</td>
			<td>73.8</td>
			<td>54.9</td>
			<td>51.4</td>
			<td>93.7%</td>
			<td></td>
		</tr>
		<tr>
			<td>PyramidDrop</td>
			<td>57.1</td>
			<td>61.6</td>
			<td>56.6</td>
			<td>1761</td>
			<td>82.3</td>
			<td>68.4</td>
			<td>72.9</td>
			<td>56.6</td>
			<td>51.0</td>
			<td>96.2%</td>
			<td></td>
		</tr>
		<tr>
			<td>VisionZip</td>
			<td>57.6</td>
			<td>62.0</td>
			<td>-</td>
			<td>1763</td>
			<td>83.2</td>
			<td>68.9</td>
			<td>75.6</td>
			<td>56.8</td>
			<td>-</td>
			<td>96.2%</td>
			<td></td>
		</tr>
		<tr>
			<td>VScan (Ours)</td>
			<td>-</td>
			<td>59.8</td>
			<td>63.0</td>
			<td>58.0</td>
			<td>1792</td>
			<td>86.1</td>
			<td>68.9</td>
			<td>77.1</td>
			<td>57.3</td>
			<td>51.7</td>
			<td>98.8%</td>
			<td></td>
		</tr>
		<tr>
			<th colspan="12">Retain 64 Tokens in Average (↓ 88.9%), ~415 TFLOPs</th>
		</tr>
		<tr>
			<td>ToMe</td>
			<td>48.6</td>
			<td>43.7</td>
			<td>-</td>
			<td>1138</td>
			<td>52.5</td>
			<td>50.0</td>
			<td>57.1</td>
			<td>45.3</td>
			<td>-</td>
			<td>70.1%</td>
			<td></td>
		</tr>
		<tr>
			<td>FastV</td>
			<td>46.1</td>
			<td>48.0</td>
			<td>52.7</td>
			<td>1256</td>
			<td>48.0</td>
			<td>51.1</td>
			<td>55.0</td>
			<td>47.8</td>
			<td>50.8</td>
			<td>76.7%</td>
			<td></td>
		</tr>
		<tr>
			<td>SparseVLM </td>
			<td>52.7</td>
			<td>56.2</td>
			<td>46.1</td>
			<td>1505</td>
			<td>75.1</td>
			<td>62.2</td>
			<td>68.2</td>
			<td>51.8</td>
			<td>50.1</td>
			<td>87.2%</td>
			<td></td>
		</tr>
		<tr>
			<td>PyramidDrop</td>
			<td>47.5</td>
			<td>58.8</td>
			<td>50.5</td>
			<td>1561</td>
			<td>55.9</td>
			<td>69.2</td>
			<td>69.2</td>
			<td>50.6</td>
			<td>50.7</td>
			<td>86.6%</td>
			<td></td>
		</tr>
		<tr>
			<td>VisionZip</td>
			<td>55.1</td>
			<td>60.1</td>
			<td>-</td>
			<td>1690</td>
			<td>77.0</td>
			<td>69.0</td>
			<td>72.4</td>
			<td>55.5</td>
			<td>-</td>
			<td>92.7%</td>
			<td></td>
		</tr>
		<tr>
			<td>VScan (Ours)</td>
			<td>-</td>
			<td>58.3</td>
			<td>62.1</td>
			<td>55.7</td>
			<td>1698</td>
			<td>85.0</td>
			<td>69.1</td>
			<td>75.4</td>
			<td>55.6</td>
			<td>51.8</td>
			<td>96.7%</td>
			<td></td>
		</tr>
	</tbody>
</table>

## Comparison based on different training methods


| Model | LLM | PT/IT | Token | $VQA^T$ | $VQA^D$ | $QA^C$ | $VQA^I$ | GQA | $VQA^{v2}$ | VizWiz | MMB | MMVet | MMMU | POPE | SEED | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileVLM V2  | Mobilellama-2.7B | 1.2M/3.6M | 144 | 52.1 | - | - | - | 59.3 | - | - | - | - | - | 84.3 | - | - |
| BLIP-2 | Vicuna-13B | 129M/- | 32 | 42.5 | - | - | - | 41.0 | 65.0 | 19.6 | - | - | - | 85.3 | 49.7 | - |
| Instruct-BLIP  | Vicuna-7B | 129M/1.2M | 64 | 50.1 | - | - | - | 49.5 | 34.5 | - | 26.3 | - | - | - | - | - |
| QwenVL  | Qwen-7B | 1.4B/50M | 256 | 63.8 | 65.1 | 65.7 | - | 59.3 | 78.8 | 35.2 | - | - | - | 62.3 | - | - |
| VILA  | Llama2-7B | 50M/1M | 576 | 64.4 | - | 58.6 | - | 62.3 | 79.9 | 57.8 | 68.9 | 34.9 | - | 85.5 | - | - |
| MobileVLM V2  | Vicuna-7B | 1.2M/3.6M | 144 | 62.3 | - | - | - | 62.6 | - | - | - | - | - | 85.3 | - | - |
| Mini-Gemini | Vicuna-7B | 1.2M/1.5M | 576 | 65.9 | - | - | - | - | - | 68.5 | 46.0 | 38.1 | - | - | - | - |
| LLaVA-1.5  | Vicuna-7B | 558K/665K | 576 | 58.2 | 28.1 | - | 25.8 | 63.3 | 78.5 | 50.0 | 64.3 | 31.1 | 35.3 | 85.9 | 66.1 | - |
| TokenPacker  | Vicuna-7B | 558K/665K | 144 | - | 26.9 | 18.1 | 21.8 | 61.9 | 77.9 | 52.0 | 65.1 | 33.0 | - | 87.0 | - | - |
| InternVL2| Internlm2.5-7B | 558K/665K | 256 | 49.7 | - | - | - | 63.0 | 77.8 | 50.6 | 70.9 | 34.1 | 39.2 | 86.8 | 71.1 | 50.8 |
| **High - resolution LLMs** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Monkey  | Qwen-7B | -/1.44M | ~1024 | 67.7 | 66.5 | 36.1 | - | 60.7 | 80.3 | 61.2 | - | - | - | - | - | - |
| TokenPacker-HD  | Vicuna-7B | 1.2M/1.5M | ~954 | 68.0 | 60.2 | - | - | - | 81.2 | 54.7 | 67.4 | - | 35.4 | - | - | - |
| Mini-Gemini-HD  | Vicuna-7B | 1.2M/1.5M | 2880 | 68.4 | 65.0 | - | - | - | 80.3 | 54.6 | 65.8 | 41.3 | 36.8 | 86.8 | - | - |
| FastVITHD | Qwen-2-7B | 558K/1.1M | 256 | 64.4 | - | - | - | - | 63.1 | - | - | - | - | 88.1 | - | - |
| LLaVA-UHD  | Vicuna-13B | 595K/665K | ~256 | 67.7 | 62.6 | 56.3 | 36.8 | 63.8 | 81.7 | 56.1 | 68.0 | 42.1 | 35.5 | 89.1 | 65.6 | 60.4 |
| LLaVA-NeXT | Vicuna-7B | 558K/765K | ~2880 | 64.9 | 74.4 | 54.8 | 37.1 | 64.2 | 81.8 | 57.6 | 68.1 | 43.9 | 35.8 | 86.5 | 68.2 | 61.4 |
| InternVL2-HD  | Internlm2.5-7B | 558K/770K | ~1282 | 65.6 | 72.6 | 69.8 | 30.9 | 63.2 | 78.9 | 56.3 | 72.1 | 35.7 | 39.9 | 87.3 | 73.4 | 62.1 |
| **Ours** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [LLaVA-Meteor](./papers_summaries/2025/Top-Down_Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual_Instruction_Tuning/Top-Down_%20Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual%20Instruction_Tuning.md) <br> *compare to LLaVA-UHD* | Vicuna-13B | 595K/665K | ~256 | 69.9 <br> 100% | 64.2 <br> +2.2 | 59.0 <br> +1.6 | 39.2 <br> +2.4 | 64.9 <br> +1.1 | 82.4 <br> +0.7 | 59.3 <br> +3.2 | 69.4 <br> +1.4 | 44.7 <br> +2.6 | 37.5 <br> +2.0 | 89.9 <br> +0.8 | 67.7 <br> +2.1 | 62.4 <br> +2.0 |
| [LLaVA-Meteor](./papers_summaries/2025/Top-Down_Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual_Instruction_Tuning/Top-Down_%20Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual%20Instruction_Tuning.md) []() <br> *compare to LLaVA-UHD* | Vicuna-13B | 595K/665K | ~114 | 68.3 <br> 44.5% | 63.1 <br> +0.6 | 58.6 <br> +0.5 | 37.7 <br> +2.3 | 64.6 <br> +0.8 | 81.8 <br> +0.1 | 57.1 <br> +1.0 | 68.4 <br> +0.4 | 42.7 <br> +0.6 | 34.6 <br> -0.8 | 88.7 <br> -0.5 | 66.9 <br> +1.3 | 61.0 <br> +0.6 |
| [LLaVA-Meteor](./papers_summaries/2025/Top-Down_Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual_Instruction_Tuning/Top-Down_%20Compression_Revisit_Efficient_Vision_Token_Projection_for_Visual%20Instruction_Tuning.md) <br> *compare to LLaVA-UHD* | Vicuna-13B | 595K/665K | ~56 | 65.0 <br> 21.8% | 58.4 <br> -2.7 | 56.5 <br> -4.2 | 37.1 <br> +0.2 | 62.4 <br> +0.3 | 81.2 <br> -1.4 | 55.3 <br> -0.5 | 68.0 <br> +0.0 | 41.6 <br> -0.5 | 34.2 <br> -1.3 | 87.2 <br> -1.9 | 64.8 <br> -0.8 | 59.3 <br> -1.1  |