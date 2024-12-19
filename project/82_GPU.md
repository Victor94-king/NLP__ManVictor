# 自然语言处理:第八十二章 大模型如何估算显存

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />



在实际工作中，经常有人问，7B、14B或70B的模型需要多大的显存才能推理？如果微调他们又需要多大的显存呢？为了回答这个问题整理一份训练或推理需要显存的计算方式。如果大家对具体细节不感兴趣，可以直接参考经验法则评估推理或训练所需要的资源。更简单的方式可以通过这个工具或者huggface官网计算推理/训练需要的显存工具在线评估。

## 数据精度

开始介绍之前，先说一个重要的概念——数据精度。数据精度指的是信息表示的精细程度，在计算机中是由数据类型和其位数决定的。如果想要计算显存，从“原子”层面来看，就需要知道我们的使用数据的精度，因为精度代表了数据存储的方式，决定了一个数据占多少bit。 目前，精度主要有以下几种：

* 4 Bytes: FP32 / float32 / 32-bit
* 2 Bytes: FP16 / float16 / bfloat16 / 16-bit
* 1 Byte: int8 / 8-bit
* 0.5 Bytes: int4 / 4-bit

## 经验法则

* 推理: 参数量 * 精度。例如，假设模型都是16-bit权重发布的，也就是说一个参数消耗16-bit或2 Bytes的内存，模型的参数量为70B，基于上述经验法则，推理最低内存需要70B * 2Bytes = 140G。
* 训练: 4 - 6 倍的推理资源。

## 推理

在模型推理阶段，需要的资源主要有三部分：模型的权重、KV Cache和激活（在推理过程中创建的张量）。

## 模型权重

加载模型权重（即模型大小）占用资源主要依赖于模型的参数量和精度。其中，参数量基本不变，精度可以通过模型量化技术进行优化。尽管量化会影响模型的性能，但相比于选择更高精度的小模型来说，量化技术更受青睐。

#### 公式[1]

模型的大小 = 模型的参数量 * 精度

![图片](https://mmbiz.qpic.cn/mmbiz_png/pCFs8q3BZjeo13Q0ZR5hIxmTzIQOaSWk9OS7azaEu4bXKlKPPbLQYGx4oIO3nY69j48fjLeVSsEmcThPvVCJQA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> 十亿参数模型在 32 位、16 位和 8 位精度下所需的近似 GPU 内存[2]

### KV Cache

在Transformer的解码阶段，每次推理生成一个token，依赖于之前的token结果，如果每次都对所有token重新计算一次，代价非常大。为了避免重新计算，通过KV Cache技术将其缓存到GPU内存中。

#### 公式 [3]

KV Cache = 2 * Batch Size * Sequence Length * Number of Layers * Hidden Size * Precision 注意：第一个因子2解释了K和V矩阵。通常，在Transformer中，Hidden Size和Number of Layers的值可以在模型相关的配置文件中找到。

### 激活内存

在模型的前向传播过程中，必须存储中间激活值。这些激活值代表了神经网络中每层的数据在向前传播时的输出。它们必须保持为 FP32 格式，以避免数值爆炸并确保收敛。

#### 公式 [4]

Activation Memory = Batch Size * Sequence Length * Hidden Size * (34 + (5 * Sequence Length * Number of attention heads) / (Hidden Size))

## 训练

训练阶段所需的资源，除了上述介绍的模型权重、KV Cache和激活内存之外，还需要存储优化器和梯度状态，因此，训练比推理需要更多的资源。

### 优化器内存

优化器需要资源来存储参数和辅助变量。这些变量包括诸如Adam或SGD等优化算法使用的动量和方差等参数。这取决于优化状态的数量及其精度。例如，AdamW优化器是最流行的微调llm，它为模型的每个参数创建并存储2个新参数。如果我们有一个70B的模型，优化器将创建140B的新参数！假设优化器的参数为float32，即每个参数占用4字节的内存。优化器至少需要 140B * 4 Bytes = 516 G的资源。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/pCFs8q3BZjeo13Q0ZR5hIxmTzIQOaSWk22EMuQDmFVgR7qGVtq17SWDNVt6icAyyOmtPwHtPLDj4dtckDeVV0Xg/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中，不同优化器的状态数量如下[1]：

* AdamW (2 states): 8 Bytes per parameter
* AdamW (bitsandbytes Quantized): 2 Bytes per parameter
* SGD (1 state): 4 Bytes per parameter

### 梯度

在模型的反向传播过程中计算梯度值。它们表示损失函数相对于每个模型参数的变化率，对于在优化过程中更新参数至关重要。作为激活值，它们必须存储在 FP32 中以保持数值稳定性 [1]。 因此，每个参数占用4字节的内存 。例如，一个70B的模型，计算梯度所需的内存需要 70B * 4 Bytes = 280 G左右。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/pCFs8q3BZjeo13Q0ZR5hIxmTzIQOaSWkpb7JsZYyjc4dToI3Go5g8IGbubTz2NViaFxUsTztetJFw7R4WFmw2Sw/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 总结

在本文中，我们介绍的评估方法，都是基于Transformer架构推算的，该评估方法不适合Transformer以外的其他体系结构。同时，目前存在大量的框架、模型和优化技术，估计运行大型语言模型的确切内存可能很困难。然而，本文可作为估计执行 LLM 推理和训练所需内存资源的起点。


训练时的显存计算公式：

总显存 = 模型大小 + KV缓存 + 激活 + (优化器状态 + 梯度) * 可训练参数数量

此外在微调中，由于优化器状态和梯度的计算，显存需求会更高。

如果显存资源有限，可以通过参数高效微调（PEFT）技术，比如 LoRA 或 QLoRA，这些方法通过固定大部分模型参数，只训练少量额外参数，能够有效减少显存占用。

最后提供一个预估显存的经验法则：

推理：参数数量 * 精度（通常为 2 或 4 字节）

训练：推理资源的 4-6 倍



参考文档

[面试官扎心一问：大模型显存如何估算？](https://mp.weixin.qq.com/s/W_LiyC584qXLbwoxSBmnEg)
