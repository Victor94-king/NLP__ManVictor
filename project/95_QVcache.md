# 自然语言处理:第九十五章 为什么有KV Cache而没有Q Cache?

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


KV Cache 在已经成为了 LLM 推理系统中必须要用到的一个优化技术。

KV Cache 的想法很简单，即 **通过把中间结果缓存下来，避免重复计算** ，但是 KV Cache 有时候又是比较令人困惑的。

学习 KV Cache，搞清楚下面的三个问题就算是完全弄清楚了：

* 缓存的是什么？
* 为什么是 **KV **Cache，而不是 **QKV **Cache？
* KV Cache 对模型的**输出有任何的影响**吗？

如果我们从矩阵乘法的角度观察 attention 运算的过程，那么上面的三个问题很容易理解。

## 01 **自回归模型的特点**

LLM 的推理被称为是自回归的过程，也就是说模型上一步的输出会被当作是下一步的输出。

如下面的动图所示，用户输入的 prompt 是"recite the first law $"。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/44YR2rIeKheb3aQCXQggCZs0xqMArWRQ8YSqJiaKfQposriczficzxT2Olic2S7g7iaVbJMxEUVtmEdRBVNaMYZPFzQ/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1)

模型产生的第一个 token 是"A"，然后输出的"A"会添加到用户的 prompt 后面，再次输入到模型中，模型输出"robot"这个 token。如此反复，直到模型输出结束词。

我们把运行一次模型成为一个  **step** ，每一个 **step **生成一个新的 token。

## 02 **KV Cache 的探索过程**

下面，我们从最朴素的推理方法出发，一步一步推理出 KV Cache 的优化方法。假设，我们的模型有两个 attention 层。

不考虑任何 Cache 的推理：

Step 0

我们先不考虑任何 Cache，观察我们的推理情况。 如图 1 所示，我们以“你好”作为 prompt 输入到模型中。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQJ2Ju0cgvFfgxHGlxqouycX0uLFjjujViczFJk8fh3v9bicNLJ6leupDw/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图1

模型的 embedding 层会将 token 转化为对应的 embedding(由于篇幅原因，embedding 层就没有展示在图 1 中) ，得到一个 2x4 的矩阵 embedding1，其中第一行是“你”的 embedding 表示，第二行是“好”的 embedding 表示。

之后，我们进入到第一个 attention 层。embedding1 会与 attention 层中的三个权重矩阵相乘，分别得到 Q、K 和 V 矩阵，如图 2 所示。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQLqqHibVCC54mr4Sn7RnoFPqQ7rtOaEnsIud6Y3apqTVpThvpoDqw0ZQ/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图2

在 attention layer 中，我们根据 attention 层的计算公式对 Q、K 和 V 矩阵进行计算。

首先，Q 和 K 矩阵相乘，得到一个 attention 矩阵，其中包括了每两个词之间的注意力值。比如，第二行第一列表示“好”这个 token 对“你”这个 token 的注意力值。

需要额外注意的是，“你”这个 token 对“好”这个 token 的注意力值被 mask 了，即 **每一个 token 只计算与之前的 token 的注意力值** ，而不计算之后的 token 的注意力值。

然后，attention 矩阵与 V 矩阵相乘，再把相乘结果输入到 FFN 层中(这里的 FFN 对我们要讨论的 KV Cache 没有什么影响，所以可以忽略)，就得到了新的 embedding2。

第二个 attenion 层也是如此类推，并输出 embedding3。

最后，embedding3 的最后一行，也就是“好”这个 token 对应的 embedding 被输入到一个分类其中，预测下一个 token。得到的预测结果是“啊“。到此，我们就生成第一个 token，即完成了 step0。

**Step 1**

在预测了 token“啊”之后，我们将“你好啊”再次输入到模型中，如图 3 所示。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQSOg70OXSSEYBUwrshZXmXocmGrjvibb1Bt4NxtWFjKX5nDiaQf9vMxyg/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图3

除了我们的 token 数量从 2 变成了 3 之外，需要执行的运算与 Step 0 完全相同，输入的 embedding 经过两层 attention 层之后，得到 embedding3。embedding3 的最后一行被输入到分类器中，预测得到 token“!”。

一个细节是，Step 1 和 Step 0 的红色的部分是相同的。比如说，Step 1 的 K1 矩阵前两行与 Step 0 的 K1 矩阵的前两行是相同的。

这是因为 K1 矩阵由 embedding1 矩阵和模型的权重矩阵相乘得到，而两个 Step 的权重矩阵是相同的，唯一不同的地方在于 Step 1 的 embedding1 比 Step 0 多了最后一行，所以两个 Step 的 QKV 矩阵的前两行是相同的，如下图所示。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQeT2ZpMvOiaVvp2bzsyLhp2TXsj6d7t8XCtHnz83CD4s6Tzx7dXCFiagw/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**图4**

**删除无效计算**

观察图 1 和图 3 可以发现，模型之后的 Classifier 只会**使用 embedding3 矩阵的最后一行**作为输入，并输出预测结果。

由此我们可以发现，实际上，每一个 embedding 矩阵只有最新输入的 token 的行是有效的。

比如，在图 3 的 embedding3 中，只有 step 0 产生的新 token“啊”对应的第三行是有用的，前两行都是无效的。因此，我们可以删除这些无用的行，以节约计算和存储开销。

图 5 以 Step1 的第一个 attention layer 作为演示。我们首先删除三个 embedding 矩阵的红色的部分。

因为 embedding2 和 embedding3 矩阵是通过 attention 和 V1 矩阵相乘得到的，所以我们也需要删除 attention 矩阵的重复部分。

而 attention 矩阵是由 Q 矩阵和 K 矩阵相乘得到，所以我们还需要删除 Q 矩阵的重复部分。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQYufPrM0vz7EkxH1A1uFMkicyCsSw1HHHCTX1MNbouiaibdywOHke6ctiaA/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图5

所以，通过删除无效计算，我们的模型推理过程可以用图 6 表示。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQmz1BbZ0sCkJ4KKg5ObqFtlicNK4m6xQb2mzxHLEZ8COEVPftrXZbc1A/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图6

缓存重复计算

经过仔细观察我们可以发现，在图 6 中，Step 1 和 Step 2 的红色部分是完全相同的，只有白色的部分是不相同的。

比如说，Step 1 的 Q1 矩阵、K1 矩阵和 V1 矩阵和 Step 0 的 Q1 矩阵、K1 矩阵和 V1 矩阵的前两行是完全相同的。

那么我们其实完全可以在 Step 0 的时候把这些矩阵 Cache 下来，然后在 Step 1 的时候 load 进来，从而节省大量的计算。

下面是进行缓存之后的计算，如图 7 所示。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/44YR2rIeKheb3aQCXQggCZs0xqMArWRQnzUzLGAJGh97oh36PJK796NeyXa1JaZYPemjWaNh09SWgFXBaHxumg/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图7

在 Step 0 中，我们把红色部分的数据 Cache 下来，然后在 Step 1 的 load 进行计算。

在图 7 的 Step 1 中，红色部分的数据都是历史缓存，只有白色部分是新产生的数据。

我们将缓存数据和新产生的数据拼接起来，即可得到一个新的矩阵。当然，在 Step 1 中，我们也需要缓存新产生的数据，以供后面的 Step 使用。

## 03 **总结**

在 KV Cache 中，我们把第一个 Step，即 Step 0，称为是 Prefill 阶段，因为这个阶段是在“填充”KV Cache。而之后的所有 Step 称为是 Decode 阶段。

通过观察我们可以发现，Prefill 阶段是是 Compute bound，即计算密集型的，因为这个阶段我们的 embedding 矩阵都特别大，需要进行大量的矩阵乘法计算；

而 Decode 阶段是 Memory bound，即访存密集型的，因为这个阶段需要从显存乃至内存中加载 KV Cache。

大模型推理系统基本上要解决下面的问题：

如何加速 Prefill 阶段的计算，比如，现在很多工作使用预查找表加速矩阵乘法计算。

如何应对 Decode 阶段的访存开销，比如一些工作尝试使用 CSD(Computational Storage Device)缓解推理过程的 Memory bound，提高推理的吞吐率。

推理任务的调度问题。因为 LLM 的是自回归的，所以我们不知道模型究竟需要执行多久。

并且，模型的输出长度越长，KV Cache 占据的存储空间就越多。所以，如何在多个服务器实例之间调度推理任务是非常 tricky 的。之后阿里今年新出的论文 Llumnix 要解决的就是 LLM 的推理任务调度的问题。

由此，我们可以回答开始提到的三个问题：

（1）缓存的是什么？

缓存的是 K 和 V 矩阵。

（2）为什么是 KV Cache，而不是 QKV Cache？

如图 5 所示，我们只需要 embedding 的最后一行，所以我们也只需要 attention 的最后一行，因此只需要 Q 矩阵的最后一行。

所以，不缓存 Q 矩阵的原因是没必要缓存，我们只需要 Q 矩阵的最后一行即可。

（3）KV Cache 对模型的输出有任何的影响吗？

因为输入到 Classifier 的 embedding 是没有改变的，所以模型的输出也不会有任何变化。
