# 自然语言处理:第四十四章 微软开源的GraphRAG爆火，Github Star量破万，生成式AI进入知识图谱时代？(转载)

项目链接: [microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system (github.com)](https://github.com/microsoft/graphrag)

官网简介: [GraphRAG: Unlocking LLM discovery on narrative private data - Microsoft Research](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

原文地址: [微软开源的GraphRAG爆火，Github Star量破万，生成式AI进入知识图谱时代？ (qq.com)](https://mp.weixin.qq.com/s/BX93FvDzW7WVLK66V2usBw)

<br />

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***


LLM 很强大，但也存在一些明显缺点，比如幻觉问题、可解释性差、抓不住问题重点、隐私和安全问题等。检索增强式生成（RAG）可大幅提升 LLM 的生成质量和结果有用性。

本月初，微软发布最强 RAG 知识库开源方案 GraphRAG，项目上线即爆火，现在星标量已经达到 10.5 k。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltbMWDdbmb99kml8XcT023KKibUQl5t3BPXkAxbPzrj6icIgsTa0Jibibs3A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

有人表示，它比普通的 RAG 更强大：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltfBIYOv4zzjMFrhibFCjFG5iaOxMic9nxiaVg6QdLVh9Wp9MlOlpJNB2AOw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

GraphRAG 使用 LLM 生成知识图谱，在对复杂信息进行文档分析时可显著提高问答性能，尤其是在处理私有数据时。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltzgILdyHV8NLRr2aoUUoON7sfcx985OBdsNE8xxNawoqCCNicZuQJXOg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*GraphRAG 和传统 RAG 对比结果*

现如今，RAG 是一种使用真实世界信息改进 LLM 输出的技术，是大多数基于 LLM 的工具的重要组成部分，一般而言，RAG 使用向量相似性作为搜索，称之为 Baseline RAG（基准RAG）。但 Baseline RAG 在某些情况下表现并不完美。例如：

* Baseline RAG 难以将各个点连接起来。当回答问题需要通过共享属性遍历不同的信息片段以提供新的综合见解时，就会发生这种情况；
* 当被要求全面理解大型数据集甚至单个大型文档中的总结语义概念时，Baseline RAG 表现不佳。

微软提出的 GraphRAG 利用 LLM 根据输入的文本库创建一个知识图谱。这个图谱结合社区摘要和图机器学习的输出，在查询时增强提示。GraphRAG 在回答上述两类问题时显示出显著的改进，展现了在处理私有数据集上超越以往方法的性能。

不过，随着大家对 GraphRAG 的深入了解，他们发现其原理和内容真的让人很难理解。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlt9WUTZFPHVwgyiaLSddnbzHDMywdicxMaOPYRRFKicT7Hq8gCmucHAJiceA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

近日，Neo4j 公司 CTO Philip Rathle 发布了一篇标题为《GraphRAG 宣言：将知识加入到生成式 AI 中》的博客文章，Rathle 用通俗易懂的语言详细介绍了 GraphRAG 的原理、与传统 RAG 的区别、GraphRAG 的优势等。

他表示：「你的下一个生成式 AI 应用很可能就会用上知识图谱。」

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlteqbvWbn9UvYppQvmqf5GFPDDmgL96OfljRqU3kZiaJkibd5TfNXK7jKg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*Neo4j CTO Philip Rathle*

下面来看这篇文章。

我们正在逐渐认识到这一点：要使用生成式 AI 做一些真正有意义的事情，你就不能只依靠自回归 LLM 来帮你做决定。

我知道你在想什么：「用 RAG 呀。」或者微调，又或者等待 GPT-5。

是的。基于向量的检索增强式生成（RAG）和微调等技术能帮到你。而且它们也确实能足够好地解决某些用例。但有一类用例却会让所有这些技术折戟沉沙。

针对很多问题，基于向量的 RAG（以及微调）的解决方法本质上就是增大正确答案的概率。但是这两种技术都无法提供正确答案的确定程度。它们通常缺乏背景信息，难以与你已经知道的东西建立联系。此外，这些工具也不会提供线索让你了解特定决策的原因。

让我们把视线转回 2012 年，那时候谷歌推出了自己的第二代搜索引擎，并发布了一篇标志性的博客文章《Introducing the Knowledge Graph: things, not strings》。他们发现，如果在执行各种字符串处理之外再使用知识图谱来组织所有网页中用字符串表示的事物，那么有可能为搜索带来飞跃式的提升。

现在，生成式 AI 领域也出现了类似的模式。很多生成式 AI 项目都遇到了瓶颈，其生成结果的质量受限于这一事实：解决方案处理的是字符串，而非事物。

快进到今天，前沿的 AI 工程师和学术研究者们重新发现了谷歌曾经的发现：打破这道瓶颈的秘诀就是知识图谱。换句话说，就是将有关事物的知识引入到基于统计的文本技术中。其工作方式就类似于其它 RAG，只不过除了向量索引外还要调用知识图谱。也就是：GraphRAG！（GraphRAG = 知识图谱 + RAG）

本文的目标是全面且易懂地介绍 GraphRAG。研究表明，如果将你的数据构建成知识图谱并通过 RAG 来使用它，就能为你带来多种强劲优势。有大量研究证明，相比于仅使用普通向量的 RAG，GraphRAG 能更好地回答你向 LLM 提出的大部分乃至全部问题。

单这一项优势，就足以极大地推动人们采用 GraphRAG 了。

但还不止于此；由于在构建应用时数据是可见的，因此其开发起来也更简单。

GraphRAG 的第三个优势是人类和机器都能很好地理解图谱并基于其执行推理。因此，使用 GraphRAG 构建应用会更简单轻松，并得到更好的结果，同时还更便于解释和审计（这对很多行业来说至关重要）。

我相信 GraphRAG 将取代仅向量 RAG，成为大多数用例的默认 RAG 架构。本文将解释原因。

**图谱是什么？**

首先我们必须阐明什么是图谱。

图谱，也就是 graph，也常被译为「图」，但也因此容易与 image 和 picture 等概念混淆。本文为方便区分，仅采用「图谱」这一译法。

图谱大概长这样：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltlhLiaNpPmWSGv2kYCPUSqDwNu73bAHmtPCBIrCTia8vg9ichic58wOw2Bw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*图谱示例*

尽管这张图常作为知识图谱的示例，但其出处和作者已经不可考。

或这样：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltNiaRvJvsd0YSJo21y2jwBEbZ7YTGtjpNBHZTYqhosdzM3OyGIxEGLJA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*《权力的游戏》人物关系图谱，来自 William Lyon*

或这样：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltK7Lib9tcmClMSlqhicicgbicRo5pp4xjU8qSCYwIJEZftAAQsjqFQGicekg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*伦敦地铁地图。有趣小知识：伦敦交通局前段时间部署了一个基于图谱的数字孪生应用，以提升事故响应能力并减少拥堵。*

换句话说，图谱不是图表。

这里我们就不过多纠结于定义问题，就假设你已经明白图谱是什么了。

如果你理解上面几张图片，那么你也许能看出来可以如何查询其底层的知识图谱数据（存储在图谱数据库中），并将其用作 RAG 工作流程的一部分。也就是 GraphRAG。

**两种呈现知识的形式：向量和图谱**

典型 RAG 的核心是向量搜索，也就是根据输入的文本块从候选的书面材料中找到并返回概念相似的文本。这种自动化很好用，基本的搜索都大有用途。

但你每次执行搜索时，可能并未思考过向量是什么或者相似度计算是怎么实现的。下面我们来看看 Apple（苹果）。它在人类视角、向量视角和图谱视角下呈现出了不同的形式：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltY6Jwa5lf4H144TdvWDbF2GZbBYmtc39bC6xaQ4v49F2icOPBmCVzAnQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*人类视角、向量视角和图谱视角下的 Apple*

在人类看来，苹果的表征很复杂并且是多维度的，其特征无法被完整地描述到纸面上。这里我们可以充满诗意地想象这张红彤彤的照片能够在感知和概念上表示一个苹果。

这个苹果的向量表示是一个数组。向量的神奇之处在于它们各自以编码形式捕获了其对应文本的本质。但在 RAG 语境中，只有当你需要确定一段文本与另一段文本的相似度时，才需要向量。为此，只需简单地执行相似度计算并检查匹配程度。但是，如果你想理解向量内部的含义、了解文本中表示的事物、洞察其与更大规模语境的关系，那使用向量表示法就无能为力了。

相较之下，知识图谱是以陈述式（declarative）的形式来表示世界 —— 用 AI 领域的术语来说，也就是符号式（symbolic）。因此，人类和机器都可以理解知识图谱并基于其执行推理。这很重要，我们后面还会提到。

此外，你还可以查询、可视化、标注、修改和延展知识图谱。知识图谱就是世界模型，能表示你当前工作领域的世界。

**GraphRAG 与 RAG**

这两者并不是竞争关系。对 RAG 来说，向量查询和图谱查询都很有用。正如 LlamaIndex 的创始人 Jerry Liu 指出的那样：思考 GraphRAG 时，将向量囊括进来会很有帮助。这不同于「仅向量 RAG」—— 完全基于文本嵌入之间的相似度。

根本上讲，GraphRAG 就是一种 RAG，只是其检索路径包含知识图谱。下面你会看到，GraphRAG 的核心模式非常简单。其架构与使用向量的 RAG 一样，但其中包含知识图谱层。

**GraphRAG 模式**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlt3gDJB6vDvKbNQ7lDeUZSGE6EwAWy8MPNzKAshjqwT4usLO0jfGZsMg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)*GraphRAG 的一种常用模式*

可以看到，上图中触发了一次图谱查询。其可以选择是否包含向量相似度组件。你可以选择将图谱和向量分开存储在两个不同的数据库中，也可使用 Neo4j 等支持向量搜索的图谱数据库。

下面给出了一种使用 GraphRAG 的常用模式：

1. 执行一次向量搜索或关键词搜索，找到一组初始节点；
2. 遍历图谱，带回相关节点的信息；

3.（可选）使用 PageRank 等基于图谱的排名算法对文档进行重新排名

用例不同，使用模式也会不一样。和当今 AI 领域的各个研究方向一样，GraphRAG 也是一个研究丰富的领域，每周都有新发现涌现。

**GraphRAG 的生命周期**

使用 GraphRAG 的生成式 AI 也遵循其它任意 RAG 应用的模式，一开始有一个「创建图谱」步骤：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltsvCxBHLWIEiapbOGCWOPHXysof0taiaQbOsBVcEGgOibH7zNhAUPbwXrg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*GraphRAG 的生命周期*

创建图谱类似于对文档进行分块并将其加载到向量数据库中。工具的发展进步已经让图谱创建变得相当简单。这里有三个好消息：

1. 图谱有很好的迭代性 —— 你可以从一个「最小可行图谱」开始，然后基于其进行延展。
2. 一旦将数据加入到了知识图谱中，就能很轻松地演进它。你可以添加更多类型的数据，从而获得并利用数据网络效应。你还可以提高数据的质量，以提升应用的价值。
3. 该领域发展迅速，这就意味着随着工具愈发复杂精妙，图谱创建只会越来越容易轻松。

在之前的图片中加入图谱创建步骤，可以得到如下所示的工作流程：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltQ96tTnWvDIpU9pNvyTrXTkcBbyJGFpG7KnqiaswaR8n2EtluJxOCMicg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*添加图谱创建步骤*

下面来看看 GraphRAG 能带来什么好处。

**为什么要使用 GraphRAG？
**

相较于仅向量 RAG，GraphRAG 的优势主要分为三大类：

1. 准确度更高且答案更完整（运行时间 / 生产优势）
2. 一旦创建好知识图谱，那么构建和维护 RAG 应用都会更容易（开发时间优势）
3. 可解释性、可追溯性和访问控制方面都更好（治理优势）

下面深入介绍这些优势。

**1. 准确度更高且答案更有用**

GraphRAG 的第一个优势（也是最直接可见的优势）是其响应质量更高。不管是学术界还是产业界，我们都能看到很多证据支持这一观察。

比如这个来自数据目录公司 Data.world 的示例。2023 年底，他们发布了一份研究报告，表明在 43 个业务问题上，GraphRAG 可将 LLM 响应的准确度平均提升 3 倍。这项基准评测研究给出了知识图谱能大幅提升响应准确度的证据。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltFJJzdKxsIa7N5GahRGApKP07aIhEHcGcCEIee9HCtD6VgaribUarYMg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*知识图谱将 LLM 响应的准确度提升了 54.2 个百分点，也就是大约提升了 3 倍*

微软也给出了一系列证据，包括 2024 年 2 月的一篇研究博客《GraphRAG: Unlocking LLM discovery on narrative private data》以及相关的研究论文《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》和软件：https://github.com/microsoft/graphrag（即上文开篇提到的 GraphRAG）。

其中，他们观察到使用向量的基线 RAG 存在以下两个问题：

* 基线 RAG 难以将点连接起来。为了综合不同的信息来获得新见解，需要通过共享属性遍历不同的信息片段，这时候，基线 RAG 就难以将不同的信息片段连接起来。
* 当被要求全面理解在大型数据集合甚至单个大型文档上归纳总结的语义概念时，基线 RAG 表现不佳。

微软发现：「通过使用 LLM 生成的知识图谱，GraphRAG 可以大幅提升 RAG 的「检索」部分，为上下文窗口填入相关性更高的内容，从而得到更好的答案并获取证据来源。」他们还发现，相比于其它替代方法，GraphRAG 所需的 token 数量可以少 26% 到 97%，因此其不仅能给出更好的答案，而且成本更低，扩展性也更好。

进一步深入准确度方面，我们知道答案正确固然重要，但答案也要有用才行。人们发现，GraphRAG 不仅能让答案更准确，而且还能让答案更丰富、更完整、更有用。

领英近期的论文《Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering》就是一个出色的范例，其中描述了 GraphRAG 对其客户服务应用的影响。GraphRAG 提升了其客户服务答案的正确性和丰富度，也因此让答案更加有用，还让其客户服务团队解决每个问题的时间中位数降低了 28.6%。

Neo4j 的生成式 AI 研讨会也有一个类似的例子。如下所示，这是针对一组 SEC 备案文件，「向量 + GraphRAG」与「仅向量」方法得到的答案：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltVcatCfcNw4KaUWNxjWS0CCxucffia2PGcnTzx3k41YQn5mbwibKPUwKg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*「仅向量」与「向量 + GraphRAG」方法对比*

请注意「描述可能受锂短缺影响的公司的特征」与「列出可能受影响的具体公司」之间的区别。如果你是一位想要根据市场变化重新平衡投资组合的投资者，或一家想要根据自然灾害重新调整供应链的公司，那么上图右侧的信息肯定比左侧的重要得多。这里，这两个答案都是准确的。但右侧答案明显更有用。

Jesus Barrasa 的《Going Meta》节目第 23 期给出了另一个绝佳示例：从词汇图谱开始使用法律文件。

我们也时不时会看到来自学术界和产业界的新示例。比如 Lettria 的 Charles Borderie 就给出了一个「仅向量」与「向量 + GraphRAG」方法的对比示例；其中 GraphRAG 依托于一个基于 LLM 的文本到图谱工作流程，将 10,000 篇金融文章整理成了一个知识图谱：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlt43ibmv9KYywwschGndic1djicic48MIf646raggyeVwcSjKudDEnavAeTg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*仅检索器方法与图检索器方法的对比*

可以看到，相比于使用普通 RAG，使用 GraphRAG 不仅能提升答案的质量，并且其答案的 token 数量也少了三分之一。

再举一个来自 Writer 的例子。他们最近发布了一份基于 RobustQA 框架的 RAG 基准评测报告，其中对比了他们的基于 GraphRAG 的方法与其它同类工具。GraphRAG 得到的分数是 86%，明显优于其它方法（在 33% 到 76% 之间），同时还有相近或更好的延迟性能。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlt85rlvibFypkcj2wiaDjZyDJV53ibd0xTYiad33ShlUQ3HXiaGToBUOR7Yeg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*RAG 方法的准确度和响应时间评估结果*

GraphRAG 正在给多种多样的生成式 AI 应用带去助益。知识图谱打开了让生成式 AI 的结果更准确和更有用的道路。

**2. 数据理解得到提升，迭代速度更快**

不管是概念上还是视觉上，知识图谱都很直观。探索知识图谱往往能带来新的见解。

很多知识图谱用户都分享了这样的意外收获：一旦投入心力完成了自己的知识图谱，那么它就能以一种意想不到的方式帮助他们构建和调试自己的生成式 AI 应用。部分原因是如果能以图谱的形式看待数据，那便能看到这些应用底层的数据呈现出了一副生动的数据图景。

图谱能让你追溯答案，找到数据，并一路追溯其因果链。

我们来看看上面有关锂短缺的例子。如果你可视化其向量，那么你会得到类似下图的结果，只不过行列数量都更多。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltcd3yrAg0Z5NBNuG7Ro6ialBgicICpUbMDvicV32R4T3qPBAibha5AMRp5Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*向量可视化*

而如果将数据转换成图谱，则你能以一种向量表示做不到的方式来理解它。

以下是 LlamaIndex 最近的网络研讨会上的一个例子，展示了他们使用「MENTIONS（提及）」关系提取向量化词块（词汇图谱）和 LLM 提取实体（领域图谱）的图谱并将两者联系起来的能力：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltqxCkOnEk1T6WdVEic2DymYev1yg63KfzTlObYD9WQA0bksrXOgCD9OQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*提取词汇图谱和领域图谱*

（也有很多使用 Langchain、Haystack 和 SpringAI 等工具的例子。）

你可以看到此图中数据的丰富结构，也能想象其所能带来的新的开发和调试可能性。其中，各个数据都有各自的值，而结构本身也存储和传达了额外的含义，你可将其用于提升应用的智能水平。

这不仅是可视化。这也是让你的数据结构能传达和存储意义。下面是一位来自一家著名金融科技公司的开发者的反应，当时他们刚把知识图谱引入 RAG 工作流程一周时间：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltm3SZsh2D74TIoDL84ppMSicG8z9xic6NYUcV7SrUFoSY7Xf8Z2mHbymg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*开发者对 GraphRAG 的反应*

这位开发者的反应非常符合「测试驱动的开发」假设，即验证（而非信任）答案是否正确。就我个人而言，如果让我百分之百地将自主权交给决策完全不透明的 AI，我会感到毛骨悚然。更具体而言，就算你不是一个 AI 末日论者，你也会同意：如果能不将与「Apple, Inc.」有关的词块或文档映射到「Apple Corps」（这是两家完全不一样的公司），确实会大有价值。由于推动生成式 AI 决策的最终还是数据，因此可以说评估和确保数据正确性才是最至关重要的。

**3. 治理：可解释性、安全及更多**

生成式 AI 决策的影响越大，你就越需要说服在决策出错时需要最终负责的人。这通常涉及到审计每个决策。这就需要可靠且重复的优良决策记录。但这还不够。在采纳或放弃一个决策时，你还需要解释其背后的原因。

LLM 本身没法很好地做到这一点。是的，你可以参考用于得到该决策的文档。但这些文档并不能解释这个决策本身 —— 更别说 LLM 还会编造参考来源。知识图谱则完全在另一个层面上，能让生成式 AI 的推理逻辑更加明晰，也更容易解释输入。

继续来看上面的一个例子：Lettria 的 Charles 将从 10,000 篇金融文章提取出的实体载入到了一个知识图谱中，并搭配一个 LLM 来执行 GraphRAG。我们看到这确实能提供更好的答案。我们来看看这些数据：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltsmnAG98AVzqbWLTF3Sa9zU79VnPRvRPpG7yKVlg2LiamyOpLTbNcUSQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*将从 10,000 篇金融文章提取出的实体载入知识图谱*

首先，将数据看作图谱。另外，我们也可以导览和查询这些数据，还能随时修正和更新它们。其治理优势在于：查看和审计这些数据的「世界模型」变得简单了很多。相较于使用同一数据的向量版本，使用图谱让最终负责人更可能理解决策背后的原因。

在确保质量方面，如果能将数据放在知识图谱中，则就能更轻松地找到其中的错误和意外并且追溯它们的源头。你还能在图谱中获取来源和置信度信息，然后将其用于计算以及解释。而使用同样数据的仅向量版本根本就无法做到这一点，正如我们之前讨论的那样，一般人（甚至不一般的人）都很难理解向量化的数据。

知识图谱还可以显著增强安全性和隐私性。

在构建原型设计时，安全性和隐私性通常不是很重要，但如果要将其打造成产品，那这就至关重要了。在银行或医疗等受监管的行业，任何员工的数据访问权限都取决于其工作岗位。

不管是 LLM 还是向量数据库，都没有很好的方法来限制数据的访问范围。知识图谱却能提供很好的解决方案，通过权限控制来规范参与者可访问数据库的范围，不让他们看到不允许他们看的数据。下面是一个可在知识图谱中实现细粒度权限控制的简单安全策略：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltia8DhUBf3bcesbPDR6jS8SaaTFCTTbnvk41guNbciafgpJ9uhoEwSOcw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*可在知识图谱中实现的一种简单安全策略*

**创建知识图谱**

构建知识图谱需要什么？第一步是了解两种与生成式 AI 应用最相关的图谱。

领域图谱（domain graph）表示的是与当前应用相关的世界模型。这里有一个简单示例：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltBnicdebGIHY7fyA1LIWhw3pStXApKeNjfNtOgSiaYKk19Q6USld3dMvg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*领域图谱*

词汇图谱（lexical graph）则是文档结构的图谱。最基本的词汇图谱由词块构成的节点组成：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltrAyWYIGcWiaFszCksqwYWqGRoBnd9SRBTiboBjJmx0VSGFL9Ioqyv5lw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*词汇图谱*

人们往往会对其进行扩展，以包含词块、文档对象（比如表格）、章节、段落、页码、文档名称或编号、文集、来源等之间的关系。你还可以将领域图谱和词汇图谱组合到一起，如下所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqlt0NDgdzmsWJNKmVke0fIfcBeakaAPic2r76NY6uvuXkzBlH3xU7CZpBg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*将领域层和词汇层组合起来*

词汇图谱的创建很简单，主要就是简单的解析和分块。至于领域图谱，则根据数据来源（来自结构化数据源还是非结构化数据源或者两种来源都有）的不同，有不同的创建路径。幸运的是，从非结构化数据源创建知识图谱的工具正在飞速发展。

举个例子，新的 Neo4j Knowledge Graph Builder 可以使用 PDF 文档、网页、YouTube 视频、维基百科文章来自动创建知识图谱。整个过程非常简单，点几下按钮即可，然后你就能可视化和查询你输入的文本的领域和词汇图谱。这个工具很强大，也很有趣，能极大降低创建知识图谱的门槛。

至于结构化数据（比如你的公司存储的有关客户、产品、地理位置等的结构化数据），则能直接映射成知识图谱。举个例子，对于最常见的存储在关系数据库中的结构化数据，可以使用一些标准工具基于经过验证的可靠规则将关系映射成图谱。

**使用知识图谱**

有了知识图谱后，就可以做 GraphRAG 了，为此有很多框架可选，比如 LlamaIndex Property Graph Index、Langchain 整合的 Neo4j 以及 Haystack 整合的版本。这个领域发展很快，但现在编程方法正在变得非常简单。

在图谱创建方面也是如此，现在已经出现了 Neo4j Importer（可通过图形化界面将表格数据导入和映射为图谱）和前面提到的 Neo4j Knowledge Graph Builder 等工具。下图总结了构建知识图谱的步骤。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltMNdKE45w8INaU62VAgKibZ6JGwTCqjG1epdCIDnLYgn2rE8yVf85Picg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*自动构建用于生成式 AI 的知识图谱*

使用知识图谱还能将人类语言的问题映射成图谱数据库查询。Neo4j 发布了一款开源工具 NeoConverse，可帮助使用自然语言来查询知识图谱：https://neo4j.com/labs/genai-ecosystem/neoconverse/

虽然开始使用图谱时确实需要花一番功夫来学习，但好消息是随着工具的发展，这会越来越简单。

**总结：GraphRAG 是 RAG 的必定未来**

LLM 固有的基于词的计算和语言技能加上基于向量的 RAG 能带来非常好的结果。为了稳定地得到好结果，就必须超越字符串层面，构建词模型之上的世界模型。同样地，谷歌发现为了掌握搜索能力，他们就必须超越单纯的文本分析，绘制出字符串所代表的事物之间的关系。我们开始看到 AI 世界也正在出现同样的模式。这个模式就是 GraphRAG。

技术的发展曲线呈现出 S 型：一项技术达到顶峰后，另一项技术便会推动进步并超越前者。随着生成式 AI 的发展，相关应用的要求也会提升 —— 从高质量答案到可解释性再到对数据访问权限的细粒度控制以及隐私和安全，知识图谱的价值也会随之愈发凸显。


![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibhVv7TGHZypZRpiavBHtqltbIKqVLeavq7dr3heERmM66qshLwzoN32M2YSh8Q0ga79Imia0HBwmWg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*生成式 AI 的进化*

你的下一个生成式 AI 应用很可能就会用上知识图谱。
