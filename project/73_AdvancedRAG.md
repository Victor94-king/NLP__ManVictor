# 自然语言处理:第七十三章 优化RAG的十大Trick

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


随着大型语言模型（LLMs）在各种场景中的广泛应用，其在提供信息的准确性和可靠性方面面临挑战。

检索增强型生成（RAG）技术应运而生，通过结合预训练模型的生成能力和基于检索的模型，为提高模型性能提供了一种有效的框架。

RAG技术特别适用于特定领域的应用快速部署，而无需更新模型参数。尽管RAG方法在理论上具有巨大潜力，但在实际实施中仍存在复杂的实施和响应时间过长的问题。

## RAG工作流程

RAG工作流程通常包含的多个处理步骤，如查询分类、检索、重排、重组、摘要等，并且每个步骤的实施方式又有很多策略可以选择。如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWK0oBxAKVowaicqsH3CdwZqUib7LFVCBZ8yu5Cf0JBMAnicZZeh8jhzWWYw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

先展示一下RAG各个步骤采用的有效策略的得分对比

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKhInLS4LFZPzgEUSJic7gLhlKjd7kthAYhyibfibaprGV6EicPwA6lKyHXw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


### 一、查询分类（Query Classification）

并非所有查询都需要检索增强型处理，因为大型语言模型（LLMs）本身具有固有的能力。虽然RAG可以提高信息的准确性并减少幻觉（hallucinations），频繁的检索可能会增加响应时间。因此，首先对查询进行分类，以确定检索的必要性。

需要检索的查询将通过RAG模块进行处理，其他查询则直接由LLMs处理。

因此，建议按任务类型进行分类，以确定查询是否需要检索。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWK0UHMRcd0wx9ibqN6OjyQHmFrgwDnN0ldySXUCHiaW6ibYtue0Bb0gEFWQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

上图展示的例子给出了，对于完全基于用户提供信息的任务，定义为“信息足够”了，所以不需要检索；否则，“信息不足”可能需要检索。

训练一个Bert分类模型，可以很好的对查询分类。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKAXpIv3UcpGD22w2ibELiaOEMb0eKn1Kq0vJaaFc67HF7sVAricyDq3iaLA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

同样我们也可以看到，查询分类后，对效果提升还是非常明显的

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKrBQBvTTutTZWibfY1hvndyice3K2ECL0hmy3dr5MbLr0KJPtN4vDjWeQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


<br />


<br />


### 二、文档分块（Chunking）

将文档分块成较小的段落对于提高检索精度和避免大型语言模型（LLMs）中的长文本问题至关重要。这一过程可以应用于不同粒度级别，如Token（词元）、句子和语义层面。

* Token-level Chunking：词元级别的分块很直接，但可能会分割句子，影响检索质量
* Semantic-level Chunking：语义级别的分块使用LLMs来确定断点，保留了上下文但耗时。
* Sentence-level Chunking：句子级别的分块在保持文本语义的同时，平衡了简单性和效率。

因此，采用句子级别的分块是比较优的选择。那么，我们如何确定分块策略呢？


<br />


<br />


#### 2.1 分块最佳的长度（chunk size）

**翻译：** 分块大小显著影响性能。较大的分块提供更多上下文，增强理解力，但增加了处理时间。较小的分块可以提高检索召回率并减少时间，但可能缺乏足够的上下文。

因此，找到最佳的分块大小需要在忠实度（Faithfulness）和相关性（Relevancy）等指标之间取得平衡。

* 忠实度衡量响应是否是幻觉，或者是否与检索到的文本匹配；
* 相关性衡量检索到的文本和响应是否与查询匹配

这两个指标可以用LlamaIndex的评估模块计算。这个值还是要根据知识库动态的去调整，经验值大概在500+ 。


<br />


<br />


#### 2.2 分块技术（Chunking Techniques）——滑动窗口更胜一筹

在得到最佳的分块大小之后，下面就该探讨分块策略了。

Small-to-big和滑动窗口（sliding window）两种分块技术有助于提升检索质量。使用小尺寸的分块（包含175个token）来匹配查询，同时返回包含这些小分块以及上下文信息的更大尺寸的分块（包含512个token， 且分块重叠20个token）。

对比发现，滑动窗口（sliding window）更胜一筹。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWK6HTIviad0JgGHJps73ia82Ntdr0AibQIEibicibNsuXsGIqtQm9RHUPSKzIA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


<br />


### 三、 向量模型的选择（Embedding Model Selection）

选择正确的向量模型，有助于提升query的语义匹配效果。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKicyRRsNCknFpJ6xicA1j7zhCU9g2FWgersPqu4hHeWoibgkBVZNOAD1EA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

使用 `FlagEmbedding`提供的评估方法进行比较，`BAAI-bge`的效果是最好的。 这个需要根据你LLM对应的选embeeding 模型


<br />


<br />


### 四、增加元数据（Metadata Addition）

使用标题、关键词、问题假设等元数据增强chunk的信息来改善检索。


<br />


### 五、选择优秀的向量数据库（Vector Databases）

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKRdOyLiaOUHfKlWWSEArvMqp9JxJcHDrHWuKuzqyw1SDOoxbFgKEs6aA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对比了5种开源数据库，其中只有Milvus全部支持以下4个指标：

* `multiple index types`(多种索引):可以根据不同的数据特征和用例优化搜索
* `billion-scale vector support`：支持十亿级数据集的向量检索
* `hybrid search`（混合检索）：将向量搜索与传统关键字搜索相结合，提高检索准确率
* `cloud-native`：支持云原生服务，可确保云环境中的无缝集成、可扩展性和管理



<br />


### 六、优秀检索模块的配方（Retrieval Methods）

检索模块就是，根据给定的用户query(查询)和文档之间的相似度，从预先构建的语料库中选择与查询最相关的前k个文档。然后，LLM模型使用这些文档生成对query相对于的answer.

但是原始的查询通常由于表达不佳和缺乏语义信息而表现不佳，这会对检索过程产生负面影响。可以从转化查询的方法来解决这样的问题。

* 查询重写（Query Rewriting）：查询重写可以优化查询，以便更好地匹配相关文档。受到Rewrite-Retrieve-Read框架的启发，可以利用LLM重写查询以提高性能。
* 查询分解（Query Decomposition）：将复杂的问题，派生出几个简单的子问题，来帮助检索。
* 伪文档生成（Pseudo-documents Generation）：利用HyDE方法，基于用户查询生成一个假想文档，并使用假想答案的嵌入来检索相似的文档。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKKbpVGxUMibp2W1npTMAbDaFAYTQPCA9hNcYSkDGdtfoVoKfW2IAhSOA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

显然，HyDE+Hybrid Search 效果最好

在使用HyDE时，将伪文档与query拼接在一起检索，效果更好

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKBWhuj1Iiam36oQfY32J8RSXys79lpuK2wqib6bYV1lFvPoiaY6JPtk5Ug/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在使用Hybrid search混合检索时，选择0.3或者0.5效果最好。

在这里 控制稀疏检索和密检索得分。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKdr7Y9AW3mE6z1dJrBryr1MuZVymvbIIrktFvJvwpCWnFFrOVrMzSAA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


<br />


### 七、重新排序（Reranking Methods）

检索之后，会采用重新排序阶段来增强检索到的文档的相关性，确保最相关的信息出现在列表的顶部。这个阶段使用更精确但耗时的方法来有效地重新排序文档，增加查询和排名最高文档之间的相似性。

考虑两种排序方案：

* DLM重新排序：这种方法利用语言模型（DLMs，例如BERT T5）进行重新排序。将文档与查询的相关性分类为“真”或“假”。在微调过程中，模型通过连接查询和文档输入进行训练，并按相关性进行标记。在推理时，文档根据“真”标记的概率进行排名。
* TILDE重新排序：通过预测模型词汇表中的Token概率，独立地计算每个查询项的可能性。文档通过求和预先计算的查询Token的对数概率来评分,从而可以在推理时快速重新排序。TILDEv2 通过仅索引文档中存在的标记、使用 NCE 损失和扩展文档来改进这一点，从而提高效率并减少索引大小。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKv6AswcbLUG1u6mIMnasnj0yPpONwL0gAwRyX62ribwI2ziaibLD5KVkiaw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

从上述表现来看，RankLLama效果表现最好，但是鉴于参数规模，利用Bert排序更具有优势。


<br />


### 八、打包文档（Document Repacking）

LLM会受到提供的文档顺序影响答案的生成。因此，在文档重排之后，需要跟上一个文档打包策略。打包方法可以概括为以下三种：

* “正向”：根据重新排序阶段的相关性得分降序重新打包文档
* “反向”：根据相关性得分从低到高排列文档
* “两侧”：将相关信息放置在输入的头部或尾部以实现最佳性能

其中，“两侧”策略是当前最有效的方式。


<br />


### 九、摘要（Summarization）

检索到的结果可能包含冗余或不必要的信息，这可能会妨碍LLMs生成准确的结果。而且较长的prompt会减慢推理过程。因此，在检索增强（RAG）流程中，高效地摘要检索也是非常重要的。

摘要任务可以分为2种：

* 提取式：将文本分割成句子，然后根据重要性对它们进行评分和排序，选取重要的句子。例如：BM25, Contriever,Recomp (extractive)
* 抽象式：压缩多个文档中的信息，重新表述并生成连贯的摘要.例如：SelectiveContext，LongLLMlingua，Recomp (abstractive)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKTLVcE1YMQeddQR8slpBMibNPSDNZfeYnsZGujuraxicBCQtdNAlRzV5Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中，表现最好的还是抽象式的 `Recomp (abstractive)`方法


<br />


### 十、LLM微调（Generator Fine-tuning）

为了探讨相关或不相关上下文对LLM性能的影响，一共设定了以下五种策略：

* Dg：上下文由与查询相关的文档组成，表示为。
* Dr：上下文包含一个随机抽样的文档，表示为。
* Dgr：上下文包括一个相关文档和一个随机选择的文档，表示为。
* Dgg：上下文由两份与查询相关的文档组成，表示为

每种策略微调后的模型分别用相应Mg、Mr、Mgr、Mgg.

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKgdvsib3ak5IjibG8FPugicS8AMmlcRL9fNr3WOghR9iaYgHibzVCk3ToVHQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

通过对比发现，Mgr是最好的。

在训练期间混合相关和随机上下文可以增强生成器对不相关信息的鲁棒性，同时确保有效利用相关上下文。

因此，我们确定在训练期间通过增加一些相关和随机选择的文档作为最佳做法。



## 总结

![图片](https://mmbiz.qpic.cn/mmbiz_png/JnQiaTcHwIEma5XByDHYd3neVUPicqWTWKQjzVhrVIYc4wZKa53IhdRz0ZXMZ0FdqPmsZic4pFqvz0FwTGvZCqibnQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最后总体回顾一下，RAG各个流程步骤。

其中，上面红色方框的部分，为各自步骤的最佳策略，组合成一个完整的RAG流程，能达到最佳的效果。

* Query Classification Module：查询分类将RAG得分从0.428提升到了0.443。延迟从16.58降低到了11.71
* Retrieval Module：使用 `Hybrid with HyDE` ,将RAG得分提高到了0.58，且保持住了11.7的推理延迟
* Reranking Module：采用MonoT5保持住了RAG最高的得分0.58， 其它方法略有下降。
* Repacking Module：Sides策略最有效保持住了RAG最高得分0.58，同时对比 `reverse`和 `forward`可以发现，与查询越相关的上下文放在更靠前的位置，可以提高RAG的得分。
* Summarization Module ：尽管通过移除摘要模块可以实现较低延迟，但是由于Recomp能够解决LLM最大长度限制问题，所以Recomp还是首选。如果对时间延迟比较敏感，可以丢弃这个模块。
