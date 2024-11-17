# 自然语言处理:第六十二章 KAG 超越GraphRAG的图谱框架

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

项目地址: [OpenSPG/KAG: KAG is a knowledge-enhanced generation framework based on OpenSPG engine, which is used to build knowledge-enhanced rigorous decision-making and information retrieval knowledge services](https://github.com/OpenSPG/KAG)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

<br />

**KAG（ **Knowledge Augmented Generation** ）框架**早在9月份就已经发布，近期终于开源了，它的核心在于提出了：

* *一种LLM友好的知识表示方法*
* *知识图谱与原始文本块之间的相互索引*
* *逻辑形式引导的混合推理引擎*
* *以及基于语义推理的知识对齐*

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtMw8ia2KQyN1kGGuic4PWRd4kpmtyOR4xibeVj7ibSKTC6b1fnictyIhEkIw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

<br />

<br />

归功于在构建更有效的索引、知识对齐和混合解决库方面的创新，KAG框架在多跳问答任务中相比于现有的RAG方法有显著的性能提升，**2wiki、MuSiQue数据集上的EM指标直接翻倍。**此外，KAG框架在**蚂蚁集团的电子政务问答和电子健康**问答场景中也表现出了更高的准确性。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtDR4L3Aic3n6IgP7ZHRCnpiaywpM6pkWJHSffrvXlNzibuI0zzOE60e57A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

<br />

<br />

**KAG构建器流水线的示例**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtafmu8kjuhUrk4kcj2hoegIKeyWusS7wylp5ibghicV0PDeFafjKlA0TA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

---

## ****LLM友好的知识表示方法****

KAG框架提出了一个针对大型语言模型（LLMs）友好的知识表示框架，称为 **LLMFriSPG** 。这个框架的目的是为了让知识图谱（KG）更好地支持LLMs的应用，并提高两者之间的协同效果。

 **LLMFriSPG** ：一个对大型语言模型（LLMs）友好的知识表示框架。通过概念将实例和概念分开，以实现与LLMs更有效的对齐。在本研究中，除非另有说明，实体实例和事件实例统称为实例。SPG属性被划分为知识和信息领域，也称为静态和动态领域，它们分别与具有强模式约束的决策专长和具有开放信息表示的文档检索索引知识兼容。红色虚线代表从信息到知识的融合和挖掘过程。增强的文档块表示为LLMs提供了可追溯和可解释的文本上下文。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtnNfwWlKjoZHXic8QOnHu790KXjH77Mo8c2w0gelO9ULegibdcjicD4mkQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

1. **数据结构定义** ：LLMFriSPG定义了一个数据结构M，包括实体类型（T）、概念类型（C）、归纳关系（ρ）和可执行规则（L）。实体类型包括预定义的属性，这些属性与LPG语法声明兼容。概念类型包括概念类、概念及其关系，每个概念树的根节点是一个与LPG语法兼容的概念类型类。
2. **实例和概念的分离** ：LLMFriSPG将实例和概念分离，以实现与LLMs的有效对齐。实体实例和事件实例统称为实例。每个实例可以与一个或多个概念类型相关联，以表达其语义类型。
3. **属性和关系** ：对于每种类型，属性和关系包括领域专家预定义的部分、临时添加的内容以及系统内置的属性，如支持块（supporting_chunks）、描述（description）、摘要（summary）和归属（belongTo）。
4. **层次化的知识表示** ：LLMFriSPG支持从数据到信息再到知识的层次化表示。知识层（KGcs）遵循SPG语义规范，支持在严格的模式约束下构建知识体系和定义逻辑规则。信息层（KGfr）通过信息抽取得到实体和关系等图数据。原始块层（RC）则是经过语义分割处理后的原始文档片段。

<br />

<br />

**知识和信息的层次表示**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtg6GwotQFxPToc2wjOSuIl3LIovDWDmdzVj8Tiau3tiaZMa4GvvfUbp5A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

---

## ****相互索引机制****

KAG框架介绍了一种相互索引机制，旨在构建知识结构和文本块之间的索引，以增强知识表示和检索的效率：

1. **语义分块（Semantic Chunking）** ：基于文档的结构层次和段落间的逻辑联系，实现语义分块，生成符合长度限制且语义连贯的文本块。
2. **信息提取与描述性上下文** ：使用大型语言模型（LLMs）提取实体、事件、概念和关系，并构建KGfr与RC之间的互索引结构，实现跨文档链接。
3. **领域知识注入和约束** ：通过迭代提取方法，将领域概念和术语及其描述存储在KG存储中，并通过openIE提取文档中的所有实例，执行向量检索以获得与领域知识对齐的集合。
4. **预定义知识结构** ：对于具有标准化结构的专业文档，如药品说明书和政务文件，可以预定义实体类型和属性，以便于信息提取和知识管理。
5. **文本块向量与知识结构的互索引** ：KAG的互索引机制遵循LLMFriSPG的语义表示，包括共享模式、实例图、文本块和概念图等核心数据结构，以及KG存储和向量存储两种存储结构。

<br />

<br />

**领域非结构化文档的KAG构建器的流程。从左到右，首先，通过信息提取获得短语和三元组，然后通过语义对齐完成消歧和融合，最后，构建的知识图谱被写入存储。**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtswmOGZZ4liaXlJyf3hE2QGr9GsBhpzlGXhyk30c4O8cVcPU2rv0ribLg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

<br />

<br />

---

## **逻辑形式引导的混合推理引擎**

KAG框架介绍了一个基于逻辑形式的混合推理和求解引擎，它能够将自然语言问题转化为结合语言和符号的解题过程。

 **逻辑形式执行的示例。** 在这张图中，左侧显示了知识图谱（KG）构建过程，而右侧是整体的推理和迭代过程。首先， **基于用户的总体问题执行逻辑形式分解** ，然后使用 **逻辑形式引导的推理进行检索和推理** 。最后 **，生成器判断用户的问题是否得到满足** 。如果没有，就提供一个新的问题，进入新的逻辑形式分解和推理过程。如果确定问题得到满足，生成器直接输出答案。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtfgu6SDNRK2fiaJF5NB3jf3sQkWEwuqEgJajBrnZ3QYCtqicicoY01OZAg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

1. **逻辑形式规划** ：通过逻辑函数来定义执行动作，将复杂问题分解为可以推理的子问题。这些逻辑函数包括检索（Retrieval）、排序（Sort）、数学计算（Math）和推理（Deduce）等。
2. **逻辑形式推理** ：使用逻辑形式来表达问题，以便能够清晰地描述语义关系。这种方法可以处理涉及逻辑推理过程的问题，如“与”、“或”、“非”以及交集和差集等。
3. **逻辑形式检索** ：在传统的RAG中，检索是通过计算问题与文档片段嵌入之间的相似度来实现的。KAG框架提出了一种结合稀疏编码器和密集检索器的方法，以提高检索的准确性。

<br />

<br />

---

## ****知识对齐策略****

详细介绍了KAG框架中的知识对齐（Knowledge Alignment）策略，旨在解决基于信息抽取构建知识图谱（KG）时在知识对齐方面遇到的挑战：

1. **知识对齐的必要性** ：传统的基于向量相似度的信息检索方法在知识对齐上存在缺陷，如语义关系的错位、知识粒度不一致、与领域知识结构不匹配等问题。这些问题导致检索结果不精确，无法满足特定领域的专业性需求。
2. **概念图的利用** ：为了增强离线索引和在线检索的语义推理能力，KAG框架利用概念图来提升知识对齐。通过概念图，可以增强知识实例的标准化、概念与实例之间的链接、概念间关系的完整性，以及领域知识的注入。
3. **语义关系的分类** ：文中总结了六种常用于检索和推理的语义关系，包括同义词（synonym）、属于（isA）、是部分（isPartOf）、包含（contains）、属于（belongTo）和导致（causes）等。
4. **增强索引（Enhance Indexing）** ：通过使用大型语言模型（LLMs）预测索引项之间的语义关系或相关知识元素，包括知识实例的消歧和融合、实例与概念之间的关系预测、概念及其关系的完整性补充。
5. **增强检索（Enhance Retrieval）** ：在检索阶段，利用语义关系推理来搜索KG索引，结合语义关系推理和相似度检索，以提高检索的专业性和逻辑性，从而获得正确的答案。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtArYTcBpdLH0CUqliab85v8PwMspbff4ToLsz9ibZgDJkLb3U2V4oXZpQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

<br />

<br />

---

## **KAG框架核心模型**

详细介绍了KAG框架中的核心模型，这个模型旨在通过优化大型语言模型（LLMs）的三个关键能力——自然语言理解（NLU）、自然语言推理（NLI）和自然语言生成（NLG），来提升知识增强生成的性能：

1. **自然语言理解（NLU）** ：NLU包括任务如文本分类、命名实体识别、关系提取等。为了提升NLU能力，KAG通过大规模指令重构，创建了一个包含超过20,000个多样化指令的NLU指令数据集，用于监督式微调，从而增强模型在下游任务中的表现。
2. ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtgRnKs7eVD6hgjVUibDF1f59QlC632lYmfLrxZZX2HzEAoAmic3iaenJ0A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
3. **自然语言推理（NLI）** ：NLI任务用于推断给定短语之间的语义关系，包括实体链接、实体消歧、分类扩展等。KAG通过收集高质量的概念知识库和本体论，构建了一个包含8,000个概念及其语义关系的概念知识集，用于提升模型的语义推理能力。
   ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHtic6ELcZS5HzEKRGvbS8F20icFwnXdFJG7xDHmQs7lNIDqyxgh5icaiaaxg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
4. **自然语言生成（NLG）** ：为了使模型生成的文本更好地符合特定领域的逻辑和风格，KAG提出了两种有效的微调方法：**K-LoRA和AKGF**。K-LoRA通过预训练和基于LoRA的微调，使模型能够识别知识图谱中信息的格式，并习得领域特定的语言风格。AKGF则利用知识图谱作为自动评估器，提供对当前响应知识正确性的反馈，引导模型进一步优化。
5. **单次推理（Onepass Infere**nce）：为了减少系统复杂性、建设成本以及模块间错误传播导致的级联损失，KAG引入了一种高效的单次推理模型（OneGen），使任意LLM能够在单次前向传递中同时进行生成和检索。

**KAG所需的模型能力**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricF2OABM4cqP8OZkeII3GnHt5qyXWYiarPkw2yYYLsEs8jC8l2zHgicicxS6a2iberVp2GLN4YaZzTtmtg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
https://arxiv.org/pdf/2409.13731
KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation
Github: https://github.com/OpenSPG/KAG
```
