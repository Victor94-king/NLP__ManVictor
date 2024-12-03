# 自然语言处理:第七十八章  RAG框架总结主流框架推荐（转载）

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**



<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


设想你正致力于构建一个智能问答系统，该系统旨在从庞大的知识库中迅速而精确地提取关键信息，并据此生成自然流畅的回答。然而，随着数据规模的不断扩大，系统面临着严峻的挑战：检索效率逐渐下滑，生成内容的质量亦趋于下降。这正是当前众多检索增强型生成（RAG）系统亟需解决的核心问题——如何在数据冗余、检索效率低下以及生成内容不相关之间找到一个最佳的平衡点。

RAG 的发展瓶颈:
传统 RAG 系统通过检索模型提取最相关的文档，再交给生成模型处理。但这种流水线式的设计存在两个主要问题：

1. 检索不够精确：简单的相似性检索模型容易漏掉重要信息或引入噪声数据。
2. 生成效率低下：无关或低质量的上下文增加了生成负担，降低了回答的质量和速度。
3. GraphRAG 框架 介绍

---

GraphRAG 框架在微软公司内部广受赞誉，并以此为契机，衍生出了一系列轻量级的优化版本，诸如 LightRAG 与 nano-GraphRAG 等。与此同时，还涌现出了一些别具一格的变体，如 KAG 框架。这些框架的核心改进之处在于，它们在传统 RAG 框架的基础上，进一步强化了实体、社区以及文本切块（Chunking）之间的内在联系，并且巧妙地将现有知识图谱（KG）中的知识融入其中。这一系列的改进措施，显著提升了信息检索的召回率与准确性，为用户带来了更为优质的信息检索体验。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0JMylh1lGX2S5xmCdt7oc5oWp6BUoPFZXMMkHYeBrlPBjA8Lx25yGdw/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 1.0 GraphRAG 微软

* github：https://github.com/microsoft/graphrag
* 论文：From Local to Global: A Graph RAG Approach to Query-Focused Summarization https://arxiv.org/pdf/2404.16130
* 项目文档：microsoft.github.io/graphrag/

最近微软团队开源了一款数据工作流与转换工具是一种结合了检索增强生成（RAG）技术和知识图谱的先进框架。它旨在通过利用外部结构化知识图谱来增强大型语言模型（LLMs）的性能，有效解决模型可能出现的 “幻觉” 问题、领域知识缺失以及信息过时等问题。GraphRAG 的核心目的在于从数据库中检索最相关的知识，以增强下游任务的答案质量，提供更准确和丰富的生成结果。

* **GraphRAG 工作原理**
* `索引建立阶段`：在 GraphRAG 的索引建立阶段，主要目标是从提供的文档集合中提取出知识图谱，并构建索引以支持后续的快速检索。这一阶段是 GraphRAG 工作流程的基础，其效率和准确性直接影响到后续检索和生成的质量。

  1. `文本块拆分`：首先，原始文档被拆分成多个文本块，这些文本块是 GraphRAG 处理的基本单元。根据微软的研究，每个文本块的大小和重叠度可以调整，以平衡处理速度和输出质量。
  2. `实体与关系提取`：利用大型语言模型（LLM），对每个文本块进行分析，提取出实体和关系。这一步骤是构建知识图谱的关键，涉及到命名实体识别（NER）和关系抽取（RE）技术。
  3. `生成实体与关系摘要`：为提取的实体与关系生成简单的描述性信息，这些信息将作为图节点的属性存储，有助于后续的检索和生成过程。
  4. `社区检测`：通过社区检测算法，如 Leiden 算法，识别图中的多个社区。这些社区代表了围绕特定主题的一组紧密相关的实体和关系。
  5. `生成社区摘要`：利用 LLM 为每个社区生成摘要信息，这些摘要提供了对数据集全局主题结构和语义的高层次理解，是回答高层次查询问题的关键。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0bAKibWFkiawjhrIEMvnSDbBBEprHXMXcElpBibSoVTxozm3ia73AJ5xWPA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

* 查询处理阶段

查询处理阶段是 GraphRAG 工作流程的最终环节，它决定了如何利用已建立的索引来回答用户的查询。

* 本地搜索（Local Search）：针对特定实体的查询，GraphRAG 通过扩展到相关实体的邻居和相关概念来推理，结合结构化数据和非结构化数据，构建用于增强生成的上下文。
* 全局搜索（Global Search）：对于需要跨整个数据集整合信息的复杂查询，GraphRAG 采用 Map-Reduce 架构。首先，利用社区摘要独立并行地回答查询，然后将所有相关的部分答案汇总生成全局性的答案。

> 在查询处理阶段，GraphRAG 展示了其在处理复杂查询任务上的优势，尤其是在需要全局理解和高层语义分析的场景中。通过结合知识图谱的结构化信息和原始文档的非结构化数据，GraphRAG 能够提供更准确、更全面的答案。

* 针对新闻文章数据集的示例问题，Graph RAG（C2）和基础 RAG 的表现
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0A9LVlSKsnYHZFgAXSwFMUGuU8apQH9ZbJaujnJIpcdlYB0nFBlRr9g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 1.1 **LightRAG** 香港大学

* `论文`：LightRAG: Simple and Fast Retrieval-Augmented Generation https://arxiv.org/abs/2410.05779v1
* `Github 地址`：https://github.com/HKUDS/LightRAG

LightRAG 在信息之间保持关系，能产生更优质的答案，同时其计算效率也更高。与之前的 RAG 模型相比，LightRAG 引入了多项创新功能：

* `图增强文本索引`：通过将图结构纳入文本索引，LightRAG 能够建立相关实体之间的复杂关系，从而提升系统的上下文理解能力。
* `双层检索系统`：LightRAG 采用双层检索机制，能够同时处理低层（具体细节）和高层（抽象概念）的查询。例如，它不仅可以回答 “谁写了《傲慢与偏见》？” 这样具体的问题，也能应对 “人工智能如何影响现代教育？” 这样抽象的问题。
* `增量更新算法`：该模型使用增量更新算法，以便在不重建整个数据索引的情况下，快速整合最新信息。这种方法能够选择性地索引新或修改过的内容，尤其适用于动态环境，比如新闻或实时分析，数据变化频繁的场景。

> LightRAG 的轻量化特性使其能够快速处理大规模知识库并生成文本，减少了计算成本，适合更多开发者和小型企业使用。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0jva9lViaXiatYhZe7IfoiaZnZOIkhKwTdJmWwH8CBicKc7Z9UFt9omuUicQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

LightRAG 的架构主要分为两个部分：基于图的文本索引和双层检索。其工作流程可以总结如下：

1. `图形文本索引`：将原始文本文件分割成小块，便于高效检索。
2. `知识图谱构建`：利用大语言模型（LLM）进行实体和关系的提取，并生成文本的键值对（K, V）。
3. `信息检索`：通过生成的键值对进行检索，包括：

   * `详细层面`：关注于文档的具体小部分，允许精确的信息检索。
   * `抽象层面`：关注整体意义，帮助理解不同部分之间的广泛连接。

通过这两种检索方式，LightRAG 能够在小文档部分中找到相关信息，并理解不同文档之间的更大、相互关联的概念。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0vq5ibNtsJgGEGbgeAKVSmSOPn32PGyOMia2RyF2ZygLR8vlPyGBCsiaOQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0iabZFia23vQN3c5cenzJBIQLewcZMMkHdt50Spt23AJaarhCNOoz4y2w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
关键词提取，LightRAG 中有 Prompts 示例：

```
示例 1:
查询："国际贸易如何影响全球经济稳定？"
################
输出：
Output:
{{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币汇率", "进口", "出口"]
}}

```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0dXprGlPyNxSp33skRJCY4XopicoHicicLLuZ3EdLgMTdMVsVheWxs9ppQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

 **LightRAG 的评估成效显著，其在检索精确度、模型可调性、响应速度以及接纳新信息的能力等多个维度上，均展现出了超越其他同类 RAG 模型，如 NaiveRAG、RQ-RAG、HyDE 以及微软研发的 GraphRAG 的优势** 。通过具体的案例研究分析，我们发现，尽管 GraphRAG 凭借其基于图的知识应用，在文档检索与文本生成方面表现不俗，但其运行所需资源更为庞大，进而导致了更高的成本投入。

* **案例研究：LightRAG 与基线方法 GraphRAG 的比较**
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0S7JDPgtP1dpibpfnet8tzwtAsGOe5gvALgPzm7rNvRgLpVVgrL94vFQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> 第三方基于 Streamlit 实现了一版开源的 LightRAG GUI，代码地址：https://github.com/aiproductguy/lightrag-gui/blob/demo2/notebooks/streamlit-app-lightrag.py

### 1.2 **GraphRAG-Ollama-UI**

使用本地 LLM 的 GraphRAG - 具有强大的 API 和用于索引 / 快速调整 / 查询 / 聊天 / 可视化 / 等的多个应用程序。这旨在成为终极的 GraphRAG/KG 本地 LLM 应用程序。

* Github 地址：https://github.com/severian42/GraphRAG-Ollama-UI

### 1.3 **nano-GraphRAG**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic09uiaAtwgIF5oRDx5vwWibPDIiamf5vw5HxyqfgT3LhXPoLGHGBCFNgDyw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

* Github 地址：https://github.com/gusye1234/nano-graphrag

nano-GraphRAG 是一款简洁且易于定制的 GraphRAG 实现。微软开源的 GraphRAG 在功能上确实非常强大，但是官方版本中对于阅读与定制修改都非常的不友好。nano-GraphRAG 项目的目的就是为您呈现了一个更为精简、高效、清晰的 GraphRAG 版本，同时保留了其核心特性。如何不考虑测试的代码，那么 nano-GraphRAG 的代码量大约只有 800 行。并且它短小精悍，易于扩展，支持异步操作，且完全采用类型注解。

对于高级用户，nano-graphrag 提供了一系列自定义选项。您可以替换默认组件，例如 LLM 函数、嵌入函数和存储组件。这种灵活性使您能够根据特定需求定制系统，并将其无缝集成到您的项目中。nano-graphrag 非常适合对 GraphRAG 感兴趣的人学习和使用，它的简洁性、易用性和可定制性使其成为开发人员选择 GraphRAG 的平替选择。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0PGHrR0xQjXVbNbDSwBHaXgnmzSLYsibE6Q4N21lpIbTWicQfb5PppAEQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

参考文章：[graphRAG 后的 triplex、itext2KG 与 nano-graphRAG 对比介绍](https://mp.weixin.qq.com/s?__biz=MzU4MjYyNzMzMw==&mid=2247484624&idx=1&sn=d7e6b0a7be6c6efef9ec2af57ee4c308&scene=21#wechat_redirect)

### 1.4 **KAG** 蚂蚁

* KAG 论文地址：https://arxiv.org/pdf/2409.13731KAG
* 项目地址：https://github.com/OpenSPG/KAG

KAG 是基于 OpenSPG 引擎和大型语言模型的逻辑推理问答框架，用于构建垂直领域知识库的逻辑推理问答解决方案。KAG 可以有效克服传统 RAG 向量相似度计算的歧义性和 OpenIE 引入的 GraphRAG 的噪声问题。KAG 支持逻辑推理、多跳事实问答等，并且明显优于目前的 SOTA 方法。

KAG 的目标是在专业领域构建知识增强的 LLM 服务框架，支持逻辑推理、事实问答等。KAG 充分融合了 KG 的逻辑性和事实性特点，其核心功能包括：

* 知识与 Chunk 互索引结构，以整合更丰富的上下文文本信息
* 利用概念语义推理进行知识对齐，缓解 OpenIE 引入的噪音问题
* 支持 Schema-Constraint 知识构建，支持领域专家知识的表示与构建
* 逻辑符号引导的混合推理与检索，实现逻辑推理和多跳推理问答

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0kmKibsOB8vtZP0cXWJ3pEgMHOVicPOoPEaXWMK0NQTLG7JeYopWJuY3g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0YLe30XaJliazaM4CRnv4ZBtPd7TO39YOCbbmLG2hFJkaNf6m83obKQA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
KAG 框架包括 kg-builder、kg-solver、kag-model 三部分：

* **kg-builder** 实现了一种对大型语言模型（LLM）友好的知识表示，在 DIKW（数据、信息、知识和智慧）的层次结构基础上，升级 SPG 知识表示能力，在同一知识类型（如实体类型、事件类型）上兼容无 schema 约束的信息提取和有 schema 约束的专业知识构建，并支持图结构与原始文本块之间的互索引表示，为推理问答阶段的高效检索提供支持。
* **kg-solver** 采用逻辑形式引导的混合求解和推理引擎，该引擎包括三种类型的运算符：规划、推理和检索，将自然语言问题转化为结合语言和符号的问题求解过程。在这个过程中，每一步都可以利用不同的运算符，如精确匹配检索、文本检索、数值计算或语义推理，从而实现四种不同问题求解过程的集成：检索、知识图谱推理、语言推理和数值计算。

### 1.5 **Fast-GraphRAG**

* Github 地址：https://github.com/circlemind-ai/fast-graphrag

circlemind-ai 组织开发了一个名为 fast-graphrag 的开源项目。这个项目的目标是提供一个高效、可解释且精度高的快速图检索增强生成（Fast GraphRAG）框架。该框架专门为 Agent 驱动的检索工作流程设计，能够轻松融入检索管道中，提供先进的 RAG 功能，同时避免了构建和设计 Agent 工作流程的繁琐复杂性。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0rUuCiaGIlkPLCrQibSicXJ8lPTrRdCYRMkPRzyUibHlx62ylbYAtksAHBg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
核心特性:

* 知识的可解释性和可调试性：利用图形提供人类可浏览的知识视图，支持查询、可视化和更新。
* 高效、低成本、快速：针对大规模运行而设计，无需昂贵的资源投入。
* 数据动态性：自动生成和优化图形，以最佳方式适应特定领域和本体需求。
* 实时更新：支持数据变化的即时更新。
* 智能探索能力：采用基于 PageRank 的图形探索，提升准确性和可靠性。
* 异步和类型化设计：完全异步，并提供完整的类型支持，确保工作流程的稳健性和可预测性

> 知识图谱中的排名优化：PageRank 作为一种经典的图算法，最初用于网页排名。但在 FastGraphRAG 中，知识库被重新建模为图结构，文档和节点间的

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic06exQMeulRv6cBlj4DLsxibdvBr6J2374Sa9Iglf4CCzO6jPTggmqicnA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 1.6 **Tiny-GraphRAG**

Tiny-Graphrag 是一个简洁版本的 GraphRAG 实现，旨在提供一个最简单的 GraphRAG 系统，包含所有必要的功能。实现了添加文档的全部流程，以及本地查询和全局查询的功能。

* github 地址：https://github.com/limafang/tiny-graphrag

> 需要说明的是，Tiny-Graphrag 是一个简化版本的 GraphRAG 实现，并不适用于生产环境，如果你需要一个更完整的 GraphRAG 实现，我们建议你使用: GraphRAG nano-graphrag

2. 传统 RAG 框架介绍（16 个）

---

传统的 RAG（Retrieval-Augmented Generation）框架，是一个集成了多重核心组件的综合性系统，这些组件涵盖了文本切块（Chunking）、向量转换（即向量化处理）、数据存储、信息检索、二次排序、内容生成以及内容评估等关键环节。该框架的精髓之处在于其高度的灵活性与适应性，能够轻松应对多种策略，包括但不限于文档处理技巧和检索策略等。

在这一框架下，涌现出了诸多具有代表性的实现方式。其中，RAGFlow 以其对深度文档理解的专注而著称；QAnything 则通过引入重排序（Rerank）机制，进一步提升了检索效果；而高度可配置的 Dify，则为用户提供了更为灵活多样的选择。尽管这些实现在具体细节上各有千秋，但它们所遵循的基本原理却是相通的。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0WEFRic6HfLNpx24kXiaKJRPtlA95Q33mgXW3hkMdH8nsbC0miamzutc4g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> 框架比较多这里也不做赘述，如 Dify、MaxKB、FastGPT 等在之前文章进行过详细讲解

### 2.1. **AnythingLLM**

具备完整的 RAG（检索增强生成）和 AI 代理能力。

* Github 地址：https://github.com/Mintplex-Labs/anything-llm

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0bWP41SdALaicY19TbDeRnJjtpq5Or9LnD8LjasBA2PZADSRaRCQcuLw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.2 **MaxKB**

Max Knowledge Base，是一款基于大语言模型和 RAG 的开源知识库问答系统，广泛应用于智能客服、企业内部知识库、学术研究与教育等场景。

* Github 地址：https://github.com/1Panel-dev/MaxKB
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0pNanq8SMib9NZjpiaLoibLKthlvnXhJPDibmq5ibj9uehQkoKWkK58a0o4w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.3 RAGFlow

RAGFlow 是一款基于深度文档理解构建的开源 RAG（Retrieval-Augmented Generation）引擎。RAGFlow 可以为各种规模的企业及个人提供一套精简的 RAG 工作流程，结合大语言模型（LLM）针对用户各类不同的复杂格式数据提供可靠的问答以及有理有据的引用。

* Github 地址：https://github.com/infiniflow/ragflow
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0blscxicj5KK7LZFXOiatplIVyaxiagH2wgFicoibesMgucyR5u5vsrkDP4g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0bxtVg1ozPpURAHYTEOwKibMCwbMBeyHVtIQ3D4ABntXCmwJiarGqsIQQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.4 **Dify**

一个开源的大型语言模型应用开发平台。Dify 直观的界面结合了 AI 工作流、RAG 流程、代理能力、模型管理、可观测性功能等，让您能快速从原型阶段过渡到生产阶段。

* Github 地址：https://github.com/langgenius/dify
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0zloJcvMe9W1V9YJfDbLk0ZkjS539kMdHFs15SwaI00Vb7cP3djkyqw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.5 **FastGPT**

FastGPT 是一个基于 LLM 大语言模型的知识库问答系统，提供开箱即用的数据处理、模型调用等能力。同时可以通过 Flow 可视化进行工作流编排，从而实现复杂的问答场景！

* Github 地址：https://github.com/labring/FastGPT
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0Q77VUG2WY7MuHyZHWtsItAicj6iahBz4evTlkWPVGBpShX1ibb8hbzqnQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.6. **Langchain-Chatchat**

基于 Langchain 和 ChatGLM 等不同大模型的本地知识库问答。

* Github 地址：https://github.com/chatchat-space/Langchain-Chatchat

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0mOSUPYH9Lg5h4HTtG29mQa6ASmfTqKYFu0ibn6exRHlUkdCiaQgapmWw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.7 **QAnything**

QAnything (Question and Answer based on Anything) 是致力于支持任意格式文件或数据库的本地知识库问答系统，可断网安装使用。您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。目前已支持格式: PDF(pdf)，Word(docx)，PPT(pptx)，XLS(xlsx)，Markdown(md)，电子邮件 (eml)，TXT(txt)，图片 (jpg，jpeg，png)，CSV(csv)，网页链接 (html)，更多格式。

* Github 地址：https://github.com/netease-youdao/QAnything

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0ZQxSw429bZOaiadmluvq4NW3dkqkHbQwaq7GJSAnCI5y31pvB89Tsmw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic074vSnBUOiadz8d9RNsg21AdlIdPGlQZZyAjcsdm1n74iaOTRIjuxVibjA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.8 **Quivr**

使用 Langchain、GPT 3.5/4 turbo、Private、Anthropic、VertexAI、Ollama、LLMs、Groq 等与文档（PDF、CSV 等）和应用程序交互，本地和私有的替代 OpenAI GPTs 和 ChatGPT。Quivr，帮助你构建你的第二大脑，利用 GenerativeAI 的力量成为你的私人助理！

* Github 地址：https://github.com/QuivrHQ/quivr
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0VwlkBy4SHDFP0R3GbYgMmIWDAAPicsCXjicD2KwqJDCpLeLmKTA9Oxkg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.9 **RAG-GPT**

RAG-GPT 利用 LLM 和 RAG 技术，从用户自定义的知识库中学习，为广泛的查询提供上下文相关的答案，确保快速准确的信息检索。使用 Flask、LLM、RAG 快速上线智能客服系统，包括前端、后端、管理控制台。

* Github 地址：https://github.com/open-kf/rag-gpt

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0FibR0wzibWbvahgMiaDiawicRWXMVzOJDf510BuPAlb9O1LibJO6OY24zn0A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.10 **Verba**

由 Weaviate 驱动的检索增强生成（RAG）聊天机器人。Verba：Golden RAGtriever，这是一款开源应用程序，旨在为开箱即用的检索增强生成 (RAG) 提供端到端、精简且用户友好的界面。只需几个简单的步骤，即可轻松探索数据集并提取见解，无论是使用 Ollama 和 Huggingface 在本地进行，还是通过 Anthrophic、Cohere 和 OpenAI 等 LLM 提供商进行。

* 地址：https://github.com/weaviate/Verba
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0uJIicUuOiaRBX4uYGgbnhY335ykdx3riasChAy6lnzmRTz2WJ9wZpicKmw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.11 **FlashRAG**

FlashRAG 是一个用于复现和开发检索增强生成 (RAG) 研究的 Python 工具包。我们的工具包包括 36 个预处理的基准 RAG 数据集和 15 种最先进的 RAG 算法。

* Github 地址：https://github.com/RUC-NLPIR/FlashRAG
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0VtS1PAdOstB7aBLSfvFlKW03LE7kyN4zfEHpHy33hUUbAp6LPsu9CQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0Mia630LZZnxn9SEck8uhkmMokc5ic7mGEKaIRN8diakAicqoKLcZNaPR9g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.12 **kotaemon**

一个开源的干净且可定制的 RAG UI。该项目作为一个功能性的 RAG UI，适用于想要对其文档进行 QA 的最终用户和想要构建自己的 RAG 管道的开发人员

* Github 地址：https://github.com/Cinnamon/kotaemon，
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0ap9oRSQiarXkj99cxIdAP7pLOzaBg8abDYQCFxgvLZrwlAFwcmIFXGw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0JweBRK9sIqjpSE1tQMor6GNspXomOicticovtINyKJjDibtprkM01aH3w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0E05FkRyrNYWQsNxuo0jYMw4rvJophWYcBTZKwzNAmsI07ndXqQ2tMw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.13 **RAGapp**

在企业中使用 Agentic RAG 的最简单方式。

* Github 地址：https://github.com/ragapp/ragapp
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic09u2DWT6JSgvjQFeuvl6zoy7naMxPib8fgWNTaqSVOqcAa3ZBb6IunGA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.14. **TurboRAG**

通过预计算的 KV 缓存加速检索增强生成，适用于分块文本。

* Github 地址：https://github.com/MooreThreads/TurboRAG
* 论文：TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text

当前的检索增强生成 (RAG) 系统会连接和处理大量检索到的文档块以进行预填充，这需要大量计算，因此导致第一个标记时间 (TTFT) 的延迟显著增加。为了减少计算开销和 TTFT，我们引入了 TurboRAG，这是一种新颖的 RAG 系统，它重新设计了当前 RAG 系统的推理范式，首先预先计算并离线存储文档的键值 (KV) 缓存，然后直接检索保存的 KV 缓存进行预填充。因此，在推理过程中无需在线计算 KV 缓存。此外，还对掩码矩阵和位置嵌入机制提供了许多见解，并对预训练语言模型进行了微调，以保持 TurboRAG 的模型准确性。我们的方法适用于大多数现有的大型语言模型及其应用，无需修改模型和推理系统。一系列 RAG 基准测试的实验结果表明，与传统 RAG 系统相比，TurboRAG 将 TTFT 降低了 9.4 倍（平均 8.6 倍），但保留了与标准 RAG 系统相当的性能。
![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0ZJ1LT9AcGoaQN6kmvKW2X8udkOH7fNNIrlGhfbM8PmBUaeu8E1TIKw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
下表总结了使用 10 个文档进行测试时的平均 TTFT：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0GdeIibZ1G4jQbQEX7SB9H8FY9U3sYGzCC7vwuY0UdcE9NfloibvVRCicA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

结果表明，使用 KV 缓存可将平均 TTFT 从约 4.13 秒显著缩短至约 0.65 秒，证明了带缓存的 TurboRAG 与不带缓存的传统 RAG 方法相比更高效。此外，随着文档数量的增加，加速效果更加明显。

### 2.15 **TEN Framework**

实时多模态 AI 代理框架。TEN 代表变革性扩展网络 (Transformative Extensions Network)，代表下一代 AI 代理框架，这是世界上第一个真正的实时多模式 AI 代理框架。

* Github 地址：https://github.com/TEN-framework/ten_framework

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0WHehYnKBJkdRMsC5ap4bbViaD4tmsPB9zxciaPP3hmrkZEmicHCSybnHg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0lb4l5ib23ppsIE8yRHlBsxwblbzXFo1eyrhnnJXJgBfWCRvxulZA28w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0CuHFk72mfpc9IHYYibC4eJNLYcxhXltF7GplCVIHpMUTfvpNwMB7QNQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.16. **AutoRAG**

RAG AutoML 工具。使用 AutoML 风格自动化进行检索增强生成 (RAG) 评估和优化的开源框架

* Github 地址：https://github.com/Marker-Inc-Korea/AutoRAG
  ![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic03pwHPFOk236cW2T1ia3KfhnYsYTwogXtHV5K9naiblkFeIypNxyPWcYQ/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WZBibiatAX4xxalMJnEk5pTcJdQngIgXic0pyzYgbXWAXacibicdCkoA6iayYQHfibbtUiaibfn1Qia7F46FV4P5ab3iaa1XA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


原文地址：[GraphRAG、Naive RAG框架总结主流框架推荐(共23个)：LightRAG、nano-GraphRAG、Dify等](https://mp.weixin.qq.com/s/oDhzc2lhWYJtAVc7Hwb7-A)
