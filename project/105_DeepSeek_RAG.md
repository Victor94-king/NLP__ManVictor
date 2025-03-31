# 自然语言处理:第一百零五章 不要盲目再使用DeepSeek R1和QWQ这些推理模型做RAG了

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


DeepSeek R1 在首次发布时就展现出了强大的推理能力。在这篇文章中，我们将详细介绍使用 DeepSeek R1 构建针对法律文件的 RAG 系统的经验。

我们之所以选择法律文件，是因为法律专业人士经常面临一项艰巨的任务：浏览案例、法规和非正式法律评论库。即使是最善意的研究也会因检索正确的文档而陷入困境，更不用说准确地总结它们了。这是 RAG 的绝佳领域！

我们在大量法律文件数据集的基础上构建了 RAG，具体技术栈是：

* 使用 Qwen2 嵌入用于检索；
* ChromaDB 作为用于存储嵌入存储和查询的向量存储；
* DeepSeek R1 生成最终答案。

通过将专用检索器与强大的推理大模型连接起来，我们可以同时获得“三全其美”的效果：

1. 高相关性文档检索
2. 推理丰富的文本生成
3. 通过直接引用减少幻觉

我们开源了构建 RAG 的整个流程（地址在文末），并分享了一些来之不易的经验——哪些有效，哪些无效。

![RAG System Website](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkgx2KoGt3N3l1mYMTJicibCkLncoWs1Mh5K3wPLTuWvicxLtSXfhYquGXg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**来之不易的教训：该做和不该做的事情  **

1. 不要使用 DeepSeek R1 进行检索

尽管 DeepSeek R1 具有出色的推理能力，但它并不适合生成嵌入，至少目前还不行。

我们发现了一些例子，说明 DeepSeek R1 生成的嵌入与专门的嵌入模型 Alibaba-NLP/gte-Qwen2-7B-instruct 相比有多糟糕，Qwen 2 是 MTEB 排行榜上目前最好的嵌入模型。

我们使用这两个模型为数据集生成嵌入，并组成两个向量数据库。然后，我们对这两个模型使用相同的查询，并在相应模型生成的向量数据库中找到前 5 个最相似的嵌入。

关于解除租约的问答

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkC8QP8R9V3iaPY07zBYd39J2VRMjicxC9DzJ9qq0yrtmW6WQaZv5aH3FA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

简单翻译下

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkwBQljTV5AxrxhmMRlkib13Rggmfu70zpwOTKdgEXwBicopgqWRmiawSbA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

关于小额赔偿

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkicyANuypqdU09MI7Ad1ZWpJib8gK2ZO2ibYaaLdxhOFhlnmjRYcxViao9Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

简单翻译下![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkAfyYjk5VhlZQh3TpnGzc44PboTumlKsutwwNiaXcw6IZYlZoiaFWvMIw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在上表中，DeepSeek R1 的检索结果明显更差。为什么？

我们认为根本问题在于 DeepSeek-R1 的训练方式。DeepSeek-R1 主要被设计为推理引擎，专注于顺序思维和逻辑连接。这意味着 DeepSeek-R1 不会将文档映射到语义空间。

相比之下，Qwen2 模型变体 (gte-Qwen2-7B-instruct) 专门针对语义相似性任务进行训练，创建了一个高维空间，其中概念相似的文档无论具体措辞如何都紧密聚集在一起。

这种训练过程的差异意味着 Qwen 擅长捕捉查询背后的意图，而 DeepSeek-R1 有时会遵循导致主题相关但实际上不相关的结果的推理路径。

除非 DeepSeek-R1 针对嵌入进行了微调，否则不应将其用作 RAG 的检索嵌入模型。

2. 务必使用 R1 进行生成：推理令人印象深刻

虽然 R1 在嵌入方面遇到困难，但我们发现它的生成能力非常出色。通过利用 R1 的思维链方法，我们看到：

* 更强的连贯性：该模型综合了来自多个文档的见解，清晰地引用相关段落。
* 减少幻觉：R1 在内部“大声思考”，通过数据的视角验证每个结论。

让我们看几个例子：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkzI6iacrZicYeRfsicVruOUjRbhVdCp7LaZXnQicUtiae1Z59qNpn2uyvyMw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkZDwkv6jFv3kTvPib3xWcFUeNteiba1r35juOJbiamslJGzTnYvxp98HrA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

从这些例子中，我们观察到了 DeepSeek R1 卓越的推理能力。它的思考过程清楚地表明了如何从源法律文件中得出结论：

* R1 首先构建了一个连贯的法律问题模型，其详细的思考过程就是明证：首先，我记得读过关于提前终止罚款的内容……文件 1 提到……这种推理优先的方法允许模型在检索之前有条不紊地将多个来源的概念联系起来。
* 在处理租约终止或小额索赔法庭问题等复杂场景时，我们观察到 R1 明确地理解了每份文件（将所有这些放在一起……），没有幻觉。
* 最后，推理大模型用精确的引文来解释其推理，将结论与来源联系起来。这建立了从问题到推理再到答案的明确联系，确保了严谨性和可访问性。

我们用各种法律查询尝试了该模型，该模型始终表现出不仅能够从源文件中提取信息，而且还能从中学习和推理的能力。

要点：对于问答和总结，R1 是循序渐进的法律逻辑的金矿。将其保留在生成答案的阶段，肯定没问题。

3. 提示工程仍然很重要

高级推理并不能消除对精心设计的提示的需求。我们发现提示中的明确指导对于以下方面至关重要：

* 鼓励在生成的答案中引用文档。
* 使用“引用或说你不知道”的方法防止幻觉。
* 以用户友好的方式构建最终答案。

我们在整个实验过程中构建了以下提示：

```
You are a helpful AI assistant analyzing legal documents and related content. When responding, please follow these guidelines:
- In the search results provided, each document is formatted as [Document X begin]...[Document X end], where X represents the numerical index of each document.
- Cite your documents using [citation:X] format where X is the document number, placing citations immediately after the relevant information.
- Include citations throughout your response, not just at the end.
- If information comes from multiple documents, use multiple citations like [citation:1][citation:2].
- Not all search results may be relevant - evaluate and use only pertinent information.
- Structure longer responses into clear paragraphs or sections for readability.
- If you cannot find the answer in the provided documents, say so - do not make up information.
- Some documents may be informal discussions or reddit posts - adjust your interpretation accordingly.
- Put citation as much as possible in your response. 
First, explain your thinking process between <think> tags.
Then provide your final answer after the thinking process.
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkBw7eYF9sLAASJNyFf4e46zb8iaHeSXyOgU9UhSCd3XRiaULmXn2lY48Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

4. 文档分块

此外，我们发现有效的文档分块对于准确的文档检索非常重要。对文档进行分块有助于使每个嵌入更简洁地表示特定主题，并减少每个嵌入生成需要处理的标记数量。

我们使用句子感知拆分（通过 NLTK）对文档应用分块。我们还让每个块的开头和结尾包含与附近块重叠的内容。它有助于模型更好地解释部分引用而不会丢失全局。文档分块代码：

```
def chunk_document(document, chunk_size=2048, overlap=512):
    """Split document into overlapping chunks using sentence-aware splitting."""
    text = document['text']
    chunks = []
    # Split into sentences first
    sentences = nltk.sent_tokenize(text)
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': document['id'],
                'name': document['name'],
                'content': document['text'],
                'chunk_start': len(' '.join(current_chunk[:-(2 if overlap > 0 else 0)])) if overlap > 0 else 0,
                # Additional metadata fields...
            })
            # Keep last few sentences for overlap
            overlap_text = ' '.join(current_chunk[-2:])  # Keep last 2 sentences
            current_chunk = [overlap_text] if overlap > 0 else []
            current_length = len(overlap_text) if overlap > 0 else 0
        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space
```

要点：

* 使用 NLTK 进行句子感知分词（tokenization），而不是基于字符的分块
* 使用块间重叠的句子保留文档上下文

5. vLLM 高效快速

由于法律文件包含大量数据，因此生成 RAG 的嵌入可能需要很长时间。

最初，我们使用默认的 HuggingFace 库 sentence_transformer。我们首先使用典型的 Nvidia L4 GPU 运行，但遇到了“最常见”的错误：CUDA 内存不足。在 Nvidia A100 上尝试后，我们发现 sentence_transformer 需要 57GB DRAM 才能加载完整的 Alibaba-NLP/gte-Qwen2-7B-instruct 模型。

![sentence_transformer_oom](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDk8ticPS6Fj55uSnqSC68B3iaIib0VdV96bcOhJxJicOQdpKqcqAJ8wWtgdQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们改用了 vLLM，这是一种高吞吐量、内存高效的 LLM 推理和服务引擎。

使用 vLLM，我们可以使用标准 Nvidia L4 GPU 运行模型，vllm 大约需要 24G DRAM GPU。L4 也比 A100 便宜得多：在 GCP 上，Nvidia L4 每小时花费超过 0.7 美元，而 Nvidia A100 每小时至少花费 2.9 美元。

在配备 80GB DRAM 的 Nvidia A100 上比较 vllm 和句子转换器时，我们发现与句子转换器相比，使用 vLLM 为 Qwen2 模型生成嵌入的速度提高了 5.5 倍。

对于包含 15000 个块的 10000 份法律文件的语料库，处理时间为：

* 标准 sentence_transformer：~5.5 小时
* vLLM 实施：~1 小时

![vllm Comparison](https://mmbiz.qpic.cn/sz_mmbiz_png/SaeK9tW7Bu8ZhJsmmnAFrVaiaeFxNoTDkr2O0icHaAicmoopPO4yNIrgKV15pR5bC7ewrWm1N9ShUia4yngfiaVBLGw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

结论

为法律文件构建 DeepSeek R1 RAG 教会了我们一些重要的经验教训：

* 利用专门的嵌入模型（如 Qwen2）实现稳健的检索。
* 在生成阶段使用 R1 的推理能力来解决复杂的法律查询。
* 提示工程仍然是控制引用和构建内容的关键。
* 使用 vLLM 加速推理，大幅提高效率和速度。
