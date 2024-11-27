# 自然语言处理:第七十一章 RAG中表格应该如何处理？

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**


<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


 目前，检索增强生成（RAG）系统成为了将海量知识赋能于大模型的关键技术之一。然而,如何高效地处理半结构化和非结构化数据，尤其是文档中的表格数据，仍然是 RAG 系统面临的一大难题。

    本文作者针对这一痛点，提出了一种处理表格数据的新颖解决方案。作者首先系统性地梳理了RAG系统中表格处理的核心技术，包括表格解析、索引结构设计等，并评述了现有的一些开源解决方案。在此基础上，作者提出了自己的创新之处——利用Nougat工具准确高效地解析文档中的表格内容，使用语言模型对表格及其标题进行内容摘要，最后构建一种新型的document summary索引结构，并给出了完整的代码实现细节。

    这种方法的优点是既能有效解析表格，又能全面考虑表格摘要与表格之间的关系，且无须使用多模态 LLM ，能够节省解析成本。让我们拭目以待该方案在实践中的进一步应用和发展。


RAG 系统的实现是一项极具挑战性的任务，特别是需要解析和理解非结构化文档中的表格时。而对于经过扫描操作数字化的文档（scanned documents）或图像格式的文档（documents in image format）来说，实现这些操作就更加困难了。至少有三个方面的挑战：

* **经过扫描操作数字化的文档（scanned documents）或图像格式的文档（documents in image format）比较复杂**，如文档结构的多样性、文档中可能包含一些非文本元素（non-text elements）以及文档中可能同时存在手写和印刷内容，都会为表格信息的准确自动化提取带来挑战。不准确的文档解析会破坏表格结构，将不完整的表格信息转换为向量表征（embedding）不仅无法有效捕捉表格的语义信息，还很容易导致 RAG 的最终输出结果出现问题。
* **如何提取每个表格的标题，如何将它们与对应的那个具体表格关联起来。**
* **如何通过合理的索引结构设计，将表格中的关键语义信息高效组织和存储起来。**

本文首先介绍了如何在检索增强生成（Retrieval Augmented Generation, RAG）模型中管理和处理表格数据。然后回顾了一些现有的开源解决方案，最后在当前的技术基础上，设计和实现了一种新颖的表格数据管理方法



<br />


<br />


<br />


# 1 **RAG表格数据相关核心技术介绍**

## ****1.1 Table Parsing 表格数据的解析****

该模块的主要功能是从非结构化文档或图像中准确提取表格结构（table structure）。

附加需求： **最好能提取出相应的表格标题，方便开发人员将表格标题与表格关联起来。**

根据我目前的理解，有以下几种方法，如图 1 所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZziaia40YXZdRloibWsXbI2rE9diasslu6BvZNgDCorJsiarvAjE7XpYcwWmg/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 1：Table parser（表格解析器）。图片由原文作者提供。

**(a).** 利用多模态LLM（如GPT-4V ^[1]^ ）识别表格，并从每个 PDF 页面中提取信息。

* 输入：图像格式的 PDF 页面
* 输出：JSON或其他格式的表格数据。如果多模态 LLM 无法提取表格数据，则应对 PDF 图像进行总结并返回内容摘要。

**(b). **利用专业的表格检测模型（如Table Transformer ^[2]^ ）来识别表格结构。

* 输入：PDF 页面图像
* 输出：表格图像

**(c).** 使用开源框架，如 unstructured^[3]^ 或其他也采用目标检测模型（object detection models）的框架（这篇文章 ^[4]^ 详细介绍了 unstructured 的表格检测过程）。这些框架可以对整个文档进行全面解析，并从解析结果中提取与表格相关的内容。

* 输入：PDF或图像格式的文档
* 输出：纯文本或 HTML 格式的表格（从整个文档的解析结果中获得）

**(d).** 使用 Nougat ^[5]^ 、Donut^[6]^ 等端到端模型（end-to-end models），解析整个文档并提取与表格相关的内容。这种方法不需要 OCR 模型。

* 输入：PDF 或图像格式的文档
* 输出：LaTeX 或 JSON 格式的表格（从整个文档的解析结果中获得）

需要说明的是，**无论使用哪种方法提取表格信息，都应同时提取出表格标题。因为在大多数情况下，表格标题是文档作者或论文作者对表格的简要描述，可以在很大程度上概括整个表格的内容。**

在上述四种方法中，**方法（d）可以较为方便地检索表格标题。**这对开发人员来说大有裨益，因为他们可以将表格标题与表格关联起来。下面的实验将对此作进一步说明。



<br />


<br />


<br />


## ****1.2 Index Structure 如何索引表格数据****

大致有以下几类建立索引的方法：

**(e).** 只为图像格式的表格建立索引。

**(f).** 只为纯文本或JSON格式的表格建立索引。

**(g).** 只为LaTeX格式的表格建立索引。

**(h).** 只为表格的摘要建立索引。

**(i).** Small-to-big（译者注：既包含细粒度索引，比如对每一行或表格摘要建立索引，也包含粗粒度索引，比如索引整个表格的图像、纯文本或 LaTeX 类型数据，形成一种分层的、从小到大的索引结构。） 或使用表格摘要建立索引结构，如图2所示。

The content of the small chunk（译者注：细粒度索引层级对应的数据块）比如将表格的每一行或摘要信息作为一个独立的小数据块。

The content of the big chunk（译者注：粗粒度索引层级对应的数据块）可能是图像格式、纯文本格式或LaTeX格式的整个表格。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzTHQEr9pyhUibHbCYqtw4luNXVrnKia8ViaX6DmLiaMib1n0oorevvib2bBfQ/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 2：以 small-to-big 的方式建立索引（上）和使用表格摘要建立索引（中、下）。图片由原文作者提供。

如上所述，表格摘要通常是使用 LLM 处理生成的：

* 输入：图像格式、文本格式或 LaTeX 格式的表格
* 输出：表格摘要



<br />


<br />


<br />


## ****1.3 无需解析表格、建立索引或使用 RAG 技术的方法****

有些算法不需要进行表格数据的解析。

**(j).** 将相关图像（PDF文档页）和用户的 query 发送给 VQA 模型（如 DAN^[7]^ 等）（译者注：视觉问答（Visual Question Answering）模型的简称。是一种结合了计算机视觉和自然语言处理技术的模型，可用于回答关于图像内容的自然语言问题。）或多模态 LLM，并返回答案。

* 要被索引的内容： 图像格式的文档
* 发送给 VQA 模型或多模态 LLM 的内容：Query + 图像形式的相应文档页面

**(k).** 向 LLM 发送相关文本格式的 PDF 页面和用户的 query ，然后返回答案。

* 要被索引的内容： 文本格式文档
* 发送给 LLM 的内容：Query + 文本格式的相应文档页面

**(l).** 向多模态 LLM（如 GPT-4V 等）发送相关文档图像（PDF 文档页面）、文本块和用户的 Query，然后直接返回答案。

* 要被索引的内容： 图像格式的文档和文本格式的文档块（document chunks）
* 发送给多模态 LLM 的内容：Query + 相应图像格式的文档 + 相应文本块（text chunks）

此外，下面还有一些不需要建立索引的方法，如图 3 和图 4 所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzAmGGbfUgIoiayMqOOW8oy0U1MSOH1THpHETDzzNpRMmbxrvC4unF9qQ/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 3：Category (m)（译者注：下面第一段介绍的内容）。图片由原文作者提供。

**(m).** 首先，使用（a）至（d）中的任何一种方法，将文档中的所有表格解析为图像形式。然后，将所有表格图像和用户的 query 直接发送给多模态 LLM（如 GPT-4V 等），并返回答案。

* 要被索引的内容： 无
* 发送给多模态 LLM 的内容：Query + 所有已经转换为图像格式的表格

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzNZWiao5kiaiboOUJJVogbqrnvNribUW9ibA9v7QdIyW7Pdrg4via58XKycnw/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 4：Ctegory (n)（译者注：下面第一段介绍的内容）。图片由原文作者提供。

**(n).** 使用通过（m）方法提取到的图像格式的表格，然后使用 OCR 模型识别表格中的所有文本，然后直接将表格中的所有文本和用户的 Query 发送到 LLM，并直接返回答案。

* 需要索引的内容： 无
* 发送给 LLM 的内容：用户的 Query + 所有表格内容（以文本格式发送）

值得注意的是，在处理文档中的表格时，有些方法并没有使用 RAG（Retrieval-Augmented Generation）技术：

* **第一类方法没有使用 LLM，而是在特定数据集上进行训练，使得 AI 模型（如基于 Transformer 架构并受到 BERT 启发的其他语言模型）来更好地支持表格理解任务的处理，比如 TAP** **AS**  ^**[8]**^ **。**
* **第二类方法是使用 LLM，采用预训练、微调方法或提示词工程，使得 LLM 能够完成表格理解任务，如 GPT4Table**  ^**[9]**^ **。**



<br />


<br />


<br />


# 2 **现有的表格处理开源解决方案**

<br />



上一节总结并归类了 RAG 系统中表格数据处理的关键技术。在提出本文要实现的解决方案之前，我们先来探索一些开源解决方案。

LlamaIndex 提出了四种方法 ^[10]^ ，其中前三种均使用了多模态模型（multimodal models）。

* 检索相关的PDF 页面图像并将其发送给 GPT-4V 响应用户 Query 。
* 将每个 PDF 页面均转换为图像格式，让 GPT-4V 对每个页面进行图像推理（image reasoning）。为图像推理过程建立 Text Vector Store index（译者注：将从图像中推理出的文本信息转换成向量形式，并建立索引），然后根据 Image Reasoning Vector Store（译者注：应当就是前面的索引，对前文建立的 Text Vector Store index 进行查询。） 查询答案。
* 使用 Table Transformer 从检索到的图像中裁剪表格信息，然后将这些裁剪后的表格图像发送到 GPT-4V 获取 query responses （译者注：向模型发送 Query 并获取模型返回的答案）。
* 在裁剪后的表格图像上应用 OCR，并将数据发送给 GPT4/ GPT-3.5 来回答用户的 query 。

总结一下上述四种方法：

* 第一种方法类似于本文中介绍的(j)方法，不需要进行表格解析。但结果表明，即使答案就在图像中，也无法产生正确答案。
* 第二种方法涉及到表格的解析，对应于方法(a)。索引内容可能是表格内容或内容摘要，完全取决于 GPT-4V 返回的结果，可能对应于方法 (f) 或 (h)。这种方法的缺点是，GPT-4V 从文档图像中识别表格并提取其内容的能力不稳定，尤其是当文档图像中包含表格、文本和其他图像（在 PDF 文档中这种情况很常见）时。
* 第三种方法与方法（m）类似，不需要编制索引。
* 第四种方法与方法（n）相似，也不需要编制索引。其结果表明，产生错误答案的原因是无法从图像中有效提取表格信息。

**通过进行测试发现，第三种方法的整体效果最好。不过，根据我进行的测试，第三种方法在检测表格这方面就很吃力，更不用说正确提取并关联合并表格标题和表格内容了。**

Langchain 也对半结构化数据的 RAG(Semi-structured RAG)^[11]^ 技术提出了一些解决方案，核心技术包括：

* 使用 unstructured 进行表格解析，这属于（c）类方法。
* 索引方法是 document summary index （译者注：将文档摘要信息作为索引内容），属于（i）类方法。细粒度索引层级对应的数据块：表格摘要内容，粗粒度索引层级对应的数据块：原始表格内容（文本格式）。

如图 5 所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzFOU9Xr6OoCSmhibTJkiaLvKk4Lf8Mx1r6u0d9zpibAOqBmo6Mom6P14kg/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 5 ：Langchain 的 Semi-structured RAG 方案 。Source: Semi-structured RAG^[11]^

Semi-structured and Multi-modal RAG^[12]^ 提出了三种解决方案，其架构如图 6 所示。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzHencmdficOwe0vodxggzRPjZGsGwR9dCp2hrMEMv7v7GuZ97pV6UR7Q/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 6：Langchain 的 semi-structured and multi-modal RAG 方案。Source: Semi-structured and Multi-modal RAG ^[12]^ .

**Option 1** 类似于前文的方法(l)。这种方法涉及使用多模态嵌入（multimodal embeddings）（如CLIP ^[13]^ ），将图像和文本转换为嵌入向量，然后使用相似性搜索算法（similarity search）检索两者，并将未经处理的图像和文本数据传递给多模态 LLM ，让它们一起处理并生成问题答案。

**Option 2** 利用多模态 LLM （如 GPT-4V^[14]^ 、LLaVA^[15]^ 或 FUYU-8b ^[16]^ ），处理图像生成文本摘要（text summaries）。然后，将文本数据转换为嵌入向量，再使用这些向量来搜索或检索与用户提出的 Query 相匹配的文本内容，并将这些文本内容传递给 LLM 生成答案。

* 表格数据的解析使用 unstructured，属于（d）类方法。
* 索引方法是 document summary index （译者注：将文档摘要信息作为索引内容），属于（i）类方法，细粒度索引层级对应的数据块：表格摘要内容，粗粒度索引层级对应的数据块：文本格式的表格内容。

**Option 3** 使用多模态 LLM （如 GPT-4V^[14]^ 、LLaVA^[15]^ 或 FUYU-8b ^[16]^ ）从图像数据中生成文本摘要，然后将这些文本摘要嵌入向量化，利用这些嵌入向量，可以对图像摘要进行高效检索（retrieve），在检索到的每个图像摘要中，都保留有一个对应的原始图像的引用（reference to the raw image），这属于上文的 (i) 类方法，最后将未经处理的图像数据和文本块传递给多模态 LLM 以便生成答案。



<br />


<br />


<br />


<br />


# 3 **本文提出的解决方案**

前文对关键技术和现有解决方案进行了总结、分类和讨论。在此基础上，我们提出了以下解决方案，如图 7 所示。为简化起见，图中省略了一些 RAG 模块，如 Re-ranking 和 query rewriting 。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzOpeq0KzSB17TMIg3YF2pD7cSjdImS3u8xoNibhpNlIL85g1Zg489h1Q/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 7：本文提出的解决方案。图片由原文作者提供。

* **表格解析技术：使用 Nougat ( (d) 类方法)。**根据我的测试，这种工具的表格检测能力比 unstructured（ (c) 类技术）更有效。此外，Nougat 还能很好地提取表格标题，非常方便与表格进行关联。
* **用于索引和检索文档摘要的索引结构（ (i) 类方法****）：**细粒度索引层级对应的内容包括表格内容摘要，粗粒度索引层级对应的内容包括 LaTeX 格式的相应表格和文本格式的表格标题。我们使用 multi-vector retriever^[17]^ （译者注：一种用于检索文档摘要索引中内容的检索器，该检索器可以同时处理多个向量，以便有效地检索与 Query 相关的文档摘要。）来实现。
* **表格内容摘要的获取方法****：** 将表格和表格标题发送给 LLM 进行内容摘要。

这种方法的优点是既能有效解析表格，又能全面考虑表格摘要与表格之间的关系。省去了使用多模态 LLM 的需求，从而节省了成本。

## ****3.1 Nougat 的工作原理****

Nougat^[18]^ 基于 Donut^[19]^ 架构开发，这种方法使用的算法能够在没有任何与 OCR 相关的输入或模块的情况下，通过隐式的方式自动识别文本。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzYx1EKRl9iawjLuBQPOfFIf5xuQiaia2y3GzMibDTCibticwoVTBiaRiaeWgMwQ/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 8 ：遵循 Donut^[19]^ 的端到端架构（End-to-end architecture）。 Swin Transformer 编码器接收文档图像并将其转换为 latent embeddings （译者注：在一个潜在空间中编码了图像的信息），然后以自回归的方式将其转换为一系列 tokens 。来源：Nougat: Neural Optical Understanding for Academic Documents.^[18]^

Nougat 在解析公式方面的能力令人印象深刻 ^[20]^ ，但它解析表格的能力也非常出色。如图 9 所示，它可以关联表格标题，非常方便：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzLf8G56evxZccC3hUVgQqUibbgic67YXVzlILBP7EyyDwVfKC71ZFcPicg/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 9 ：Nougat 的运行结果，结果文件为 Mathpix Markdown 格式（可通过 vscode 插件打开），表格以 LaTeX 格式呈现。

在我对十几篇论文进行的一项测试中，我发现表格标题总是固定在表格的下一行。这种一致性表明这并非偶然。因此，我们比较感兴趣 Nougat 是如何实现这种功能的。

**鉴于这是一个缺乏中间结果的端到端模型，它的效果很可能严重依赖于其训练数据。**

根据代码分析，表格标题部分的存储位置和方式，似乎与训练数据中表格的组织格式是相符的（紧随 `\end{table}` 之后就是 `caption_parts` ）。

```
def format_element(
    element: Element, keep_refs: bool = False, latex_env: bool = False
) -> List[str]:
 """
    Formats a given Element into a list of formatted strings.

    Args:
        element (Element): The element to be formatted.
        keep_refs (bool, optional): Whether to keep references in the formatting. Default is False.
        latex_env (bool, optional): Whether to use LaTeX environment formatting. Default is False.

    Returns:
        List[str]: A list of formatted strings representing the formatted element.
    """
 ...
 ...
 if isinstance(element, Table):
        parts = [
 "[TABLE%s]\n\\begin{table}\n"
 % (str(uuid4())[:5] if element.id is None else ":" + str(element.id))
 ]
        parts.extend(format_children(element, keep_refs, latex_env))
        caption_parts = format_element(element.caption, keep_refs, latex_env)
        remove_trailing_whitespace(caption_parts)
        parts.append("\\end{table}\n")
 if len(caption_parts) > 0:
            parts.extend(caption_parts + ["\n"])
        parts.append("[ENDTABLE]\n\n")
 return parts
 ...
 ...
```

## ****3.2 Nougat 的优点和缺点****

***优点：***

* Nougat 可以将以前的解析工具难以解析的部分（如公式和表格）准确地解析为 LaTeX 源代码。
* Nougat 的解析结果是一种类似于 Markdown 的半结构化文档。
* 能够轻松获取表格标题，并方便地与表格进行关联。

***缺点：***

* Nougat 的解析速度较慢，这一点可能会在大规模应用时造成困难。
* 由于 Nougat 的训练数据集基本上都是科学论文，因此在具有类似结构的文档上这种技术表现出色。而在处理非拉丁文本文档时，其性能就会下降。
* Nougat 模型每一次只训练一篇科学论文的一页，缺乏对其他页面的了解。这可能会导致解析内容中存在一些前后不一致的现象。因此，如果识别效果不佳，可以考虑将 PDF 分成单独的几页，然后逐页进行解析。
* 双栏论文（two-column papers）中的表格解析不如单栏论文（single-column papers）的解析效果好。

## ****3.3 代码实现****

首先，安装相关的 Python 软件包：

```
pip install langchain
pip install chromadb
pip install nougat-ocr
```

安装完成后，需要检查 Python 软件包的版本：

```
langchain                                0.1.12
langchain-community                      0.0.28
langchain-core                           0.1.31
langchain-openai                         0.0.8
langchain-text-splitters                 0.0.1

chroma-hnswlib                           0.7.3
chromadb                                 0.4.24

nougat-ocr                               0.1.17
```

建立工作环境并导入软件包：

```
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"

import subprocess
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
```

将论文 《Attention Is All You Need》^[21]^ 下载到路径 `YOUR_PDF_PATH`，运行 nougat 来解析 PDF 文件，并从解析结果中获取 latex 格式的表格数据和文本格式的表格标题。第一次执行该程序会下载必要的模型文件到本地环境。

```
def june_run_nougat(file_path, output_dir):
 # Run Nougat and store results as Mathpix Markdown
    cmd = ["nougat", file_path, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping"]
    res = subprocess.run(cmd) 
 if res.returncode != 0:
 print("Error when running nougat.")
 return res.returncode
 else:
 print("Operation Completed!")
 return 0

def june_get_tables_from_mmd(mmd_path):
    f = open(mmd_path)
    lines = f.readlines()
    res = []
    tmp = []
    flag = ""
 for line in lines:
 if line == "\\begin{table}\n":
            flag = "BEGINTABLE"
 elif line == "\\end{table}\n":
            flag = "ENDTABLE"

 if flag == "BEGINTABLE":
            tmp.append(line)
 elif flag == "ENDTABLE":
            tmp.append(line)
            flag = "CAPTION"
 elif flag == "CAPTION":
            tmp.append(line)
            flag = "MARKDOWN"
 print('-' * 100)
 print(''.join(tmp))
            res.append(''.join(tmp))
            tmp = []

 return res

file_path = "YOUR_PDF_PATH"
output_dir = "YOUR_OUTPUT_DIR_PATH"

if june_run_nougat(file_path, output_dir) == 1:
 import sys
    sys.exit(1)

mmd_path = output_dir + '/' + os.path.splitext(file_path)[0].split('/')[-1] + ".mmd" 
tables = june_get_tables_from_mmd(mmd_path)
```

函数 june_get_tables_from_mmd 用于从一个 mmd 文件中提取所有内容（从 `\begin{table}` 到 `\end{table}`，还包括 `\end{table}` 后的第一行），如图10所示。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzM3CDOyaKvgKKuSCLROVF0hz9hhJROdibtNS6fAe3CD43xAEgDsQHXibg/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 10：Nougat 的运行结果，结果文件为 Mathpix Markdown 格式（可通过 vscode 插件打开），解析出来的表格内容为 latex 格式。函数 june_get_tables_from_mmd 的功能是提取红框中的表格信息。图片由原文作者提供。

**不过，目前还没有官方文件规定表格标题必须放在表格下方，或者表格应以 \begin{table} 开始，以 \end{table} 结束。因此，june_get_tables_from_mmd 是一种启发式的方法（heuristic）。**

以下是对 PDF 文档的表格解析结果：

```
Operation Completed!
----------------------------------------------------------------------------------------------------
\begin{table}
\begin{tabular}{l c c c} \hline \hline Layer Type & Complexity per Layer & Sequential Operations & Maximum Path Length \\ \hline Self-Attention & \(O(n^{2}\cdot d)\) & \(O(1)\) & \(O(1)\) \\ Recurrent & \(O(n\cdot d^{2})\) & \(O(n)\) & \(O(n)\) \\ Convolutional & \(O(k\cdot n\cdot d^{2})\) & \(O(1)\) & \(O(log_{k}(n))\) \\ Self-Attention (restricted) & \(O(r\cdot n\cdot d)\) & \(O(1)\) & \(O(n/r)\) \\ \hline \hline \end{tabular}
\end{table}
Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. \(n\) is the sequence length, \(d\) is the representation dimension, \(k\) is the kernel size of convolutions and \(r\) the size of the neighborhood in restricted self-attention.

----------------------------------------------------------------------------------------------------
\begin{table}
\begin{tabular}{l c c c c} \hline \hline \multirow{2}{*}{Model} & \multicolumn{2}{c}{BLEU} & \multicolumn{2}{c}{Training Cost (FLOPs)} \\ \cline{2-5}  & EN-DE & EN-FR & EN-DE & EN-FR \\ \hline ByteNet [18] & 23.75 & & & \\ Deep-Att + PosUnk [39] & & 39.2 & & \(1.0\cdot 10^{20}\) \\ GNMT + RL [38] & 24.6 & 39.92 & \(2.3\cdot 10^{19}\) & \(1.4\cdot 10^{20}\) \\ ConvS2S [9] & 25.16 & 40.46 & \(9.6\cdot 10^{18}\) & \(1.5\cdot 10^{20}\) \\ MoE [32] & 26.03 & 40.56 & \(2.0\cdot 10^{19}\) & \(1.2\cdot 10^{20}\) \\ \hline Deep-Att + PosUnk Ensemble [39] & & 40.4 & & \(8.0\cdot 10^{20}\) \\ GNMT + RL Ensemble [38] & 26.30 & 41.16 & \(1.8\cdot 10^{20}\) & \(1.1\cdot 10^{21}\) \\ ConvS2S Ensemble [9] & 26.36 & **41.29** & \(7.7\cdot 10^{19}\) & \(1.2\cdot 10^{21}\) \\ \hline Transformer (base model) & 27.3 & 38.1 & & \(\mathbf{3.3\cdot 10^{18}}\) \\ Transformer (big) & **28.4** & **41.8** & & \(2.3\cdot 10^{19}\) \\ \hline \hline \end{tabular}
\end{table}
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

----------------------------------------------------------------------------------------------------
\begin{table}
\begin{tabular}{c|c c c c c c c c|c c c c} \hline \hline  & \(N\) & \(d_{\text{model}}\) & \(d_{\text{ff}}\) & \(h\) & \(d_{k}\) & \(d_{v}\) & \(P_{drop}\) & \(\epsilon_{ls}\) & train steps & PPL & BLEU & params \\ \hline base & 6 & 512 & 2048 & 8 & 64 & 64 & 0.1 & 0.1 & 100K & 4.92 & 25.8 & 65 \\ \hline \multirow{4}{*}{(A)} & \multicolumn{1}{c}{} & & 1 & 512 & 512 & & & & 5.29 & 24.9 & \\  & & & & 4 & 128 & 128 & & & & 5.00 & 25.5 & \\  & & & & 16 & 32 & 32 & & & & 4.91 & 25.8 & \\  & & & & 32 & 16 & 16 & & & & 5.01 & 25.4 & \\ \hline (B) & \multicolumn{1}{c}{} & & \multicolumn{1}{c}{} & & 16 & & & & & 5.16 & 25.1 & 58 \\  & & & & & 32 & & & & & 5.01 & 25.4 & 60 \\ \hline \multirow{4}{*}{(C)} & 2 & \multicolumn{1}{c}{} & & & & & & & & 6.11 & 23.7 & 36 \\  & 4 & & & & & & & & 5.19 & 25.3 & 50 \\  & 8 & & & & & & & & 4.88 & 25.5 & 80 \\  & & 256 & & 32 & 32 & & & & 5.75 & 24.5 & 28 \\  & 1024 & & 128 & 128 & & & & 4.66 & 26.0 & 168 \\  & & 1024 & & & & & & 5.12 & 25.4 & 53 \\  & & 4096 & & & & & & 4.75 & 26.2 & 90 \\ \hline \multirow{4}{*}{(D)} & \multicolumn{1}{c}{} & & & & & 0.0 & & 5.77 & 24.6 & \\  & & & & & & 0.2 & & 4.95 & 25.5 & \\  & & & & & & & 0.0 & 4.67 & 25.3 & \\  & & & & & & & 0.2 & 5.47 & 25.7 & \\ \hline (E) & \multicolumn{1}{c}{} & \multicolumn{1}{c}{} & & \multicolumn{1}{c}{} & & & & & 4.92 & 25.7 & \\ \hline big & 6 & 1024 & 4096 & 16 & & 0.3 & 300K & **4.33** & **26.4** & 213 \\ \hline \hline \end{tabular}
\end{table}
Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

----------------------------------------------------------------------------------------------------
\begin{table}
\begin{tabular}{c|c|c} \hline
**Parser** & **Training** & **WSJ 23 F1** \\ \hline Vinyals \& Kaiser et al. (2014) [37] & WSJ only, discriminative & 88.3 \\ Petrov et al. (2006) [29] & WSJ only, discriminative & 90.4 \\ Zhu et al. (2013) [40] & WSJ only, discriminative & 90.4 \\ Dyer et al. (2016) [8] & WSJ only, discriminative & 91.7 \\ \hline Transformer (4 layers) & WSJ only, discriminative & 91.3 \\ \hline Zhu et al. (2013) [40] & semi-supervised & 91.3 \\ Huang \& Harper (2009) [14] & semi-supervised & 91.3 \\ McClosky et al. (2006) [26] & semi-supervised & 92.1 \\ Vinyals \& Kaiser el al. (2014) [37] & semi-supervised & 92.1 \\ \hline Transformer (4 layers) & semi-supervised & 92.7 \\ \hline Luong et al. (2015) [23] & multi-task & 93.0 \\ Dyer et al. (2016) [8] & generative & 93.3 \\ \hline \end{tabular}
\end{table}
Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)* [5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. _CoRR_, abs/1406.1078, 2014.
```

然后使用 LLM 对表格数据进行总结：

```
# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. The table is formatted in LaTeX, and its caption is in plain text format: {element}  """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
# Get table summaries
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
print(table_summaries)
```

以下是对《Attention Is All You Need》 ^[21]^ 中四个表格的内容摘要，如图11所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZzcicnBpKGia2K44UokwHUTVNMMAlsrBqlaFkXBpbO38ZJ93G3Q11Hmxag/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 11：《Attention Is All You Need》 ^[21]^ 中四个表格的内容摘要。

使用 Multi-Vector Retriever （译者注：一种用于检索文档摘要索引中内容的检索器，该检索器可以同时处理多个向量，以便有效地检索与 Query 相关的文档摘要。）构建 document summary index structure ^[17]^（译者注：一种索引结构，用于存储文档的摘要信息，并可根据需要检索或查询这些摘要信息）。

```
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name = "summaries", embedding_function = OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore = vectorstore,
    docstore = store,
    id_key = id_key,
    search_kwargs={"k": 1} # Solving Number of requested results 4 is greater than number of elements in index..., updating n_results = 1
)

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content = s, metadata = {id_key: table_ids[i]})
 for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))
```

一切准备就绪后，建立一个简单的 RAG pipeline 并执行用户的 queries ：

```
# Prompt template
template = """Answer the question based only on the following context, which can include text and tables, there is a table in LaTeX format and a table caption in plain text format:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")


# Simple RAG pipeline
chain = (
 {"context": retriever, "question": RunnablePassthrough()}
 | prompt
 | model
 | StrOutputParser()
)


print(chain.invoke("when layer type is Self-Attention, what is the Complexity per Layer?")) # Query about table 1

print(chain.invoke("Which parser performs worst for BLEU EN-DE")) # Query about table 2

print(chain.invoke("Which parser performs best for WSJ 23 F1")) # Query about table 4

```

运行结果如下，这几个问题都得到了准确的回答，如图 12 所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3FmD9EKJYf7zvdwqQYVP9wYy1f2zeuZz7X8GHvkiabypcoM9fOrPHQMqF6a20mjibIK9mokXgpmWTF4blhRIicYibQ/640?from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 12 ：对三个用户 queries 的回答结果。第一行对应于《Attention Is All You Need》中的表 1，第二行对应于表 2，第三行对应于表 4。

总体代码如下：

```
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"

import subprocess
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough


def june_run_nougat(file_path, output_dir):
 # Run Nougat and store results as Mathpix Markdown
    cmd = ["nougat", file_path, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping"]
    res = subprocess.run(cmd) 
 if res.returncode != 0:
 print("Error when running nougat.")
 return res.returncode
 else:
 print("Operation Completed!")
 return 0

def june_get_tables_from_mmd(mmd_path):
    f = open(mmd_path)
    lines = f.readlines()
    res = []
    tmp = []
    flag = ""
 for line in lines:
 if line == "\\begin{table}\n":
            flag = "BEGINTABLE"
 elif line == "\\end{table}\n":
            flag = "ENDTABLE"

 if flag == "BEGINTABLE":
            tmp.append(line)
 elif flag == "ENDTABLE":
            tmp.append(line)
            flag = "CAPTION"
 elif flag == "CAPTION":
            tmp.append(line)
            flag = "MARKDOWN"
 print('-' * 100)
 print(''.join(tmp))
            res.append(''.join(tmp))
            tmp = []

 return res

file_path = "YOUR_PDF_PATH"
output_dir = "YOUR_OUTPUT_DIR_PATH"

if june_run_nougat(file_path, output_dir) == 1:
 import sys
    sys.exit(1)

mmd_path = output_dir + '/' + os.path.splitext(file_path)[0].split('/')[-1] + ".mmd" 
tables = june_get_tables_from_mmd(mmd_path)


# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. The table is formatted in LaTeX, and its caption is in plain text format: {element}  """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
# Get table summaries
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
print(table_summaries)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name = "summaries", embedding_function = OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore = vectorstore,
    docstore = store,
    id_key = id_key,
    search_kwargs={"k": 1} # Solving Number of requested results 4 is greater than number of elements in index..., updating n_results = 1
)

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content = s, metadata = {id_key: table_ids[i]})
 for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))


# Prompt template
template = """Answer the question based only on the following context, which can include text and tables, there is a table in LaTeX format and a table caption in plain text format:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")

# Simple RAG pipeline
chain = (
 {"context": retriever, "question": RunnablePassthrough()}
 | prompt
 | model
 | StrOutputParser()
)

print(chain.invoke("when layer type is Self-Attention, what is the Complexity per Layer?")) # Query about table 1

print(chain.invoke("Which parser performs worst for BLEU EN-DE")) # Query about table 2

print(chain.invoke("Which parser performs best for WSJ 23 F1")) # Query about table 4
```

`<br/>`


<br />


<br />


<br />


<br />


<br />


# 4 **Conclusion**

本文讨论了在 RAG 系统中表格处理操作的关键技术和现有解决方案，并提出了一种解决方案及其实现方法。

在本文中，我们使用了 Nougat 来解析表格。不过，如果有更快、更有效的解析工具可用，我们会考虑替换掉 Nougat 。**我们对工具的态度是先有正确的 idea ，然后再找工具来实现它，而不是依赖于某个工具。**

在本文中，我们将所有表格内容输入到 LLM 。但是，在实际场景中，我们**需要考虑到表格大小超出 LLM 上下文长度的情况。我们可以通过使用有效的分块（chunking）方法来解决这个问题。**
