# 自然语言处理:第七十五章 利用LLM从非结构化PDF中提取结构化知识

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

原文连接: [利用LLM从非结构化PDF中提取结构化知识](https://mp.weixin.qq.com/s/lVcEqoe4gwP29lAf7QO4ig)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />




在当今数据驱动的世界中，组织机构们坐拥着无数的PDF文档，这些文档中蕴含着丰富的信息宝藏。然而，尽管人类可以轻易地阅读这些文件，但对于试图理解和利用其内容的机器来说，却构成了巨大的挑战。无论是研究论文、技术手册还是商业报告，PDF文件常常包含能够驱动智能系统、助力数据驱动决策的有价值知识。但如何将这些非结构化的PDF数据转化为机器能够高效处理的结构化知识，成为了现代信息处理系统面临的核心挑战之一。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/rkNbXNjSJokXYPxq4qtPF31SliaGfUdppugicicraJSUp8nJJWDUYHyPfK7bUoghjILfIEduKp5ic3RicglyTx7waNg/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 一、非结构化PDF数据的挑战

尽管PDF文档中的信息丰富多样，但面对非结构化数据时，三大核心挑战应运而生：

1. 缺乏可解释性：难以追踪系统是如何得出特定答案的。
2. 分析能力受限：非结构化数据限制了复杂分析的可能性。
3. 精度降低：在处理大量信息时，这一点尤为明显。

这些限制凸显了结构化格式（如表格和知识图谱）（[GraphRAG原理深入剖析-知识图谱构建](http://mp.weixin.qq.com/s?__biz=MzU4OTY4MDU4MQ==&mid=2247485128&idx=1&sn=b5ff763359e26ea25236f916824d3b30&chksm=fdc89c62cabf15743ae81b298bc18183a096be4a543e2206eb88de977a67182dee941ed0db8e&scene=21#wechat_redirect)）的强大之处。它们能够将原始信息转化为有组织、可查询的数据，从而使机器能够更有效地处理。为了将非结构化文档与结构化数据之间的鸿沟缩小，以便进行高级分析和AI应用，我们必须采取创新的手段。这不仅是现代信息处理系统的核心挑战，也是构建综合性知识图谱的重要目标。

### 二、PDF解析与结构化知识提取

将PDF内容转化为知识图谱的过程，不仅仅是简单的文本提取，而是需要理解上下文、识别关键概念以及识别思想之间的关系。这要求我们首先解析PDF内容。

#### 1. PDF解析工具选择

在众多可用的解析库中，本文选择了PyMuPDF及其扩展PyMuPDF4LLM。PyMuPDF4LLM的Markdown提取功能保留了诸如标题和列表等关键结构元素，这极大地提升了大型语言模型（LLMs）对文档结构的识别和解释能力，从而显著增强了检索增强生成（Retrieval-Augmented Generation, RAG）的结果。

在解析PDF时，我使用PyMuPDF4LLM生成的Markdown作为每页文档的文本内容，同时提取所有图像，并将它们的OCR输出附加到相同的Markdown输出中。这种方法能够自动处理不同类型的PDF：仅有图像的扫描PDF、包含文本的PDF以及同时包含图像和文本的PDF。

#### 2. OCR技术

对于图像中的文本提取，我们使用了PyTesseract，它是Google Tesseract-OCR引擎的Python封装。

```
import base64
from typing import Union, List, Dict, Any


import pymupdf
import pymupdf4llm
import numpy as np
from PIL import Image
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage




def ocr_images_pytesseract(images: List[Union[np.ndarray | Image.Image]]) -> str:
    import pytesseract


    all_text: str = ""
    for image in images:
        if isinstance(image, Image.Image):
            image = image.filter(ImageFilter.SMOOTH())
        all_text += '\n' +  pytesseract.image_to_string(image, lang="eng", config='--psm 3 --dpi 300 --oem 1')
    return all_text.strip('\n \t')


# Use PyMuPDF to open the document and PyMuPDF4LLM to get the markdown output
doc = pymupdf.Document(pdf_path)
doc_markdown = pymupdf4llm.to_markdown(doc, page_chunks=True, write_images=False, force_text=True)




def extract_page_info(page_num: int, doc_metadata: dict, toc: list = None) -> Dict[str, Any]:
    """Extracts text and metadata from a single page of a PDF document.


    Args:
        page_num (int): The page number
        doc_metadata (dict): The whole document metadata to store along with each page metadata
        toc (list | None): The list that represents the table-of-contents of the document

    Returns:
        A dictionary with the following keys:
            1) text: Text content extracted from the page
            2) page_metadata: Metadata specific to the page only like page number, chapter number etc.
            3) doc_metadata: Metadata common to the whole document like filename, author etc.

    """
    page_info = {}


    # Read the page of 
    page = doc[page_num]


    # doc_markdown stores the page-by-page markdown output of the document
    text_content: str = self.doc_markdown[page_num]['text']


    # Get a list of all the images on this page - automatically, perform OCR on all these images and store the output as page text
    # There are 3 options available: each with different preprocessing steps and all of them implemented in the self._ocr_func method
    images_list: list[list] = page.get_images()
    if len(images_list) > 0:
        imgs = []
        for img in images_list:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            # using PyTesseract requires storing the image as PIL.Image though it could've been bytes or np.ndarray as well
            cspace = pix.colorspace
            if cspace is None:
                mode: str = "L"
            elif cspace.n == 1:
                mode = "L" if pix.alpha == 0 else "LA"
            elif cspace.n == 3:
                mode = "RGB" if pix.alpha == 0 else "RGBA"
            else:
                mode = "CMYK"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode != "L":
                img = img.convert("L")
            imgs.append(img)

        text_content += '\n' + ocr_images_pytesseract(imgs)
    text_content = text_content.strip(' \n\t')
    page_info["text"] = text_content
    page_info["page_metadata"] = get_page_metadata(page_num=page_num+1, page_text=text_content, toc=toc)
    page_info["doc_metadata"] = doc_metadata
    return page_info
```

### 三、构建知识图谱的架构

在提取PDF中的结构化内容之前，我们首先需要定义输出结构。这通常通过现代LLMs提供的“结构化输出”功能来实现。本文使用了LangChain来处理与LLMs相关的所有流程，并使用了其with_structured_output方法。需要注意的是，此方法并非所有模型都支持，因此在使用前需确认模型兼容性。

#### 1. 节点与关系模型

知识图谱架构（[GraphRAG原理深入剖析-知识图谱构建](http://mp.weixin.qq.com/s?__biz=MzU4OTY4MDU4MQ==&mid=2247485128&idx=1&sn=b5ff763359e26ea25236f916824d3b30&chksm=fdc89c62cabf15743ae81b298bc18183a096be4a543e2206eb88de977a67182dee941ed0db8e&scene=21#wechat_redirect)）由两个主要组件构成：节点和关系。使用Pydantic库定义了节点和关系的模型。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/rkNbXNjSJokXYPxq4qtPF31SliaGfUdppDN3VzM4iacoJ5YicpWzLEMvXmUhkWTgjodk3QCwbtJxiboiaBDHicvAJ19w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

* 节点类：定义了单个实体，包括唯一标识符、实体类型、附加属性（可选）、别名（可选）以及文本定义（可选）。虽然别名和定义是可选的，但它们在后续处理步骤中至关重要，它们有助于：
  * 在后续处理步骤中进行节点消歧。
  * 指导LLMs保持一致性，防止重复创建节点。
  * 丰富知识图谱，提供有价值的上下文，提升RAG性能。
* 关系类：捕获节点之间的连接，包括源和目标节点ID、关系类型、属性（可选）以及提取上下文（可选）。与节点别名和定义类似，关系上下文保留了关于连接如何和在哪里被识别的有价值信息，提高了图谱对下游任务的实用性。

```
from pydantic import BaseModel, Field


class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")


class Node(BaseModel):
    id: str = Field(..., description="The identifying property of the node in Title Case")
    type: str = Field(..., description="The entity type / label of the node in PascalCase.")
    properties: Optional[List[Property]] = Field(default=[], description="Detailed properties of the node")
    aliases: List[str] = Field(default=[], description="Alternative names or identifiers for the entity in Title Case")
    definition: Optional[str] = Field(None, description="A concise definition or description of the entity")


class Relationship(BaseModel):
    start_node_id: str = Field(..., description="The id of the first node in the relationship")
    end_node_id: str = Field(..., description="The id of the second node in the relationship")
    type: str = Field(..., description="TThe specific, descriptive label of the relationship in SCREAMING_SNAKE_CASE")
    properties: List[Property] = Field(default=[], description="Detailed properties of the relationship")
    context: Optional[str] = Field(None, description="Additional contextual information about the relationship")

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: list[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: list[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
```

#### 2. 数据提取提示

接下来，我们定义了详细的提取提示，包括：

* 最终目标：构建具有丰富上下文信息的知识图谱。
* 节点ID、节点类型和关系类型的输出格式。
* 保持节点一致性并解决指代消解至其最完整形式。

利用这个提示和结构化架构，创建了一个数据提取链，该链将在后续步骤中用于从文本中提取知识图谱。

```
from langchain_core.prompts import ChatPromptTemplate


DATA_EXTRACTION_SYSTEM = """# Knowledge Graph Extraction for Rich Information Retrieval
## 1. Overview
You are an advanced algorithm designed to extract knowledge from various types of informational content. \
Your task is to build a knowledge graph that provides rich, contextual information for downstream tasks.


## 2. Content Focus
- Extract detailed information about concepts, entities, processes, and their relationships from the given text.
- Prioritize information that provides rich context and is likely to be useful for answering a wide range of questions.
- Include relevant attributes, properties, and descriptive information for each extracted entity.


## 3. Node Extraction
- **Node IDs**: Use clear, unambiguous identifiers in Title Case. Avoid integers, abbreviations, and acronyms.
- **Node Types**: Use PascalCase. Be as specific and descriptive as possible to aid in Wikidata matching.
- Include all relevant attributes of the entity in the node properties.
- Extract and include alternative names or aliases for entities when present in the text.


## 4. Relationship Extraction
- Use SCREAMING_SNAKE_CASE for relationship types.
- Create detailed, informative relationship types that clearly describe the nature of the connection.
- Include directional relationships where applicable (e.g., PRECEDED_BY, FOLLOWED_BY instead of just RELATED_TO).


## 5. Contextual Information
- For each node and relationship, strive to capture contextual information that might be useful for answering questions.
- Include temporal information when available (e.g., dates, time periods, sequence of events).
- Capture geographical or spatial information if relevant.


## 6. Handling Definitions and Descriptions
- For key concepts, include concise definitions or descriptions as node properties.
- Capture any notable characteristics, functions, or use cases of entities.


## 7. Coreference and Consistency
- Maintain consistent entity references throughout the graph.
- Resolve coreferences to their most complete form, including potential aliases or alternative names.


## 8. Granularity
- Strike a balance between detailed extraction and maintaining a coherent graph structure.
- Create separate nodes for distinct concepts, even if closely related.
"""




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DATA_EXTRACTION_SYSTEM),
        ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])

```

四、节点和关系提取

（一）基本提取过程
给定一系列文档，使用上述模式和提示从文本中提取节点和关系。通过在每个文档上调用文档提取链，可以得到输出节点和关系的列表。

```
nodes = []
rels = []


for doc in docs:
    output: KnowledgeGraph = data_ext_chain.invoke(
        {
            "input": doc.page_content
        }
    )
    nodes.extend(format_nodes(output.nodes))
    rels.extend(format_rels(output.rels))
```

（二）面临的挑战

1. **节点类型增殖**
   没有对创建不同节点类型进行限制，导致结果图谱 “分散”，降低了对下游任务的有效性。
2. **重复实体管理**
   没有防止文档之间重复节点和关系的措施，可能导致后续处理步骤中的可扩展性问题。
3. **来源可追溯性**
   没有跟踪每个节点的来源，降低了信息本身的可靠性，无法验证信息。

（三）解决措施

1. **针对节点类型增殖问题**
   更新提示，接受现有节点类型列表和一个 “主题” 字符串，作为提取实体类型的指导和限制。
2. **针对重复实体管理问题**
   用集合替换列表，并在代码中添加检查，确保只提取唯一的节点和关系。
3. **针对来源可追溯性问题**
   在文档的元数据中添加从每个文档中提取的节点列表，在创建知识图谱时，可以为每个文档创建节点，并建立文档节点与从该文档中提取的每个实体节点之间的关系，如 (:Document)-[:MENTIONS]->(:Entity)。

```
DATA_EXTRACTION_SYSTEM = """# Knowledge Graph Extraction for Rich Information Retrieval
## 1. Overview
You are an advanced algorithm designed to extract knowledge from various types of informational content. \
Your task is to build a knowledge graph that provides rich, contextual information for downstream tasks.
{subject}
## 2. Content Focus
- Extract detailed information about concepts, entities, processes, and their relationships from the given text.
- Prioritize information that provides rich context and is likely to be useful for answering a wide range of questions.
- Include relevant attributes, properties, and descriptive information for each extracted entity.


## 3. Node Extraction
- **Node IDs**: Use clear, unambiguous identifiers in Title Case. Avoid integers, abbreviations, and acronyms.
- **Node Types**: Use PascalCase. Be as specific and descriptive as possible to aid in Wikidata matching.
- Include all relevant attributes of the entity in the node properties.
- Extract and include alternative names or aliases for entities when present in the text.
- Following are some existing node types that were extracted from previous samples of the same document:\n{node_types}


## 4. Relationship Extraction
- Use SCREAMING_SNAKE_CASE for relationship types.
- Create detailed, informative relationship types that clearly describe the nature of the connection.
- Include directional relationships where applicable (e.g., PRECEDED_BY, FOLLOWED_BY instead of just RELATED_TO).


## 5. Contextual Information
- For each node and relationship, strive to capture contextual information that might be useful for answering questions.
- Include temporal information when available (e.g., dates, time periods, sequence of events).
- Capture geographical or spatial information if relevant.


## 6. Handling Definitions and Descriptions
- For key concepts, include concise definitions or descriptions as node properties.
- Capture any notable characteristics, functions, or use cases of entities.


## 7. Coreference and Consistency
- Maintain consistent entity references throughout the graph.
- Resolve coreferences to their most complete form, including potential aliases or alternative names.


## 8. Granularity
- Strike a balance between detailed extraction and maintaining a coherent graph structure.
- Create separate nodes for distinct concepts, even if closely related.
"""
from langchain_core.documents.base import Document




def merge_nodes(nodes: List[Node]) -> Node:
    '''
    Merges all the nodes with the same ID and type into a single node


    The only difference in these nodes is their list of properties, aliases and definition so 
    I extract and combine a list of unique properties and aliases from all these nodes
    and pick the longest definition to use as the definition of the merged node


    Args: 
        nodes (List[Node]):- List of nodes to merge into one

    Returns: A single Node object created by merging all the nodes
    '''
    props: List[Property] = []
    definition = ""
    aliases = set([])
    max_def_len = -1
    for node in nodes:
        props.extend([
            p for p in node.properties
            if not p in props
        ])
        if len(node.definition) >= max_def_len:
            # Pick the longest definition based on an inaccurate assumption: 
            # longest description = most descriptive definition
            definition = node.definition
            max_def_len = len(node.definition)
        aliases.update(set(node.aliases))
    return Node(id=nodes[0].id, type=nodes[0].type, properties=props, definition=definition, aliases=aliases)


def docs2nodes_and_rels(docs: List[Document], text_subject: str = '', use_existing_node_types: bool = False) -> Tuple[List[Document], List[Node], List[Relationship]]:
    '''
    Extract the list of nodes and relationships from the list of documents


    Args:
        docs (List[Document]): List of documents
        text_subject (str): The overall subject of the text that the given documents belong to. Default=''
        use_existing_node_types (bool): Whether to use the node types of the already existing nodes in the
                                        KG as a guide to the LLM. Default=False.


    Returns:
        A tuple containing the following lists:
            docs (List[Document]): A list of the slightly modified input docs
            nodes (List[Node]): The list of unique extracted nodes
            rels (List[Relationship]): The list of extracted relationships
    '''
    # Only store unique nodes - Helps later when adding to KG
    nodes_dict: Dict[str, list] = dict({})
    rels: list = []


    ex_node_types = set({})
    if use_existing_node_types:
        # get_existing_labels is a method that returns the set of labels of the already existing nodes in the KG
        ex_node_types = get_existing_labels()


    for i, doc in enumerate(docs):

        output: KnowledgeGraph = data_ext_chain.invoke(
            {
                "subject": text_subject,
                "input": doc.page_content, 
                "node_types": ','.join(list(ex_node_types))
            }
        )


        output_nodes: List[Node] = format_nodes(output.nodes)
        output_rels: List[Relationship] = format_rels(output.rels)


        ntypes = set([n.type for n in output_nodes])


        # Store nodes with same type and ID together so that they can be merged together later
        dnodes_dict: Dict[str, List[str]] = {}
        for n in output_nodes:
            nk = f"{n.id}::{n.type}"
            if nk in dnodes_dict:
                if not n in dnodes_dict[nk]:
                    dnodes_dict[nk].append(n)
            else:
                dnodes_dict[nk] = [n]


        for r in output_rels:
            if not r in rels:
                rels.append(r)


        # Store Node IDs+Node Type of nodes mentioned in this doc/chunk in doc metadata
        doc.metadata['mentions'] = list(dnodes_dict.keys())
        nodes_dict = {**nodes_dict, **dnodes_dict}
        # Update existing node types for next iteration
        ex_node_types: set[str] = ex_node_types.union(ntypes)


    # Merge all duplicate nodes (nodes with the same ID and type) into a single node
    # This involves combining their properties, aliases and definition
    nodes: List[Node] = [merge_nodes(nds) for _, nds in nodes_dict.items()]


    return docs, nodes, list(rels)
```

通过利用 LLM 的结构化输出功能进行可靠的信息提取，同时设计了包含节点和关系的完整模式，不仅包括基本的 ID 和类型，还包括一些附加属性，以减少歧义。在提取过程中，通过解决节点类型增殖、重复实体管理和来源可追溯性等关键问题，确保了数据质量。接下来，我们还需要探索如何将这些结构化知识集成到知识图谱中（[GraphRag-知识图谱结合LLM 的检索增强](http://mp.weixin.qq.com/s?__biz=MzU4OTY4MDU4MQ==&mid=2247485080&idx=1&sn=00a65a4a167b4da414dca946b099e550&chksm=fdc89c32cabf1524eab4f2892ddc04e0336b2d3e26c03e1e19515f9738f0756ed94a60dfa2de&scene=21#wechat_redirect)），并实现与各种智能应用的无缝对接。这涉及到更加复杂的技术和工具，如图数据库、知识图谱平台等。但无论如何，我们已经迈出了从非结构化PDF中提取结构化知识的关键一步，为未来的智能应用发展奠定了坚实的基础。
