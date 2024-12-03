# 自然语言处理:第七十章 Langchain的长文本总结处理方案

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

### 引言

随着大型语言模型（LLM）技术的飞速发展，模型支持的token数量不断增加，从最初的4k、8k扩展到现在的16k、32k，使得我们能够处理更长的文本内容。然而，这一进步也带来了新的挑战：

- **机器资源不足**：尽管模型能够处理更长的文本，但显卡资源可能成为限制因素，导致无法一次性处理全部样本。
- **理解能力不足**：即使资源充足，许多开源模型在理解和总结长文本方面仍存在不足，一次性输入过多token可能导致模型抓不住重点或产生幻觉。

最近在帮人做长文本的QA，一个思路做问题router之后进行判断问题是全局还是局部。如果是全局的话对长文本进行summary再回答。所以这里需要对长本文进行总结，在这里看到langchain有几条处理方式

<br />

<br />

<br />

<br />

### 处理方式

下面都会以代码+解析的形式出现，然后我这边用的Xinference启动的服务，可以参考下。

#### 1. Stuff

这是一种直接的处理长文本的方法，将所有内容一次性输入给LLM。

- **优点**：只需调用一次LLM，上下文完整，便于模型理解全貌。
- **缺点**：对GPU资源和模型理解能力要求高，且存在max_token限制，对于更长的文本仍存在限制。

  ![1732550209076](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550209076.png)

  代码：

  ```

  from langchain.document_loaders import Docx2txtLoader  # 新增这个导入
  from langchain_openai import ChatOpenAI
  from langchain.prompts import PromptTemplate
  from langchain.chains.llm import LLMChain
  from langchain.chains.combine_documents.stuff import StuffDocumentsChain
  import time

  start_time = time.time()  # 记录开始时间

  # 导入 docx 文档
  loader = Docx2txtLoader("xxx.docx")  # 修改这里的文件名
  # 将文本转成 Document 对象
  docs = loader.load()

  OPENAI_API_BASE = 'http://0.0.0.0:10000/v1'
  OPENAI_API_KEY = 'EMPTY'
  MODEL_PATH = "qwen2.5-instruct"

  llm = ChatOpenAI(model=MODEL_PATH, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY)

  # 定义提示
  prompt_template = """请用中文总结以下文档内容：\n\n{text}
  总结内容:"""
  prompt = PromptTemplate.from_template(prompt_template)
  CHAIN_TYPE = "refine"

  llm_chain = LLMChain(llm=llm, prompt=prompt)

  # 定义StuffDocumentsChain
  stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

  docs = loader.load()
  result = stuff_chain.invoke(docs)
  print(result["output_text"])

  end_time = time.time()  # 记录结束时间
  elapsed_time = end_time - start_time  # 计算耗时
  print(f"耗时: {elapsed_time:.2f} 秒")
  ```

#### 2. Map Reduce

类似于分布式处理，将长文本分成多段，多个mapper处理后再由reducer聚合结果。

- **优点**：理论上mapper可以无限多，支持横向扩展，支持并发处理。
- **缺点**：需要多次调用LLM，单块mapper容易信息丢失，且与Block的切分有关，切分不佳会影响总结效果。

![1732550279975](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550279975.png)

代码

```
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import time 

start_time = time.time()  # 记录开始时间
from langchain.document_loaders import Docx2txtLoader  # 新增这个导入

# 导入 docx 文档
loader = Docx2txtLoader("xxx.docx")  # 修改这里的文件名
# 将文本转成 Document 对象
docs = loader.load()

OPENAI_API_BASE = 'http://0.0.0.0:10000/v1'
OPENAI_API_KEY = 'EMPTY'
MODEL_PATH = "qwen2.5-instruct"

llm = ChatOpenAI(model=MODEL_PATH, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY , max_tokens=3000)

# 映射步骤
map_template = """下面是文档的集合
{docs}
根据这文档的集合，请用中文进行总结:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# 归约步骤
reduce_template = """请基于已有的中文摘要和新的内容，生成更新后的中文摘要：\n\n已有摘要: {docs}\n\n新内容: """
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000,
)

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

result = map_reduce_chain.invoke(docs)
print(result["output_text"])

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算耗时
print(f"执行时间: {elapsed_time:.6f} 秒")
```

#### 3. Refine

将长文本分块后，采用链式总结过程，不断合并生成全文总结。

- **优点**：相比Map Reduce，信息丢失较少，因为每个总结都依赖上一个块的信息。
- **缺点**：需要多次调用LLM，且生成阶段无法并行，只能串行，文本过长时生成时间受影响。
  ![1732550311111](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550311111.png)

代码

```
import os
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import CharacterTextSplitter
from docx import Document
from langchain.document_loaders import UnstructuredFileLoader

from langchain.prompts import PromptTemplate


# 定义提示模板
prompt_template = PromptTemplate(
   input_variables=["text"],
   template="请用中文总结以下内容：\n\n{text}")

refine_template = PromptTemplate(
   input_variables=["existing_answer", "text"],
   template="请基于已有的中文摘要和新的内容，生成更新后的中文摘要：\n\n已有摘要: {existing_answer}\n\n新内容: {text}")

def extract_docx_text(docx_path):
    try:
        # 打开docx文档
        doc = Document(docx_path)
  
        # 用于存储所有段落的文本
        full_text = []
  
        # 遍历所有段落
        for para in doc.paragraphs:
            full_text.append(para.text)
  
        # 将所有文本用换行符连接
        return '\n'.join(full_text)
  
    except Exception as e:
        print(f"解析文档时出错: {str(e)}")
        return ""

OPENAI_API_BASE = 'http://0.0.0.0:10000/v1'
OPENAI_API_KEY = 'EMPTY'
MODEL_PATH = "qwen2.5-instruct"

llm = ChatOpenAI(model=MODEL_PATH, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY , max_tokens=2000)

import time 

start_time = time.time()  # 记录开始时间
docx_path = "xxx.docx"
state_of_the_union = extract_docx_text(docx_path)
 

 
# 定义文本分割器 每块文本大小为500，不重叠
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
)
 

summary_chain = load_summarize_chain(
   llm, 
   chain_type="refine",
   verbose=False)

summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain, text_splitter=text_splitter)
res = summarize_document_chain.run(state_of_the_union)
print(res)

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算耗时
print(f"执行时间: {elapsed_time:.6f} 秒")
```

#### 4. Map Rerank

对长文进行Block操作，返回结果时选择相关性分数最高的分块总结。

- **优点**：可以并行执行，效率高。
- **缺点**：多次调用LLM，无法进行全文总结，适合基于相关性指标检索最适合的分块。

![1732550346206](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550346206.png)

#### 5. Binary Map

对长文进行Block操作，按照二分的方式合并结果。

- **优点**：提高合并效率，适用于block过多场景。
- **缺点**：两两组合的方式会丢失一部分信息，但比MR少。

![1732550351508](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550351508.png)

<br />

<br />

<br />

### 结果展示

![1732550409316](https://file+.vscode-resource.vscode-cdn.net/f%3A/NLP/graphRAG/image/main/1732550409316.png)

### 参考文档

[实战 - 对超长文本进行总结 - 《LangChain 中文入门教程》 - 书栈网 · BookStack](https://www.bookstack.cn/read/LangChain-Chinese-Getting-Started-Guide/spilt.3.spilt.4.langchain.md)

[LLM - 长文本总结处理方案_当文档的长度过长时,llm怎样处理-CSDN博客](https://blog.csdn.net/BIT_666/article/details/138184623)

[LangChain进行文本摘要 总结_langchain 文本摘要-CSDN博客](https://blog.csdn.net/weixin_46933702/article/details/139388743)
