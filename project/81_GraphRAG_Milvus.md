# 自然语言处理:第八十一章  用Milvus向量数据库实现GraphRAG

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


GraphRAG技术借助知识图谱，给RAG应用注入了新的动力，使其能够在海量数据中精确检索所需信息。本文将带你了解GraphRAG的实现方法，包括如何创建索引以及如何利用Milvus向量数据库进行查询，助你在信息检索的道路上事半功倍。

## 1 先决条件

在运行本文中的代码之前，请确保已安装以下依赖项：

```
pip install --upgrade pymilvus
pip install git+https://github.com/zc277584121/graphrag.git
```

注意：通过一个分支仓库来安装GraphRAG，这是因为Milvus的存储功能在本文编写时还未被官方正式合并。

## 2 数据准备

为了进行GraphRAG索引，我们需要准备一个小型文本文件。我们将从Gutenberg项目（https://www.gutenberg.org/）下载一个大约一千行的文本文件，这个文件包含了关于达芬奇的故事。

利用这个数据集，构建一个涉及达芬奇所有关系的知识图谱索引，并使用Milvus向量数据库来检索相关知识，以便回答相关问题。

以下是Python代码，用于下载文本文件并进行初步处理：

```
import nest_asyncio
nest_asyncio.apply()

import os
import urllib.request

index_root = os.path.join(os.getcwd(), 'graphrag_index')
os.makedirs(os.path.join(index_root, 'input'), exist_ok=True)
url = "https://www.gutenberg.org/cache/epub/7785/pg7785.txt"
file_path = os.path.join(index_root, 'input', 'davinci.txt')
urllib.request.urlretrieve(url, file_path)
with open(file_path, 'r+', encoding='utf-8') as file:
    # 使用文本文件的前934行，因为后面的行与本例无关。
    # 如果想节省API密钥成本，可以截断文本文件以减小大小。
    lines = file.readlines()
    file.seek(0)
    file.writelines(lines[:934])  # 如果想节省API密钥成本，可以减少这个数字。
    file.truncate()
```

## 3 初始化工作空间

现在，使用GraphRAG对文本文件进行索引。首先运行 `<span leaf="">graphrag.index --init</span>`命令初始化工作空间。

```
python -m graphrag.index --init --root ./graphrag_index
```

## 4 配置环境变量文件

在索引的根目录下，能找到一个名为 `<span leaf="">.env</span>`的文件。要启用这个文件，请将你的OpenAI API密钥添加进去。

**注意事项：**

* 本例将使用OpenAI模型作为一部分，请准备好你的API密钥。
* GraphRAG索引的成本相对较高，因为它需要用LLM处理整个文本语料库。运行这个演示可能会花费一些资金。为了节省成本，你可以考虑将文本文件缩减尺寸。

## 5 执行索引流程

运行索引需要一些时间，请耐心等待。执行完毕后，你可以在 `<span leaf="">./graphrag_index/output/<timestamp>/</span>`路径下找到一个新创建的文件夹，里面包含了多个parquet格式的文件。

执行以下命令开始索引过程：

```
python -m graphrag.index --root ./graphrag_index
```

## 6 使用Milvus向量数据库进行查询

在查询阶段，我们使用Milvus来存储GraphRAG本地搜索中实体描述的向量嵌入。

这种方法将知识图谱的结构化数据与输入文档的非结构化数据相结合，为LLM提供了额外的相关实体信息，从而能够得出更准确的答案。

```
import os

import pandas as pd
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    # read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores import MilvusVectorStore
```

```
output_dir = os.path.join(index_root, "output")
subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)]
latest_subdir = max(subdirs, key=os.path.getmtime)  # 获取最新的输出目录
INPUT_DIR = os.path.join(latest_subdir, "artifacts")
```

```
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
```

## 7 从索引过程中加载数据

在索引过程中，会生成几个parquet文件。我们将其加载到内存中，并将实体描述信息存储在Milvus向量数据库中。

读取实体：

```
# 读取节点表以获取社区和度量数据
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
```

```
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
description_embedding_store = MilvusVectorStore(
    collection_name="entity_description_embeddings",
)
# description_embedding_store.connect(uri="http://localhost:19530") # 用于Milvus docker服务
description_embedding_store.connect(uri="./milvus.db") # For Milvus Lite
entity_description_embeddings = store_entity_semantic_embeddings(
    entities=entities, vectorstore=description_embedding_store
)
print(f"实体数量:{len(entity_df)}")
entity_df.head()
```

实体数量：651

![图片](https://mmbiz.qpic.cn/mmbiz_png/euPYkyibX10VF5xXaUpOlia51remAxsJ8cjPKXNEmgRMwznv9BfsFhew3YNePNojkUjLjzpbOz3lW91uEhKsRpibA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

读取关系

```
relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)
```

```
print(f"关系数量: {len(relationship_df)}")
relationship_df.head()
```

关系数量：290

![图片](https://mmbiz.qpic.cn/mmbiz_png/euPYkyibX10VF5xXaUpOlia51remAxsJ8cnBMnzoKrqD001tH2qKsbEc0C2EKnbdmrFQvCAicHiaMLdG1Oibn0Sd5jA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

读取社区报告

```
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
```

```
print(f"报告记录:{len(report_df)}")
report_df.head()
```

报告记录：45

![图片](https://mmbiz.qpic.cn/mmbiz_png/euPYkyibX10VF5xXaUpOlia51remAxsJ8cjVcJauvZ6AhXqicErZkuRYUekZCiaxibXpmfg2q5CtAlOPJc72H6gGVyw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

读取文本单元

```
text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)
```

```
print(f"文本单元记录:{len(text_unit_df)}")
text_unit_df.head()
```

文本单元记录：51

![图片](https://mmbiz.qpic.cn/mmbiz_png/euPYkyibX10VF5xXaUpOlia51remAxsJ8cvnPB3Mh17iaJGKfhmj035EwMv9wd9F4ibOB9vnURrZtfM0fjtdj2p7ibQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 8 构建本地搜索引擎

一切准备就绪，现在已经有了本地搜索引擎所需的所有数据。接下来，我们可以利用这些数据，配合一个大型语言模型（LLM）和一个嵌入模型，来构建一个 `<span leaf="">LocalSearch</span>`实例，为搜索任务提供强大的支持。

```
api_key = os.environ["OPENAI_API_KEY"]  # 你的OpenAI API密钥
llm_model = "gpt-4o"  # 或 gpt-4-turbo-preview
embedding_model = "text-embedding-3-small"

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)
token_encoder = tiktoken.get_encoding("cl100k_base")
text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_base=None,
    api_type=OpenaiApiType.OpenAI,
    model=embedding_model,
    deployment_name=embedding_model,
    max_retries=20,
)

context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=None, #covariates,#todo
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,  # 如果向量存储使用实体标题作为ID，则将此设置为EntityVectorStoreKey.TITLE
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # 如果向量存储使用实体标题作为ID，则将此设置为EntityVectorStoreKey.TITLE
    "max_tokens": 12
_000,  # 根据你的模型的令牌限制更改此设置（如果你使用的是8k限制的模型，一个好设置可能是5000）
}

llm_params = {
    "max_tokens": 2_000,  # 根据你的模型的令牌限制更改此设置（如果你使用的是8k限制的模型，一个好设置可能是1000=1500）
    "temperature": 0.0,
}

search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",  # 描述响应类型和格式的自由形式文本，可以是任何内容，例如优先列表、单段、多段、多页报告
)
```

## 9 进行查询

```
result = await search_engine.asearch("Tell me about Leonardo Da Vinci")
print(result.response)
```

```
# 莱昂纳多·达·芬奇
莱昂纳多·达·芬奇，1452年出生于佛罗伦萨附近的文奇镇，被广泛誉为意大利文艺复兴时期最多才多艺的天才之一。他的全名是莱昂纳多·迪·塞尔·皮耶罗·达·安东尼奥·迪·塞尔·皮耶罗·迪·塞尔·圭多·达·芬奇，他是塞尔·皮耶罗的非婚生和长子，塞尔·皮耶罗是一位乡村公证人[数据：实体（0]。莱昂纳多的贡献涵盖了艺术、科学、工程和哲学等多个领域，他被誉为基督教时代最万能的天才[数据：实体（8]。

## 早期生活和训练
莱昂纳多早期的才华得到了他父亲的认可，他父亲将他的一些画作带给了安德烈亚·德尔·维罗基奥，一位著名的艺术家和雕塑家。维罗基奥对莱昂纳多的才华印象深刻，于1469-1470年左右接受了他进入自己的工作室。在这里，莱昂纳多遇到了包括博蒂切利和洛伦佐·迪·克雷迪在内的其他著名艺术家[数据：来源（6, 7]。到1472年，莱昂纳多被佛罗伦萨画家行会录取，标志着他职业生涯的开始[数据：来源（7）]。

## 艺术杰作
莱昂纳多或许以他的标志性画作最为人所知，如《蒙娜丽莎》和《最后的晚餐》。《蒙娜丽莎》以其微妙的表情和详细的背景而闻名，现藏于卢浮宫，仍然是世界上最著名的艺术品之一[数据：关系（0, 45]。《最后的晚餐》是一幅壁画，描绘了耶稣宣布他的一个门徒将背叛他的那一刻，位于米兰圣玛利亚·格拉齐教堂的餐厅[数据：来源（2]。其他重要作品包括《岩间圣母》和他大约在1489-1490年开始的《绘画论》[数据：关系（7, 12]。

## 科学和工程贡献
莱昂纳多的天才超越了艺术，延伸到各种科学和工程事业。他在解剖学、光学和水力学方面做出了重要观察，他的笔记本里充满了预示许多现代发明的草图和想法。例如，他预示了哥白尼关于地球运动的理论和拉马克对动物的分类[数据：关系（38, 39]。他对光影法则和明暗对比的掌握对艺术和科学都产生了深远影响[数据：来源（45）]。

## 赞助和职业关系
莱昂纳多的职业生涯受到他的赞助人的重大影响。米兰公爵卢多维科·斯福尔扎雇佣莱昂纳多作为宫廷画家和普通工匠，委托了各种作品，甚至在1499年赠送给他一个葡萄园[数据：关系（9, 19, 84]。在他的晚年，莱昂纳多在法国国王弗朗西斯一世的赞助下搬到了法国，国王为他提供了丰厚的收入，并对他评价很高[数据：关系（114, 37]。莱昂纳多在法国安博瓦兹附近的克洛克斯庄园度过了他最后几年，国王经常拜访他，他得到了他的密友和助手弗朗切斯科·梅尔齐的支持[数据：关系（28, 122）]。

## 遗产和影响
莱昂纳多·达·芬奇的影响远远超出了他的一生。他在米兰创立了一所绘画学校，他的技术和教导被他的学生和追随者，如乔瓦尼·安布罗焦·达·普雷迪斯和弗朗切斯科·梅尔齐传承下去[数据：关系（6, 15, 28]。他的作品继续受到庆祝和研究，巩固了他作为文艺复兴时期最伟大的大师之一的遗产。莱昂纳多将艺术和科学融合的能力在两个领域都留下了不可磨灭的印记，激励着无数的艺术家和科学家[数据：实体（148, 86]; 关系（27, 12）]。

总之，莱昂纳多·达·芬奇对艺术、科学和工程的无与伦比的贡献，加上他的创新思维和对同时代人及后代的深远影响，使他成为人类成就史上的一位杰出人物。他的遗产继续激发着钦佩和研究，强调了他的天才的永恒相关性。
```

GraphRAG的结果具体，明确标出了引用的数据源。

## 10 问题生成

GraphRAG还可以根据历史查询生成问题，这对于在聊天机器人对话中创建推荐问题非常有用。这种方法结合了知识图谱的结构化数据和输入文档的非结构化数据，产生与特定实体相关的候选问题。

```
question_generator = LocalQuestionGen(
   llm=llm,
   context_builder=context_builder,
   token_encoder=token_encoder,
   llm_params=llm_params,
   context_builder_params=local_context_params,
)

question_history = [
    "Tell me about Leonardo Da Vinci",
    "Leonardo's early works",
]

Generate questions based on history.

candidate_questions = await question_generator.agenerate(
        question_history=question_history, context_data=None, question_count=5
    )
candidate_questions.response

["- 莱昂纳多·达·芬奇的早期作品有哪些，它们存放在哪里？",
"莱昂纳多·达·芬奇与安德烈亚·德尔·维罗基奥的关系如何影响了他的早期作品？",
"莱昂纳多·达·芬奇在米兰期间承担了哪些重要项目？",
"莱昂纳多·达·芬奇的工程技能如何促成他的项目？",
"莱昂纳多·达·芬奇与法国弗朗西斯一世的关系有何重要性？"]
```

如果你想删除索引以节省空间，可以移除索引根。

```
# import shutil
#
# shutil.rmtree(index_root)
```

## 11 结语

本文带领大家深入了解了GraphRAG技术，这是一种融合知识图谱来强化RAG应用的创新手段。GraphRAG特别擅长处理那些需要跨信息片段进行多步骤推理和全面回答问题的复杂任务。

结合Milvus向量数据库后，GraphRAG能够高效地在庞大的数据集中探索复杂的语义联系，从而得出更精准、更深刻的分析结果。这种强强联合的解决方案，使GraphRAG成为众多实际通用人工智能（GenAI）应用中的得力助手，为理解和处理复杂信息提供了强有力的支持。
