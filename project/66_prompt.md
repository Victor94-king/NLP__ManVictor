# 自然语言处理:第六十六章 17 种 prompt engineering 方法大集合

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

原文地址：[17 种 prompt engineering 方法大集合](https://mp.weixin.qq.com/s/ezkeDKMbGiRS7TltGLjC-w)


<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

<br />

本文主要介绍prompt engineering的多种方法，具体如下：

1. **Zero-shot prompt**
2. **Few-shot prompt**
3. **链式思考（COT）prompt**
4. **自我一致性**
5. **生成知识prompt**
6. **prompt chaining**
7. **思维树（TOT）**
8. **检索增强生成（RAG）**
9. **自动推理并使用工具（ART）**
10. **自动prompt工程师**
11. **Activet-prompt**
12. **方向性刺激prompt**
13. **PAL程序辅助语言模型**
14. **ReAct框架**
15. **自我反思 Reflexion**
16. **多模态思维链prompt**
17. **基于图的prompt**

下面，将逐一进行介绍。

参考链接：https://www.promptingguide.ai/zh

<br />

<br />

<br />

## Part.**01 Zero-shot prompt**

**1.介绍**

Zero-shot prompt 是一种无需提供示例或训练数据，直接利用自然语言描述任务或问题，让模型生成相应回答的技术方法。

**2.示例**

```
# prompt
将文本分类为中性、负面或正面。
文本：我认为这件事也太好了！
情感：


# output
正面
```

**3.相关论文：**

FINETUNED LANGUAGE MODELS ARE ZERO-SHOTLEARNERS （https://arxiv.org/pdf/2109.01652）

<br />

<br />

<br />

## Part.**02 Few-shot prompt**

**1.介绍**

Few-shot prompt 是一种通过提供少量示例来引导模型生成回答的技术，能在一定程度上提升模型的表现和适应性。

**2.示例**

```
# prompt
这太棒了！// Negative
这太糟糕了！// Positive
哇，那部电影太棒了！// Positive
多么可怕的节目！//


# output
Negative
```

**3.相关论文**

（1）LLaMA: Open and Efficient Foundation Language Models（https://arxiv.org/pdf/2302.13971）

（2）Language Models are Few-Shot Learners（https://arxiv.org/abs/2005.14165）

（3）Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?（https://arxiv.org/abs/2202.12837）

<br />

<br />

<br />

## Part.**03 链式思考（COT）prompt**

**1.介绍**

链式思考（COT）prompt 是一种引导模型进行逐步推理和连续思考，以生成更深入、准确答案的提示方法。

**2.示例**

```
# prompt
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。然后我去买了5个苹果并吃了1个。我还剩下多少苹果？
让我们逐步思考。


# output
首先，您从10个苹果开始。
您给了邻居和修理工各2个苹果，所以您还剩下6个苹果。
然后您买了5个苹果，所以现在您有11个苹果。
最后，您吃了1个苹果，所以您还剩下10个苹果。
```

**3.自动思维链 ****Auto-CoT**

Auto-CoT对具有多样性的问题进行采样，并生成推理链来构建演示。主要包含两个阶段：

 **（1）问题聚类：** 将给定问题划分为几个聚类

 **（2）演示抽样：** 从每组数组中选择一个具有代表性的问题，并使用带有简单启发式的 Zero-Shot-CoT 生成其推理链。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmiby0tGIDRDEx3T6FXiaz3nyIcmtGqbMaCVLzSuzx7rgAO4rKKCT8gJCA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**4.相关论文**

（1）Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（https://arxiv.org/abs/2201.11903）

（2）Large Language Models are Zero-Shot Reasoners（https://arxiv.org/abs/2205.11916）

（3）Automatic Chain of Thought Prompting in Large Language Models（https://arxiv.org/abs/2210.03493）

<br />

<br />

<br />

## Part.**04 自我一致性**

**1.介绍**

自我一致性旨在“替换链式思维提示中使用的天真贪婪解码方法”。其想法是 **通过少样本 CoT 采样多个不同的推理路径，并使用生成结果选择最一致的答案** 。有助于提高 CoT 提示在涉及算术和常识推理的任务中的性能。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmSBWywUB9ZAjFsdFVmWoELNibjFTWnUGXBSnsnibanJGpqdyOrpicExb1Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

```
# prompt
Q：林中有15棵树。林业工人今天将在林中种树。完成后，将有21棵树。林业工人今天种了多少棵树？
A：我们从15棵树开始。后来我们有21棵树。差异必须是他们种树的数量。因此，他们必须种了21-15 = 6棵树。答案是6。
Q：停车场有3辆汽车，又来了2辆汽车，停车场有多少辆汽车？
A：停车场已经有3辆汽车。又来了2辆。现在有3 + 2 = 5辆汽车。答案是5。
Q：Leah有32块巧克力，她的姐姐有42块。如果他们吃了35块，他们总共还剩多少块？
A：Leah有32块巧克力，Leah的姐姐有42块。这意味着最初有32 + 42 = 74块巧克力。已经吃了35块。因此，他们总共还剩74-35 = 39块巧克力。答案是39。
Q：Jason有20个棒棒糖。他给Denny一些棒棒糖。现在Jason只有12个棒棒糖。Jason给Denny多少棒棒糖？
A：Jason有20个棒棒糖。因为他现在只有12个，所以他必须把剩下的给Denny。他给Denny的棒棒糖数量必须是20-12 = 8个棒棒糖。答案是8。
Q：Shawn有五个玩具。圣诞节，他从他的父母那里得到了两个玩具。他现在有多少个玩具？
A：他有5个玩具。他从妈妈那里得到了2个，所以在那之后他有5 + 2 = 7个玩具。然后他从爸爸那里得到了2个，所以总共他有7 + 2 = 9个玩具。答案是9。
Q：服务器房间里有9台计算机。从周一到周四，每天都会安装5台计算机。现在服务器房间里有多少台计算机？
A：从周一到周四有4天。每天都添加了5台计算机。这意味着总共添加了4 * 5 =
20台计算机。一开始有9台计算机，所以现在有9 + 20 = 29台计算机。答案是29。
Q：Michael有58个高尔夫球。星期二，他丢失了23个高尔夫球。星期三，他又丢失了2个。星期三结束时他还剩多少个高尔夫球？
A：Michael最初有58个球。星期二他丢失了23个，所以在那之后他有58-23 = 35个球。星期三他又丢失了2个，所以现在他有35-2 = 33个球。答案是33。
Q：Olivia有23美元。她用每个3美元的价格买了五个百吉饼。她还剩多少钱？
A：她用每个3美元的价格买了5个百吉饼。这意味着她花了15美元。她还剩8美元。
Q：当我6岁时，我的妹妹是我的一半年龄。现在我70岁了，我的妹妹多大？
A：


# output1
当我6岁时，我的妹妹是我的一半年龄，也就是3岁。现在我70岁了，所以她是70-3 = 67岁。答案是67。
# output2
当叙述者6岁时，他的妹妹是他年龄的一半，也就是3岁。现在叙述者70岁了，他的妹妹应该是70-3 = 67岁。答案是67。
# output3
当我6岁时，我的妹妹是我的一半年龄，也就是3岁。现在我70岁了，所以她是70/2 = 35岁。答案是35。


# 后处理
找出推理路径并通过选择最终答案集中最一致的答案进行聚合。
```

**3.相关论文**

SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHTREASONING IN LANGUAGE MODELS（https://arxiv.org/pdf/2203.11171）

<br />

<br />

<br />

## Part.**05 生成知识prompt**

**1.介绍**

生成知识prompt是指先从语言模型中生成知识，然后在回答问题时作为额外的知识进行提供。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmrgZlgFgNDnydcqJR90ib2eHTuECWOrxrWiab3PhxDJUlWvEFpeKUdXicw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

```
# 生成知识
输入：希腊比墨西哥大。
知识：希腊的面积约为131,957平方公里，而墨西哥的面积约为1,964,375平方公里，使墨西哥比希腊大了1,389%。
输入：眼镜总是会起雾。
知识：当你的汗水、呼吸和周围的湿度中的水蒸气落在冷的表面上，冷却并变成微小的液滴时，会在眼镜镜片上产生冷凝。你看到的是一层薄膜。你的镜片相对于你的呼吸会比较凉，尤其是当外面的空气很冷时。
输入：鱼有思考能力。
知识：鱼比它们看起来更聪明。在许多领域，如记忆力，它们的认知能力与或超过非人类灵长类动物等“更高级”的脊椎动物。鱼的长期记忆帮助它们跟踪复杂的社交关系。
输入：一个人一生中吸烟很多香烟的常见影响是患肺癌的几率高于正常水平。
知识：那些一生中平均每天吸烟不到一支香烟的人，患肺癌的风险是从不吸烟者的9倍。在每天吸烟1到10支香烟之间的人群中，死于肺癌的风险几乎是从不吸烟者的12倍。
输入：一块石头和一颗卵石大小相同。
知识：卵石是一种根据Udden-Wentworth沉积学尺度的颗粒大小为4到64毫米的岩屑。卵石通常被认为比颗粒（直径2到4毫米）大，比卵石（直径64到256毫米）小。
输入：高尔夫球的一部分是试图获得比其他人更高的得分。
知识：


# 知识
高尔夫球的目标是以最少的杆数打完一组洞。一轮高尔夫球比赛通常包括18个洞。每个洞在标准高尔夫球场上一轮只打一次。每个杆计为一分，总杆数用于确定比赛的获胜者。


# 将知识整合并得出预测。将问题重新格式化为 QA 格式，以指导答案格式。
问题：高尔夫球的一部分是试图获得比其他人更高的得分。是或否？
知识：高尔夫球的目标是以最少的杆数打完一组洞。一轮高尔夫球比赛通常包括18个洞。每个洞在标准高尔夫球场上一轮只打一次。每个杆计为一分，总杆数用于确定比赛的获胜者。
解释和答案： 


# 答案
不是，高尔夫球的目标不是获得比其他人更高的得分。相反，目标是以最少的杆数打完一组洞。总杆数用于确定比赛的获胜者，而不是总得分。
```

**3.相关论文**

Generated Knowledge Prompting for Commonsense Reasoning（https://arxiv.org/pdf/2110.08387）

<br />

<br />

<br />

<br />

## Part.**06 prompt chaining**

**1.介绍**

链式提示（prompt chaining），一个任务被分解为多个子任务，根据子任务创建一系列prompt操作。

**2.示例**

文档问答中的链式提示：想要更好阅读大文本文档，可以设计两个不同的prompt，第一个prompt负责提取相关引文以回答问题，第二个prompt则以引文和原始文档为输入来回答给定的问题。换句话说，可以创建两个不同的prompt来执行根据文档回答问题的任务。

```
# prompt1
你是一个很有帮助的助手。你的任务是根据文档回答问题。第一步是从文档中提取与问题相关的引文，由####分隔。请使用<quotes></quotes>输出引文列表。如果没有找到相关引文，请回应“未找到相关引文！”。
####
{{文档}}
####


# output1
<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>


# prompt2
根据从文档中提取的相关引文（由<quotes></quotes>分隔）和原始文档（由####分隔），请构建对问题的回答。请确保答案准确、语气友好且有帮助。
####
{{文档}}
####
<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>


# output2
文档中提到的提示技术包括：
1. Chain-of-thought (CoT) prompting[27]
2. Generated knowledge prompting[37]
3. Least-to-most prompting[38]
4. Self-consistency decoding[39]
5. Complexity-based prompting[41]
6. Self-refine[42]
7. Tree-of-thought prompting[43]
8. Maieutic prompting[45]
9. Directional-stimulus prompting[46]
10. Textual inversion and embeddings[59]
11. Using gradient descent to search for prompts[61][62][63][64]
12. Prompt injection[65][66][67]
每一种技术都采用独特的策略来增强或明确与大型语言模型的互动，以产生期望的结果。
```

<br />

<br />

<br />

## Part.**07 思维树（TOT）**

**1.介绍**

ToT 维护着一棵思维树，思维由连贯的语言序列表示，这个序列就是解决问题的中间步骤。使用这种方法，LM 能够自己对严谨推理过程的中间思维进行评估。LM 将生成及评估思维的能力与搜索算法（如广度优先搜索和深度优先搜索）相结合，在系统性探索思维的时候可以向前验证和回溯。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmPR6r6IP568PSylibianiaKKszygaAicItziaBwtCZXfZLOp6JmmIr1UDeRQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

**（1）24点游戏**

 **任务说明：** 24点游戏是一个数学推理挑战，目标是使用4个数字和基本的四则运算（+-*/）得到24。例如，给定输入“4 9 10 13”，一个解决方案的输出可以是“(10 - 4) * (13 - 9) = 24”。

 **步骤：** 当将 24 点游戏框架化为 ToT 时，将思考分解为 3 个步骤，且每个步骤都是一个中间等式。如图 所示，在每个树节点处，我们提取“left”数字，并提示LM提出一些可能的下一步。对于这 3 个思考步骤，使用的是同一个“提出提示”，尽管它仅包含一个由 4 个输入数字组成的示例。在 ToT 中进行广度优先搜索（BFS），在每个步骤中保留最佳的 b=5 个候选者。为了在 ToT 中进行有意识的广度优先搜索，提示LM对每个思考候选进行评估，以确定其是否为“确定/可能/不可能”达到 24。其目的是促进那些可以在少数向前看试验中判定的正确部分解，并基于“太大/太小”的常识来消除不可能的部分解，而其余部分则视为“可能”。我们对每个思考值进行 3 次采样。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmLYO9FhDUWwjvpumQxsIkpG89BcCxDdpErWrvWwAob1NJ6at1DpGgTw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**（2）创意写作**

 **任务说明：** 输入是4个随机的句子，输出应该是一个有4个段落的连贯段落，每个段落以相应的输入句子结尾。

 **步骤：** 构建了一个深度为 2（仅包含 1 个中间思维步骤）的 ToT。首先，语言模型生成 k=5 个计划，并从中挑选出最佳计划；然后，同样基于最佳计划，语言模型又生成 k=5 个段落，并选出最佳段落。在这里，广度限制设置为 b=1，即每个步骤仅保留一个选择。在两个步骤上都使用了简单的零样本投票提示（“分析下面的选择，然后决定哪一个对指令最有前途”），并进行了 5 次投票。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmviaCm6OVM7ruyko18Ee63Y8rQcAXDIHDLJ0RHf72a2LegY9hIxSMnwg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmviaCm6OVM7ruyko18Ee63Y8rQcAXDIHDLJ0RHf72a2LegY9hIxSMnwg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**（3）迷你纵横游戏**

 **任务说明：** 对于每个任务，输入描述了5个水平线索和5个垂直线索，输出应该是一个5×5 = 25个字母的填字板，以解决纵横填字游戏。

 **步骤：** 利用深度优先搜索持续探索最有潜力的下一个单词线索，直至状态无前途时再回溯到父状态探索其他思维。为保证搜索可行，后续思维不能改变已填充的单词或字母，因此 ToT 最多有 10 个中间思维步骤。在思维生成时，每个状态下，将现有的思维转化为剩余线索的字母约束，并提示 5 次提案，提供下一个单词的填充位置和内容的候选方案。同时，提示LLM给出不同思维的置信水平并汇总，得到下一个要探索思维的排序清单。在状态评估时，也类似地将状态转化为剩余线索的字母约束，然后评估对于每个线索，在给定约束的情况下是否可能填充。若有剩余线索被认为 “不可能” 填充，则剪枝该状态子树的探索，深度优先搜索回溯到父状态探索下一个有前途的思维。将 DFS 搜索步骤限制为 100 步，将最深探索的状态（多个状态时为第一个探索的状态）输出为最终输出。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmDumGhRq3sCgicbUEuTJVE3KVZkWytenmqnelg2icicVkXoJt8vNDdc4Cw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**（4）将 ToT 框架的主要概念概括成了一段简短的提示词，指导 LLM 在一次提示中对中间思维做出评估**

```
# prompt示例
假设三位不同的专家来回答这个问题。
所有专家都写下他们思考这个问题的第一个步骤，然后与大家分享。
然后，所有专家都写下他们思考的下一个步骤并分享。
以此类推，直到所有专家写完他们思考的所有步骤。
只要大家发现有专家的步骤出错了，就让这位专家离开。
请问...
```

**3.相关论文**

（1）Tree of Thoughts: Deliberate Problem Solvingwith Large Language Models（https://arxiv.org/pdf/2305.10601）

（2）Large Language Model Guided Tree-of-Thought

（https://arxiv.org/pdf/2305.08291）

（3）Github：https://github.com/princeton-nlp/tree-of-thought-llm

（4）https://github.com/jieyilong/tree-of-thought-puzzle-solver

（5）https://github.com/princeton-nlp/tree-of-thought-llm

<br />

<br />

<br />

## Part.**08 检索增强生成（RAG）**

**1.介绍**

RAG 会接受输入并检索出一组相关文档，并给出文档的来源。这些文档作为上下文和输入的原始提示词组合，送给文本生成器得到最终的输出。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmPJtaqn81WhXb67Ad79lqY63lYHl43KvSQQy3Wcicry9dsYoI34E1xzQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

基于LangChain构建检索增强生成（RAG）应用程序：

https://python.langchain.com/docs/tutorials/rag/

**3.相关论文**

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（https://arxiv.org/abs/2005.11401）

<br />

<br />

<br />

## Part.**09 自动推理并使用工具（ART）**

**1.介绍**

ART（Automatic Reasoning and Tool-use）的工作原理如下：

**（1）接到一个新任务的时候，从任务库中选择多步推理和使用工具的示范。**

**（2）在测试中，调用外部工具时，先暂停生成，将工具输出整合后继续接着生成。**

ART 引导模型总结示范，将新任务进行拆分并在恰当的地方使用工具。ART 采用的是零样本形式。ART 还可以手动扩展，只要简单地更新任务和工具库就可以修正推理步骤中的错误或是添加新的工具。这个过程如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmvR9nKOAFrFOrIFrp1uzwxGTdMnIJaicQ1ETQrhmibqun4LTmib4FbVqCg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.论文**

（1）ART: Automatic multi-step reasoning and tool-use forlarge language models（https://arxiv.org/pdf/2303.09014）

<br />

<br />

<br />

## Part.**10 自动prompt工程师**

**1.介绍**

自动提示工程师（APE）是一种将指令生成视为自然语言程序合成的方法，通过让大语言模型（LLM）生成并搜索候选指令，利用执行准确性或对数概率等评分函数选择最优指令，从而自动生成和选择指令，在零样本学习等任务上实现了与人类水平相当的性能。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmUxRG4Wj7qLPNAaVBWjDthIUwd8IAtYs13lcQVssNLHHFxxdKC8h1wg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

自动提示工程师（APE）的工作流程，其主要步骤如下：

 **（1）生成指令候选** ：APE 首先利用LLM，根据少量输入 - 输出对形式的示例，生成若干指令候选。这一步可以通过 “正向” 模式生成（如在指令归纳实验中，按照特定模板让 LLM 生成）或 “反向” 模式生成（利用具有填充能力的 LLM，如 T5、GLM、InsertGPT 等，通过填充空白来推断缺失的指令）。

 **（2）评估指令得分：** 对于每个生成的指令候选，APE 在目标模型下通过计算所选评分函数（如执行准确性或对数概率）的得分来评估其质量。评估过程会使用训练数据的子集，采用自适应过滤方案，先在小数据集上评估所有候选指令，对于得分高于一定阈值的候选指令，再从训练数据中采样新的非重叠子集进行评估，以更新得分的移动平均值，最终在整个训练数据集上评估剩余的少量候选指令，以降低计算成本。

 **（3）筛选和优化指令：** 根据计算得到的得分，筛选出得分最高的前 k% 的指令，组成新的指令集。然后，可以选择直接使用这些指令，或者利用 LLM 再次生成与高分指令语义相似的新指令，进入下一轮迭代，不断优化指令集，直到满足收敛条件。最终选择得分最高的指令作为输出，用于引导目标模型执行任务。

通过这样的流程，APE 能够自动寻找最适合任务的指令，提高模型在各种任务上的性能。

**2.论文**

（1）LARGE LANGUAGE MODELS ARE HUMAN-LEVELPROMPT ENGINEERS（https://arxiv.org/pdf/2211.01910）

（2）Large Language Models are Zero-Shot Reasoners （https://arxiv.org/pdf/2205.11916）

<br />

<br />

<br />

## Part.1**1 Activet-prompt**

**1.介绍**

通过基于不确定性的主动选择策略，从任务特定问题池中挑选最有帮助和信息量大的问题进行注释，引入了如分歧、熵等多种不确定性度量方式，以提升模型在复杂推理任务中的性能。

流程包括：

 **（1）不确定性估计（Uncertainty Estimation）** ：在思维链设置下，将LLM前向传递 k 次，为每个问题获取 k 个答案，然后通过不同方式衡量问题的不确定性。例如，用分歧（disagreement）计算预测答案中唯一答案的比例；用熵（entropy）根据预测答案的频率分布计算不确定性；用方差（variance）计算预测答案的方差，并对预测答案进行归一化处理。

 **（2）选择与注释（Selection and Annotation）：** 根据每个问题的不确定性进行排序，选择不确定性最大的 n 个问题（若不确定性最大的问题数量超过 n，则随机选择 n 个），由人工注释者为这些问题添加推理步骤和答案，构建新的示例集 E。

 **（3）推理（Inference）：** 在推理阶段，使用新注释的示例集 E 为每个测试问题添加提示，并应用自一致性（self - consistency）方法，将温度 T 设置为 0.7，对每个问题进行 40 次推理，最后选择最一致的答案作为预测结果。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmicJ7iblhbicsiaW6s3rEpF3UkDtC8nLUsiccFEBbHGv9EicPZVvaOc6MONiaA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.论文**

（1）Active Prompting with Chain-of-Thought for Large Language Models （https://arxiv.org/pdf/2302.12246）

<br />

<br />

<br />

## Part.**12 方向性刺激prompt**

**1.介绍**

通过小型可调策略模型为输入实例生成辅助方向刺激提示，以引导黑盒大语言模型生成符合期望输出的新框架。

下图中可以看出，对于同一篇新闻文章，除了原始输入外，还会提供诸如 “Bob Barker; TV; April 1; "The Price Is Right”; 2007; 91.” 这样的提示信息。模型根据这些提示生成摘要，如 “On April 1, Bob Barker returned to the TV show"The Price Is Right" after eight years. Looking spry at 91, Barker handled the first price-guessing game. Barker stepped down as host in 2007, but didn't seem to miss a beat.”。这些提示作为 “暗示” 和 “线索”，为模型提供了更具体的指导，帮助模型更好地理解用户期望在摘要中包含的关键内容。通过这种方式，模型能够生成更接近参考摘要的输出。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmqRHXgSMlia5yeJ2wBou39AFj4nibln5tQfk6Tib03XRicsur766D0THsicQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.论文**

（1）Guiding Large Language Models viaDirectional Stimulus Prompting（https://arxiv.org/pdf/2302.11520）

<br />

<br />

<br />

## Part.**13 PAL程序辅助语言模型**

**1.介绍**

PAL程序辅助语言模型是一种使用 LLMs 读取自然语言问题并生成程序作为中间推理步骤的方法。它与思维链prompt不同，因为它不是使用自由形式文本来获得解决方案，而是将解决步骤卸载到类似 Python 解释器的编程运行时中。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmYCfebUib77BfictfIuibeuUU9zpsF0lR650rUhf7IpM2pHORztVbJNOtQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

```
import openai
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv


# 环境配置
load_dotenv()
## API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
## for LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# 设置模型实例
llm = OpenAI(model_name='text-davinci-003', temperature=0)


# 设置提示+问题：
question = "Today is 27 February 2023. I was born exactly 25 years ago. What is the date I was born in MM/DD/YYYY?"
DATE_UNDERSTANDING_PROMPT = """
# Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?
# If 2015 is coming in 36 hours, then today is 36 hours before.
today = datetime(2015, 1, 1) - relativedelta(hours=36)
# One week from today,
one_week_from_today = today + relativedelta(weeks=1)
# The answer formatted with %m/%d/%Y is
one_week_from_today.strftime('%m/%d/%Y')
# Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?
# If the first day of 2019 is a Tuesday, and today is the first Monday of 2019, then today is 6 days later.
today = datetime(2019, 1, 1) + relativedelta(days=6)
# The answer formatted with %m/%d/%Y is
today.strftime('%m/%d/%Y')
# Q: The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?
# If the concert was scheduled to be on 06/01/1943, but was delayed by one day to today, then today is one day later.
today = datetime(1943, 6, 1) + relativedelta(days=1)
# 10 days ago,
ten_days_ago = today - relativedelta(days=10)
# The answer formatted with %m/%d/%Y is
ten_days_ago.strftime('%m/%d/%Y')
# Q: It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?
# It is 4/19/1969 today.
today = datetime(1969, 4, 19)
# 24 hours later,
later = today + relativedelta(hours=24)
# The answer formatted with %m/%d/%Y is
today.strftime('%m/%d/%Y')
# Q: Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?
# If Jane thought today is 3/11/2002, but today is in fact Mar 12, then today is 3/12/2002.
today = datetime(2002, 3, 12)
# 24 hours later,
later = today + relativedelta(hours=24)
# The answer formatted with %m/%d/%Y is
later.strftime('%m/%d/%Y')
# Q: Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?
# If Jane was born on the last day of Feburary in 2001 and today is her 16-year-old birthday, then today is 16 years later.
today = datetime(2001, 2, 28) + relativedelta(years=16)
# Yesterday,
yesterday = today - relativedelta(days=1)
# The answer formatted with %m/%d/%Y is
yesterday.strftime('%m/%d/%Y')
# Q: {question}
""".strip() + '\n'


llm_out = llm(DATE_UNDERSTANDING_PROMPT.format(question=question))
print(llm_out)


# 输出内容
# If today is 27 February 2023 and I was born exactly 25 years ago, then I was born 25 years before.
today = datetime(2023, 2, 27)
# I was born 25 years before,
born = today - relativedelta(years=25)
# The answer formatted with %m/%d/%Y is
born.strftime('%m/%d/%Y')


# llm_out 是一段 python 代码，我们可以使用 exec 执行它：
exec(llm_out)
print(born)
# 输出 02/27/1998
```

**3.论文**

PAL: Program-aided Language Models

（https://arxiv.org/abs/2211.10435）

<br />

<br />

<br />

## Part.**14 ReAct框架**

**1.介绍**

ReAct 是一个将推理和行为与 LLMs 相结合通用的范例。ReAct 提示 LLMs 为任务生成口头推理轨迹和操作。这使得系统执行动态推理来创建、维护和调整操作计划，同时还支持与外部环境(例如，Wikipedia)的交互，以将额外信息合并到推理中。下图展示了 ReAct 的一个示例以及执行问题回答所涉及的不同步骤。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmznuaFCSwibHfetWJAGk85jc6hnC3SVrc6c9oekI0kyvv3vm7MV0rYZA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.示例**

```
agent.run("奥利维亚·王尔德的男朋友是谁?他现在的年龄的0.23次方是多少?")
> 正在输入新代理执行器链......
  我得查出奥利维亚·王尔德的男友是谁然后计算出他的年龄的 0.23 次方。
操作: 搜索
操作输入: “奥利维亚·王尔德的男友”
观察: 奥利维亚·王尔德与杰森·苏代基斯在多年前订婚，在他们分手后，她开始与哈里·斯泰尔斯约会 — 参照他们的关系时间线。
思考: 我需要找出哈里·斯泰尔斯的年龄。
操作: 搜索
操作输入: “哈里·斯泰尔斯的年龄”
观察: 29 岁
思考: 我需要计算 29 的 0.23 次方。
操作: 计算器
操作输入: 29^0.23
观察: 答案: 2.169459462491557

思考: 现在我知道最终答案了。
最终答案: 哈里·斯泰尔斯, 奥利维亚·王尔德的男朋友, 29 岁。他年龄的 0.23 次方是 2.169459462491557。

> 结束链。
```

相关示例：https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/notebooks/react.ipynb

**3. 论文：**

REACT: SYNERGIZING REASONING AND ACTING INLANGUAGE MODELS（https://arxiv.org/pdf/2210.03629）

<br />

<br />

<br />

## Part.15 自我反思 **Reflexion**

**1.介绍**

自我反思 Reflexion通过将策略参数化为agent的记忆编码与 LLM 参数的组合，使智能体能够从过往失败中学习，而无需更新模型权重，为语言智能体的训练提供了一种轻量级且高效的替代方法。

如下图，一个 Reflexion agent通过试验、错误和自我反思来学习优化自身行为，以解决决策、编程和推理任务。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSm3zUTZc5lDgdL4f35I7BIGctGX0O5bmqwVjicccAMibtHopqQEOEJtKoA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**（1）Reflexion Actor 指令**

```
你是一个 Python 写作助手。你将得到你之前实现的一个函数、一系列单元测试结果以及你对之前实现的自我反思。通过仅回复改进后的函数主体来应用下面的必要更改。在你的回复中不要包含签名。你的回复的第一行应该有 4 个空格的缩进，以便在语法上与用户提供的签名相匹配。用户会给你几个例子。
Reflexion Actor生成遵循以下形式：
（指令）
（函数实现）
（单元测试反馈）
（自我反思）
（下一个函数实现的指令）
```

**（2）自我反思的指令和示例**

```
你是一个 Python 写作助手。你将得到你之前实现的一个函数、一系列单元测试结果以及你对之前实现的自我反思。通过仅回复改进后的函数主体来应用下面的必要更改。在你的回复中不要包含签名。你的回复的第一行应该有 4 个空格的缩进，以便在语法上与用户提供的签名相匹配。用户会给你一些例子。反思自我反思的生成遵循以下形式：
（指令）
（函数实现）
（单元测试反馈）
```

**2.论文**

（1）Reflexion: Language Agents withVerbal Reinforcement Learning（https://arxiv.org/pdf/2303.11366）

（2）Can LLMs Critique and Iterate on Their Own Outputs?（https://evjang.com/2023/03/26/self-reflection.html）

（3）Github：https://github.com/noahshinn024/reflexion

<br />

<br />

<br />

## Part.**16 多模态思维链prompt**

**1.介绍**

多模态思维链提示将文本和视觉融入到一个两阶段框架中。

多模态思维链由两个阶段组成：（i）推理生成阶段和（ii）答案推断阶段。两个阶段共享相同的模型结构，但在输入和输出方面有所不同。在第一阶段，我们向模型输入语言和视觉信息以生成推理。在第二阶段，将第一阶段生成的推理附加到原始语言输入上。然后，将更新后的语言输入与原始视觉输入一起提供给模型以推断答案。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSmibDFOfibe5nfIDB91jDW0FpM7nxCusTGNvjQkVOW1kQruSibyXfsCU8yw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.论文**

（1）Multimodal Chain-of-Thought Reasoning in Language Models（https://arxiv.org/pdf/2302.00923）

（2）Github：https://github.com/amazon-science/mm-cot.

<br />

<br />

<br />

## Part.**17 基于图的prompt**

**1.介绍**

通过 **将预训练和下游任务统一为基于子图相似性的模板** ，GraphPrompt 实现了两者的有效结合。预训练学习到的子图相似性知识能够自然地迁移到下游任务中，无论是节点分类还是图分类任务，都能基于此进行有效的预测。

 **可学习prompt向量在下游任务中进一步区分不同任务的需求** ，根据任务的特点引导 ReadOut 操作采用不同的聚合方案，从而使下游任务能够更好地利用预训练模型，实现任务特定的最优性能。这种框架设计使得 GraphPrompt 在处理不同下游任务时具有灵活性和有效性，能够在有限监督的情况下（如少样本学习任务）取得较好的效果。

![图片](https://mmbiz.qpic.cn/mmbiz_png/EzsRicoa4TRrHDksR8h7bgokC6dsKibLSm34p7xot3J6YDVJ13awjh5V1gHORpoujMyRJHKNyDy4zAGPGSL5hicoQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2.论文**

GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks（https://arxiv.org/pdf/2302.08043）
