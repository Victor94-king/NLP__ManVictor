# 自然语言处理:第一百零二章 如何去掉DeepSeek R1思考过程

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


## 在与类似DeepSeek类似的大语言模型交互时，你是否曾经遇到过这样的困惑：

> DeepSeek-R1的思考过程是否可以去掉？

大多数情况下，我们希望AI能直接给出答案，而不是在输出中展示“思考过程”。DeepSeek-R1会在回答前生成一个 `<span leaf=""><think></span>`标签，表示其推理过程。如果这个过程过长，用户可能会感到冗余，甚至影响使用体验。

那么， **能否去掉DeepSeek-R1的思考过程呢** ？

答案是 **肯定的** 。但与此同时，我们也要注意到：

* **去除思考过程可能会影响回答质量** ，因为模型的思考步骤有助于提高推理的准确性。

 **本文主要将介绍一种去除DeepSeek-R1的思考过程，而不考虑去掉思考过程后的回答质量** 。

## 背景知识：从补全模型到对话模型

在深度学习的发展历程中，文本生成任务经历了从“ **补全** ”到“ **对话** ”的演进。

早期的GPT模型（如GPT-2）主要基于 **文本补全** ，即根据已有的输入预测下一个最可能的单词或句子。
而ChatGPT等对话模型（如GPT-3.5、GPT-4）在此基础上引入了 **消息结构（messages）** ，允许多轮交互，并优化了对话的连贯性，更符合人类的表述方式。

DeepSeek-R1是一款强大的对话模型，它采用类似ChatGPT的架构，同时引入了 **思考过程（thinking process）** ，即在生成最终答案之前，模型会先进行推理，并将推理步骤以 `<span leaf=""><think></span>`标签的形式输出。

## 方法1：使用 Chat Prefix Completion

DeepSeek提供了Chat Prefix Completion (Beta) ^[1]^ 功能，它允许我们通过特定的提示方式控制模型的输出格式。

可以通过以下方式去掉 `<span leaf=""><think></span>`标签，使模型直接给出答案：

```
curl https://api.deepseek.com/beta \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{
     "model": "deepseek-ai/DeepSeek-R1",
     "messages": [{"role": "user", "content": "9.11和9.8哪个大"}, {"role": "assistant", "content": "<think>\n</think>\n\n"}],
     "temperature": 0.6
   }'
```

在这里，我们在 `<span leaf="">messages</span>`参数中直接告诉模型， **思考部分为空** （`<span leaf=""><think>\n</think>\n\n</span>`），这样它就不会输出思考过程，而是直接给出答案。

注意，此方法仅适用于 DeepSeek 的官方 API。

## 方法2：使用 Completion API

除了Chat模式，OpenAI 的接口规范也支持补全模式（completion）。在补全模式下，我们可以根据不同模型的对话模板（chat template）直接调整 `<span leaf="">prompt</span>`，让模型忽略 `<span leaf=""><think></span>`部分。

```
curl https://api.siliconflow.cn/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "prompt": "<|begin▁of▁sentence|><|User|>1+2+3+..+100等于多少<|Assistant|><think>\n</think>\n\n",
    "max_tokens": 7,
    "temperature": 0.6
  }'
```

或者使用Python API：

```
from openai import OpenAI
client = OpenAI(base_url="https://api.deepseek.com/beta", api_key=<API_KEY>)

client.completions.create(
  model="deepseek-ai/DeepSeek-R1",
  prompt="<|begin▁of▁sentence|><|User|>1+2+3+..+100等于多少<|Assistant|><think>\n</think>\n\n",
  max_tokens=7,
  temperature=0.6
)
```

通过这种方式，我们可以控制模型忽略 `<span leaf=""><think></span>`标签，直接生成答案。

## 可能存在的问题

尽管可以去除DeepSeek-R1的思考过程，但仍然有一些问题值得注意：

1. **稳定性** ：不同的输入可能会导致模型仍然输出思考过程，因此需要进一步测试上述方法的稳定性。
2. **回答质量** ：去除思考过程可能会影响回答的准确性，特别是在推理和计算任务中。

## 结论

 **去除DeepSeek-R1的思考过程是可行的，但需要权衡使用体验与回答质量之间的关系** 。如果你的应用场景对思考过程的可读性要求不高，而更倾向于直接获取答案，那么可以尝试上述方法来优化模型的输出。

同时，希望这篇文章能帮助你更好地理解 **补全（Completion）与对话（Chat Completion）** 之间的区别！
