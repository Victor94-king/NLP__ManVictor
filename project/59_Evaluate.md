# 自然语言处理:第五十九章 LLM评估终极指南

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

<br />

## **1. 概述**

在当今快速发展的自然语言处理领域，大型语言模型（LLM）正发挥着越来越重要的作用。从自动翻译到文本生成，这些模型在许多应用场景中表现出了惊人的能力。然而，要确保这些模型能够在实际应用中表现稳定且高效，必须对其进行严谨的评估。这篇文章将详细探讨LLM评估指标的定义、方法和最佳实践，并提供相应的代码示例，帮助您构建强大的LLM评估流程。

<br />

<br />

## **2. LLM评估指标的定义与重要性**

***1) 什么是LLM评估指标？***

LLM评估指标是用来量化和评估大型语言模型（LLM）输出质量的工具。它们根据一系列标准来评分模型的输出，确保模型在各种任务中表现良好。常见的评估指标包括答案相关性、正确性、幻觉检测、上下文相关性、责任指标以及任务特定指标。这些指标对于优化LLM的性能、识别潜在问题以及提升模型的实用性至关重要。

***2) LLM评估的挑战***

尽管LLM评估对于优化模型至关重要，但这一过程常常面临挑战。首先，LLM的输出可能具有高度的多样性和不确定性，使得评估变得复杂。此外，如何选择和实施合适的评估指标，以准确反映模型的实际性能，也是一大难题。因此，开发一个全面且有效的LLM评估框架是构建成功LLM应用的关键。

<br />

<br />

<br />

## **3. LLM评估指标的主要类别**

![图片](https://mmbiz.qpic.cn/mmbiz_png/VtGaZVqpDn2gVY2bFoFfHmwacUbw0uFQyeGBfTSQzCvPyUVrylUvicr1QbKtRg61XI41js3Ss5B8EqZxwFNIHeA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

***1) 答案相关性***

答案相关性是评估LLM是否能够有效回答用户问题的关键指标。它衡量模型输出是否能够准确、全面地回应输入信息。例如，在问答系统中，答案相关性指标会评估模型是否提供了有用的、与问题相关的回答。

```
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_relevance_score(predicted_answer, reference_answer, model):
    pred_embedding = model.encode(predicted_answer)
    ref_embedding = model.encode(reference_answer)
    score = cosine_similarity([pred_embedding], [ref_embedding])[0][0]
    return score

# 示例
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
predicted_answer = "The capital of France is Paris."
reference_answer = "Paris is the capital city of France."
relevance_score = compute_relevance_score(predicted_answer, reference_answer, model)
print(f"Relevance Score: {relevance_score}")
```

***2) 正确性***

正确性指标用于验证LLM的输出是否符合事实。它可以通过比对模型生成的内容与已知的真实信息来计算。这对于确保模型在处理事实性问题时的准确性尤其重要。

```
import requests

def check_factual_accuracy(output_text, fact_checking_api_url):
    response = requests.post(fact_checking_api_url, data={'text': output_text})
    result = response.json()
    return result['is_factual']

# 示例
fact_checking_api_url = "https://example.com/fact-check-api"
output_text = "The Earth revolves around the Sun."
is_factual = check_factual_accuracy(output_text, fact_checking_api_url)
print(f"Is Factual: {is_factual}")
```

***3) 幻觉检测***

幻觉指的是LLM生成虚假或不准确的信息。检测幻觉对于提高模型的可靠性和用户信任至关重要。幻觉检测可以通过人工审核或结合自动化工具来实现。

```
def detect_hallucination(output_text, known_facts):
    hallucinations = [fact for fact in known_facts if fact not in output_text]
    return len(hallucinations) > 0

# 示例
known_facts = ["The capital of France is Paris.", "Water boils at 100 degrees Celsius."]
output_text = "The capital of France is Berlin."
has_hallucination = detect_hallucination(output_text, known_facts)
print(f"Has Hallucination: {has_hallucination}")
```

***4) 上下文相关性***

在基于检索增强生成（RAG）的系统中，上下文相关性指标评估模型是否能够有效利用检索到的相关信息生成回答。这种指标确保模型在生成文本时考虑了合适的背景信息。

```
def evaluate_context_relevance(retrieved_context, generated_output, similarity_model):
    context_embedding = similarity_model.encode(retrieved_context)
    output_embedding = similarity_model.encode(generated_output)
    similarity_score = cosine_similarity([context_embedding], [output_embedding])[0][0]
    return similarity_score

# 示例
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
retrieved_context = "The Eiffel Tower is located in Paris, France."
generated_output = "The Eiffel Tower is a famous landmark in Paris."
context_relevance_score = evaluate_context_relevance(retrieved_context, generated_output, similarity_model)
print(f"Context Relevance Score: {context_relevance_score}")
```

***5) 责任指标***

责任指标评估LLM的输出是否包含偏见、毒性或其他可能的有害内容。这些指标对确保模型输出符合伦理标准和社会期望至关重要。

```
def check_toxicity(output_text, toxicity_api_url):
    response = requests.post(toxicity_api_url, data={'text': output_text})
    result = response.json()
    return result['toxicity_score']

# 示例
toxicity_api_url = "https://example.com/toxicity-api"
output_text = "This is a hate speech example."
toxicity_score = check_toxicity(output_text, toxicity_api_url)
print(f"Toxicity Score: {toxicity_score}")
```

***6) 任务特定指标***

任务特定指标根据具体的应用场景和需求定制，如在文本摘要任务中评估摘要的全面性和一致性。这些指标通常需要根据任务特点进行设计和调整。

```
def evaluate_summary_completeness(summary, original_text):
    completeness_score = len(summary) / len(original_text)  # 简单比例衡量
    return completeness_score

# 示例
original_text = "In-depth information about the history of France and its landmarks."
summary = "A brief history of France."
completeness_score = evaluate_summary_completeness(summary, original_text)
print(f"Summary Completeness Score: {completeness_score}")
```

<br />

<br />

<br />

## **4. LLM评估方法的比较**

![图片](https://mmbiz.qpic.cn/mmbiz_png/VtGaZVqpDn2gVY2bFoFfHmwacUbw0uFQMcAnIXyHY8mWUHqMTHzDOHVH6wRIlt4PQIC0FvSCQd6sL3GU47kRibQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

***1) 统计评分器***

统计评分器基于统计方法对文本进行评估，包括以下几种常见类型：

* BLEU评分器：评估翻译质量，基于n-gram的匹配精度计算得分。
* ROUGE评分器：评估摘要质量，基于n-gram的召回率计算得分。
* METEOR评分器：综合精确度和召回率，考虑词汇变体和顺序差异。
* Levenshtein距离：计算字符串之间的编辑距离，适用于拼写纠正和文本对齐任务。

【优缺点分析】

统计评分器具有计算简单、效率高的优点，但在处理语义复杂的LLM输出时可能不够准确。这是因为统计评分器主要基于词汇和句法匹配，而忽视了语义信息。

***2) 基于模型的评分器***

基于模型的评分器依赖于自然语言处理模型进行评估，如：

* NLI评分器：使用自然语言推理模型评估逻辑一致性。
* BLEURT评分器：使用预训练模型（如BERT）进行评分。

【优缺点分析】

基于模型的评分器相较于统计评分器在语义理解方面更具优势，但由于其概率性质，评分结果可能不够稳定。此外，这些评分器的准确性也依赖于模型的训练数据和质量。

***3) LLM评审员***

LLM作为评审员是使用LLM本身来评估其输出的一种方法，例如G-Eval。G-Eval通过生成评估步骤和计算分数来提供高对齐度的评估结果。

![图片](https://mmbiz.qpic.cn/mmbiz_png/VtGaZVqpDn2gVY2bFoFfHmwacUbw0uFQa1kHiaibicdibU6PpLEib8c54eZyo4CfiasibTw0ApA7jFicRGDgGOKtQgJO8w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

【优缺点分析】

LLM评审员可以充分考虑LLM输出的完整语义，提高评估的准确性。但由于评估过程的主观性和模型的随机性，评估结果可能不稳定。此外，LLM评审员需要较高的计算资源和模型接口支持。

<br />

<br />

## **5. 如何实施和确定最佳LLM评估指标集**

***1) 确定评估目标***

在选择LLM评估指标时，首先需要明确评估目标。这包括识别模型在特定应用场景中的关键表现指标，如准确性、相关性或上下文相关性。根据具体的应用需求，选择适合的评估指标。

***2) 选择合适的评估方法***

选择评估方法时，考虑以下因素：

* **任务性质** ：根据任务的要求选择统计、基于模型或LLM评审员的方法。
* **评估准确性** ：优先考虑能够准确反映模型性能的评估方法。
* **计算资源** ：评估方法的选择应考虑计算资源的可用性和成本。

***3) 实施评估流程***

实施LLM评估流程包括以下步骤：

* **定义评估标准** ：根据任务要求和目标确定评估标准。
* **选择评估工具** ：根据评估标准选择适当的评分工具和方法。
* **执行评估** ：使用选定的工具对LLM输出进行评估，并记录结果。
* **分析结果** ：对评估结果进行分析，以识别模型的优点和不足之处。
* **迭代改进** ：根据评估结果调整和改进模型，重复评估过程以确保持续优化。

<br />

<br />

<br />

## **6. RAG指标与微调指标**

### ***1) RAG指标***

RAG作为一种方法，通过补充额外的上下文来增强LLM，生成定制化的输出，非常适合构建聊天机器人。它由两个组件组成——检索器和生成器。

![图片](https://mmbiz.qpic.cn/mmbiz_png/VtGaZVqpDn2gVY2bFoFfHmwacUbw0uFQDWePTic7yKDREbthI40NGdLAsYfrkk9GYianibJCUKEjU8yRaNLMRQ7hA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



【典型的RAG架构 】

以下是RAG工作流程的典型操作方式：

1. RAG系统接收输入。
2. 检索器使用此输入在您的知识库（现在大多数情况下是向量数据库）中执行向量搜索。
3. 生成器接收检索上下文和用户输入作为额外上下文，以生成定制化的输出。

这里有一点需要记住——高质量的LLM输出是出色的检索器和生成器的产物。因此，优秀的RAG指标专注于以可靠和准确的方式评估您的RAG检索器或生成器。（事实上，RAG指标最初被设计为无需参考的指标，意味着它们不需要真实情况，即使在生产环境中也适用。）

#### **# 忠实度（**  **Faithfulness** **） **

忠实度是RAG指标，评估您的RAG管道中的LLM/生成器是否生成与检索上下文中呈现的信息事实一致的LLM输出。但我们应该如何为忠实度指标选择评分器呢？

剧透警报：QAG评分器是RAG指标的最佳评分器，因为它在目标明确的评价任务中表现出色。对于忠实度，如果您将其定义为LLM输出中与检索上下文相关的真理主张的比例，可以通过以下算法使用QAG计算忠实度：

1. 使用LLM提取输出中提出的所有主张。
2. 对于每个主张，检查它是否与检索上下文中的每个单独节点一致或相矛盾。在这种情况下，QAG中的封闭式问题将是：“给定的主张是否与参考文本一致”，其中“参考文本”将是每个单独检索到的节点。（请注意，您需要将答案限制为“是”、“否”或“不知道”。“不知道”状态表示检索上下文不包含给出肯定/否定答案的相关信息的边缘情况。）
3. 累加真实主张（“是”和“不知道”）的总数，并将其除以提出的总主张数量。

这种方法通过利用LLM的高级推理能力确保准确性，同时避免LLM生成分数的不可靠性，使其成为比G-Eval更好的评分方法。

如果您觉得这太复杂而难以实现，您可以使用DeepEval。这是一个开源软件包，提供了LLM评估所需的所有评估指标，包括忠实度指标。

```
# Install
pip install deepeval
# Set OpenAI API key as env variable
export OPENAI_API_KEY="..."
```

```
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = FaithfulnessMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

DeepEval将评估视为测试用例。在这里，actual_output就是您的LLM输出。另外，由于忠实度是LLM-Eval，您能够获得最终计算分数的原因。

#### **# 答案相关性 （Answer Relevancy）**

答案相关性是RAG指标，评估您的RAG生成器输出是否简洁，可以通过确定LLM输出中与输入相关的句子比例来计算（即，将相关句子的数量除以总句子数量）。

构建健壮的答案相关性指标的关键是考虑检索上下文，因为额外的上下文可能证明看似不相关句子的相关性。以下是答案相关性指标的实现：

```
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = AnswerRelevancyMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

#### **# 上下文精度 （Contextual Precision）**

上下文精度是RAG指标，评估您的RAG管道的检索器质量。当我们谈论上下文指标时，我们主要关心检索上下文的相关性。高上下文精度分数意味着在检索上下文中相关的节点排名高于不相关的节点。这很重要，因为LLM对检索上下文中较早出现的节点中的信息给予更多权重，这会影响最终输出的质量。

```
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualPrecisionMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

#### **# 上下文召回率 （** **Contextual Recall）**

上下文精度是评估检索增强生成器（RAG）的额外指标。它通过确定预期输出或真实情况中可以归因于检索上下文中的节点的句子比例来计算。较高的分数表示检索到的信息与预期输出之间的一致性更高，表明检索器有效地获取了相关和准确的内容，以帮助生成器产生上下文适当的响应。

```
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualRecallMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

#### **# 上下文相关性 （Contextual Relevancy）**

可能是最简单的指标，上下文相关性仅仅是检索上下文中与给定输入相关的句子比例。

```
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = ContextualRelevancyMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

### ***2) 微调指标***

当提到“微调指标”时，实际上指的是评估LLM本身的指标，而不是整个系统。除了成本和性能优势之外，LLM通常被微调以纳入额外的上下文知识或调整其行为。

如果您想微调自己的模型，这里有一个如何在不到2小时内在Google Colab内微调LLaMA-2的分步教程，包括评估。

#### **# 幻觉 （Hallucination）**

您可能认识到这与忠实度指标相同。虽然类似，但微调中的幻觉更为复杂，因为通常很难确定给定输出的确切真实情况。为了解决这个问题，我们可以利用SelfCheckGPT的零次拍摄方法来抽样LLM输出中幻觉句子的比例。

```
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Note that 'context' is not the same as 'retrieval_context'.
  # While retrieval context is more concerned with RAG pipelines,
  # context is the ideal retrieval results for a given input,
  # and typically resides in the dataset used to fine-tune your LLM
  context=["..."],
)
metric = HallucinationMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.is_successful())
```

#### **# 毒性 （Toxicity）**

毒性指标评估文本包含攻击性、有害或不适当语言的程度。可以使用现成的预训练模型，如使用BERT评分器的Detoxify，来评估毒性。

```
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase

metric = ToxicityMetric(threshold=0.5)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)

metric.measure(test_case)
print(metric.score)
```

然而，这种方法可能不准确，因为“如果评论中存在与咒骂、侮辱或亵渎相关的词汇，无论作者的语气或意图如何，例如幽默/自嘲，都可能被归类为有毒”。

#### **# 偏见 （Bias）**

偏见指标评估文本内容中的政治、性别和社会偏见等方面。这对于涉及定制LLM参与决策过程的应用尤其重要。例如，协助银行贷款批准提供无偏见的建议，或在招聘中，它协助确定候选人是否应该被列入面试名单。

与毒性类似，偏见可以使用G-Eval进行评估。（但不要误会，QAG也可以是毒性和偏见等指标的可行评分器。）

```
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)
toxicity_metric = GEval(
    name="Bias",
    criteria="Bias - determine if the actual output contains any racial, gender, or political bias.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

metric.measure(test_case)
print(metric.score)
```

偏见是一个非常主观的问题，在不同的地理、地缘政治和地缘社会环境中差异显著。例如，在一个文化中被认为是中性的语言或表达方式在另一个文化中可能带有不同的含义。（这也是为什么少次评估对偏见效果不佳的原因。）

一个可能的解决方案是为评估微调定制LLM或提供非常清晰的指导方针进行上下文学习，因此我认为偏见是所有指标中最难实施的。

<br />

<br />

## ****7. 特定用例指标****

总之，所有优秀的摘要：

1. 与原文事实一致。
2. 包含来自原文的重要信息。

使用QAG，我们可以计算事实一致性和包含分数，以计算最终摘要分数。在DeepEval中，我们将两个中间分数的最小值作为最终摘要分数。

```
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

# This is the original text to be summarized
input = """
The 'inclusion score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher inclusion score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
"""

# This is the summary, replace this with the actual output from your LLM application
actual_output="""
The inclusion score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
"""

test_case = LLMTestCase(input=input, actual_output=actual_output)
metric = SummarizationMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
```

## **8. 结语**

LLM评估指标的主要目标是量化您的LLM（应用程序）的性能，为了实现这一目标，我们有多种评分器可供选择，其中有些比其他的更准确。在LLM评估中，使用LLM的评分器（如G-Eval、Prometheus、SelfCheckGPT和QAG）由于其高推理能力而最为准确，但我们需要额外注意确保这些评分的可靠性。

归根结底，指标的选择取决于您的用例和LLM应用程序的实现情况，RAG和微调指标是评估LLM输出的良好起点。对于更具体的用例指标，您可以进一步深究，以获得最准确的结果。

**参考：**

1. https://www.confident-ai.com/blog/evaluating-llm-systems-metrics-benchmarks-and-best-practices
2. https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
3. https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
