
# 自然语言处理:第八十八章 如何生成Function Calling微调数据？

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


> ❝在大型语言模型（LLM）的发展过程中，"function calling"成为提升模型实际应用能力的重要研究方向之一。随着AI技术的进步，许多应用场景要求模型能够自动调用不同的API来执行任务。这不仅包括正确选择合适的API，还需要生成符合规范的函数调用，从而实现复杂的任务自动化。因此，理解如何生成高质量的function calling微调数据成为了AI开发者和研究者们的关键任务。

本文将深入探讨如何为function calling任务生成高质量的微调数据，并通过具体例子和实战步骤带领读者逐步理解这一技术核心。


## 大型语言模型在函数调用中的关键能力

在function calling微调过程中，模型需要掌握以下关键能力：

1. **准确的****API****选择** ：识别并选择最合适的API执行用户的查询。这要求模型具备强大的理解能力，能够分析用户的自然语言意图，并从多个可选的API中匹配到最相关的一个或多个。
2. **精确的参数映射** ：将用户的自然语言输入映射到所选API的参数中。这一过程要求模型能够处理复杂的嵌套结构、依赖关系等复杂的API调用场景。
3. **遵循****API****格式** ：生成的函数调用必须严格遵循API的预定义格式和语法要求，确保函数调用可以被正确执行。
4. **推理和一致性** ：在复杂的决策过程中，模型需要能够推理，评估用户的意图和工具的要求，并确保参数选择的准确性和一致性。

接下来，我们将分步骤讲解如何通过数据生成和微调来提升模型在这些方面的表现。

### 如何提升API选择的准确度？

#### 分析用户意图

模型必须具备准确解析用户自然语言的能力，并从中提炼出操作意图。例如，用户可能会说：“帮我查找附近的餐馆。”模型需要知道这与某个地理位置查询API相关，并可能需要调用一个提供附近餐馆的API。

**示例：** 用户输入：

```
用户：我想订一个离我最近的意大利餐厅。
```

模型解析意图，确定需要调用一个“餐厅搜索API”，其中包含餐厅类型、位置等参数。

### 提供合适的API选项

一旦意图被解析出来，模型需要从多样的API列表中选择最合适的。比如，如果有多个API可以执行相似的任务（如Google Maps API、Yelp API等），模型需要能够根据用户的特定要求选择合适的API。

![图片](https://mmbiz.qpic.cn/mmbiz_png/Ea6oETrjsv84CJx8Y3pR2DabFnaZdO0QOYf7Q77WBYicmTm2YdBiaHXExgXY75UK7rQCeegc1vZ44KPciadVXKsLg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
img

**示例：**

```
API1: Google Maps API - 提供餐厅、位置服务
API2: Yelp API - 提供餐厅推荐和评价
```

此时模型需要判断用户是否需要餐厅的推荐（评价）还是仅需位置信息，进而选择合适的API。

使用合成数据提升API选择

为了提升模型对API选择的敏感度，可以通过生成合成数据集。合成数据集通过创建不同的用户场景来训练模型选择合适的API。比如，生成一些不同问题场景的模拟数据，并让模型通过微调来学会正确的API选择。

**例子：**

* 场景1：用户询问天气 → 调用天气API
* 场景2：用户询问路线 → 调用地图API

通过这些合成场景，模型逐渐学习并优化API选择的策略。

### 如何实现精确的参数映射？

#### 简单参数映射

在简单场景下，用户的输入会直接映射到API的参数。例如，用户想搜索一家餐厅，输入中包含餐厅类型和位置信息。模型需要将这些信息直接映射到API的参数中。

**示例：**

```
用户：搜索一下北京的咖啡馆。
映射后的API调用：searchRestaurants(location="北京", type="咖啡馆")
```

#### 复杂嵌套结构的处理

有时，API调用需要处理嵌套参数或复杂的数据结构。例如，用户可能会在查询中包含多个条件，如餐厅类型、价格范围、评价等。模型需要能够将这些复杂条件映射到API的嵌套参数中。

**示例：**

```
用户：找一家北京的意大利餐馆，评分至少4星，价格适中。
映射后的API调用：searchRestaurants(location="北京", type="意大利餐馆", rating>=4, price_range="适中")
```

#### 使用合成API提升参数映射能力

通过创建一些复杂的合成API，并使用合成数据进行训练，可以帮助模型更好地处理嵌套结构和复杂参数。

**示例：**

* 合成API：search(location, type, rating, price_range) 合成数据：场景1：用户查询餐厅并提供价格范围和评价 场景2：用户查询电影，并提供导演和上映年份

通过多种合成场景，模型能够学习如何处理不同类型的参数映射。

### 如何保证生成数据与Schema的一致性？

#### 定义严格的Schema

![图片](https://mmbiz.qpic.cn/mmbiz_png/Ea6oETrjsv84CJx8Y3pR2DabFnaZdO0QarCsLcEGSvRl2kBQicCD2WiaNibYSGBfHnibbNGUtBxu3RbSxlaych2aWQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
img

为了确保生成的函数调用符合API的预定义格式，需要定义一个严格的Schema。Schema规定了每个API的参数类型、结构以及可能的取值范围。微调过程中，模型需要学会在调用API时严格遵循这些Schema。

**示例：** Schema：

```
{
  "location": "string",
  "type": "string",
  "rating": "float",
  "price_range": "string"
}
```

生成的数据必须符合这个Schema，例如：

```
searchRestaurants(location="北京", type="意大利餐馆", rating=4.5, price_range="适中")
```

#### 使用Schema验证生成数据

![图片](https://mmbiz.qpic.cn/mmbiz_png/Ea6oETrjsv84CJx8Y3pR2DabFnaZdO0QbkdoSfRTia6zROnzujySBXeSmickPl97Ak1uplaovF6Vwp1SiaUH3hNWg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
img

在训练过程中，所有生成的API调用都需要经过Schema验证，以确保格式的一致性。如果生成的数据不符合Schema要求，则进行修正并反馈给模型。这种反馈机制可以帮助模型逐渐学习如何生成符合格式的数据。

### 如何确保用户对话与函数调用的一致性？

#### 用户意图和API调用的匹配

模型在生成函数调用时，必须确保API调用准确反映用户的意图。为此，可以在训练过程中引入对话数据，并要求模型不仅生成API调用，还要确保API调用能够正确响应用户的需求。

**示例：**

```
用户：我想查找最近的电影院。
映射后的API调用：searchTheaters(location="当前", distance="最近")
```

#### 使用对话数据进行微调

对话数据是训练function calling模型的重要资源。通过真实的用户对话场景进行训练，可以帮助模型更好地理解用户意图，并生成与之匹配的API调用。

### 合成API生成与场景设计的几点启发

1. 合成API的优势

Function calling微调的目标是提升模型的function calling能力，而不是去拟合现有的API。因此，使用合成API进行训练是一种有效的方法。合成API允许开发者根据特定的应用场景设计各种API接口，并在这些接口上进行模型训练。

2. 如何设计合成API？

合成API的设计可以基于特定领域的文本文档，提取不同类型的问题场景。然后，基于这些场景，模型可以思考应对这些场景的API调用形式。

3. 提升API复杂度

通过调整API的属性字段类型、嵌套关系等，可以进一步提升合成API的复杂度。这能够帮助模型更好地处理复杂的API调用场景。

4. 生成对话数据

在function calling微调过程中，生成多样化的对话数据至关重要。一个有效的方法是让模型在给定场景下扮演提出问题的用户，然后通过交互生成多样化的对话数据。这可以最大程度保证对话数据的多样性，并帮助模型在更多的场景下进行function calling的训练。

## 结论

通过以上思路，开发者可以通过合成API和复杂场景的设计，逐步提升大型语言模型的API选择、参数映射、格式一致性和对话匹配能力。这些微调方法能够帮助模型在实际应用场景中表现得更加智能和高效。
