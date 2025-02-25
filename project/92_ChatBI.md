# 自然语言处理:第九十二章 chatBI 经验(转载)

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

原文连接: [一文分享 ChatBI 实践经验](https://mp.weixin.qq.com/s/G6I2SfdrrXV332_uPrT7kA)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


********一、前言********

随着大语言模型（LLMs）为基础的人工智能技术持续发展，它可以为商业智能（BI）领域带来哪些新的机遇？

首先，让我们简要回顾一下BI领域所经历的几个关键阶段：

① 第一阶段（报表式BI产品）： 用户提出数据需求，再由数据研发取数后通过图表组件渲染实现可视化，这是一种按需开发方式。虽然这类产品能够满足用户的基本数据查询需求，但它们开发成本较高，且静态报表的应用场景有限，缺乏灵活性，导致边际效益较低。

② 第二阶段（自助式BI产品）：在这一阶段，BI产品在宽数据集上支持用户通过拖拉拽的方式自助搭建报表。这有效减少了对数据开发的依赖，而且产品灵活性增强，能够适应多样化的数据分析场景。然而，它要求用户对指标体系有深入了解，并具备一定的配置技能，仍然具有不低的使用门槛。

③ 第三阶段（智能式BI产品）：借助大模型强大的理解、推理能力，用户只需要通过简单的交互操作（如点击按钮、进行对话）即可完成报表搭建和数据分析与洞察，这显著降低了BI产品的使用门槛，实现了让每个人都能成为数据分析师的目标。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPhloStibvwhs78TOOb42ZGe311BfwBjVGJ9ZE2nPdRzCv1IchIhwoQxg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲BI产品发展历程

二、大模型在BI领域的落地场景

经过近一年对“LLM+BI”业务场景的探索，笔者落地了三个主要的应用场景：

1、数据解读与总结

将结构化数据转化为用户易懂的文本报告，例如总结客户动态或分析员工行为轨迹。本质上是基于固定的业务规则组装数据集，然后利用大模型的归纳总结能力，按照指令要求生成总结和建议。


<video data-v-960a6ffa="" src="https://mpvideo.qpic.cn/0bc3xmaaaaaauqalc2dm7ntvbo6dac5qaaaa.f10002.mp4?dis_k=70fc35a740134c73c0846949288669cb&dis_t=1737016584&play_scene=10120&auth_info=V5CW8dIuQFVN8cq5hlNeSHI0T19Jdi1lXT4VRDtbIzlsTh46NxFgNG8aMGQbYkVWITJFBxUsJX9mSUJaNgF1bnBKVjA=&auth_key=27dd8c67600e09748409dcabb72418c6&vid=wxv_3784674984996634630&format_id=10002&support_redirect=0&mmversion=false" poster="http://mmbiz.qpic.cn/sz_mmbiz_jpg/dTlTyMSkaIzulayXtFp0ic5Lrqo6DKIiaUUWK6GgTGp5kv4SLSFGfRxU9icssibGTYw2H4hMiaTBCqCvL83C8BH2JPw/0?wx_fmt=jpeg&wxfrom=16" webkit-playsinline="isiPhoneShowPlaysinline" playsinline="isiPhoneShowPlaysinline" preload="metadata" crossorigin="anonymous" controlslist="nodownload" class=""></video>

▲数据总结演示

2、动态可视化模板

在自助报表分析的阶段，用户需要对指标和维度有一定理解后，才能通过拖拉拽的方式搭建报表，这对用户的配置能力要求很高。为了降低搭建报表的门槛，常见的方法是为不同行业的用户提供多样化的报表模板。这在一定程度上降低了报表的配置难度，但用户仍需浏览每一个模板的配置以找到合适的模板，这个过程既繁琐又耗时。

在智能化阶段，我们可以对报表模板进行标注和向量化存储。用户只需简单描述需求，系统便能通过“RAG+LLM”技术快速匹配模板并自动搭建合适的报表。

<video data-v-960a6ffa="" src="https://mpvideo.qpic.cn/0bc3fqaaaaaaoualgy3lgntvalgdaawaaaaa.f10002.mp4?dis_k=ee25e2a3c00eeaeb59041f1c6487acf2&dis_t=1737016584&play_scene=10120&auth_info=JIXNx0cWA0z0y72EWwxAIzpDUUguLWFYaxYVPQtwb2sdHzlmFzZibh8xYBlqF15wPEkJFHQle2McQQswUSY4dxlXMw==&auth_key=71426837aa2ad5a35f97ecfa6bfc9035&vid=wxv_3783585042874630156&format_id=10002&support_redirect=0&mmversion=false" poster="http://mmbiz.qpic.cn/sz_mmbiz_jpg/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPKL324B0V7NsWUfj6ibyYFlQp1Uej9Y8rfD5fO4umKv9Hdvib6vXNcLxA/0?wx_fmt=jpeg&wxfrom=16" webkit-playsinline="isiPhoneShowPlaysinline" playsinline="isiPhoneShowPlaysinline" preload="metadata" crossorigin="anonymous" controlslist="nodownload" class="video_fill"></video>

▲动态可视化模板演示

3、对话式查数（ChatBI）

ChatBI（基于聊天界面的商业智能工具），主要支持用户通过自然语言与数据进行交互，从而轻松完成业务数据查询，下方将重点围绕ChatBI进行分享。

********三、有关ChatBI的实现方案选择********

ChatBI是笔者在过去的CRM产品智能化改造中，遇到最具挑战性的项目之一，期间碰壁无数。而ChatBI最终实现的效果不仅依赖于大模型的性能，产品的实现方案同样关键。其中，行业内讨论最多的两种方案是“Text2SQL”和“Text2DSL”。

#### Text2SQL

Text2SQL（Text-to-SQL）可以简单理解为，通过大模型将自然语言问题转换为结构化查询语言（SQL），使数据库能够理解并返回数据查询结果。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPPAKpTFxaUiaJYdmNYQDjeusHHEYmnHEYDI3z71iaJff0YDOnaro0dM3Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)▲Text2SQL流程

这种方式有效利用了大模型的泛化能力，灵活性很高，极大地简化了数据分析过程。但同时也会存在不少的挑战：

* **生成SQL的准确性与可执行性**  **：生成准确且可执行的SQL查询是一项重大挑战，需要大模型深入理解SQL语法、数据库模式和特定方言** **，同时****依赖于Prompt中对表名、表字段以及各个表之间关系的清晰****描述。**
* 特定业务的复杂查询：例如跨表或跨数据集查询。对于特定业务场景的数据分析可能涉及多表关联（JOIN），这要求模型具备强大的语义理解和逻辑推理能力。
* **性能问题** ：在企业级数据查询中，宽表可能包含上百个字段，输入Prompt和输出SQL语句的复杂度会影响大模型的响应时间。超过3秒的响应时间会严重影响用户体验，导致用户流失。

Text2DSL

Text2DSL（Text-to-DSL）技术是将自然语言转换为领域特定语言（DSL）。“领域特定语言”听起来有点抽象，但可以理解为是一种更易于用户理解和使用的语言，例如在BI领域，它指的是从底层数据集中抽象出的指标、维度和过滤条件等报表配置化参数。

结合SQL这种数据库操作的标准语言，Text2DSL既简化了用户表达，又确保了系统能高效执行查询。基本流程如下：

① 用户提问时，大模型依据Prompt来理解用户的需求意图，并将自然语言需求转换为结构化的DSL查询。

② 业务方根据规则将DSL转换为SQL以执行数据查询，并将结果进行可视化展示。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPIqj1iaGYGlmyThNdHBftA7pM4usLm8ub9cV7Ndb84BdhC9qxDTvrHRQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)▲Text2DSL流程

简单举个例子：

我要找到华南区上个月业绩表现最好的3名员工

在Text2SQL的方案中，需要大模型对用户的提问进行理解后，输出一段可执行的SQL，如：

```
SELECT
    f_user_name,SUM(f_amount) AS total_amount
FROM user_performance 
WHERE f_department_id='华南大区'   /* '华南大区'是部门名称 */
      AND f_date >='2024-11-01'     
      AND f_date <='2024-11-30'   /* 时间筛选为上个月 */
GROUP BY f_user_name              /* 对员工进行分组 */
ORDER BY total_amount DESC        /* 对订单成交金额进行倒叙 */
LIMIT 3;                          /* 取前三 */
```

在Text2DSL的方案中，会对SQL进行了一层业务封装，只需要大模型识别提问后返回关键参数如：

```
时间='上月'
数据范围='华南大区'
指标='订单成交金额'
维度='员工名称'
排序='倒叙'
```

然后业务方基于大模型返回的参数（DSL），根据规则生成对应的SQL，执行查数命令。

由此可见，Text2DSL本质上是一个text to DSL to SQL的过程。简而言之，DSL是对于SQL的一层抽象，而SQL是对于数据输出的具体执行。

Text2DSL方案同样面临挑战：基于Text2DSL搭建的ChatBI需要依赖成熟的指标体系，而且查询的灵活性和扩展性受限于现有指标和维度，本质上是报表搭建参数智能检索召回后的自动化数据查询流程。

但相比于Text2SQL，Text2DSL更易于实现，并且在指标体系能够满足大多数用户需求的情况下，能提供更准确、时效性更强的结果。

适用场景

两种方案都有其适用场景和限制，选择最合适的方案需要综合考虑业务需求、BI基础能力、实现成本和用户体验。

* **Text2SQL**  **：**  **适合没有特定的复杂业务分析要求，需要高度灵活性和可扩展性的标准化数据分析场景，如：平台级BI工具** **。**
* Text2DSL：适合业务场景明确，产品已建立成熟的数据资产（例如完善的指标体系和数据服务API）且分析深度可控的情况，如：企业内部系统或垂直业务软件系统的BI工具。

四、ChatBI的实现思路分享

接下来，将以笔者负责的ChatBI项目跟大家分享一下实践经历。

1、背景分析

目前所在的SaaS CRM产品主要面向中小型企业用户，他们的BI需求更多围绕CRM产品内产生的数据进行分析，如：员工销售行为分析；客户转化分析等。

经过了近几年的打磨，我们团队已经搭建了一套完善的指标体系，覆盖7个数据域，包含400多个指标。用户可以根据业务需求，通过指标、维度、过滤条件和图表组件等配置化参数，自助搭建报表来完成数据分析。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJP3rfuBiaRe7xqXfOicCreiayfrYc9Zq8tRibqFIc7MHPAbkWZpWL2Omw8qQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲报表配置框架图

尽管自助分析能满足大多数用户的日常需求，但在实际使用中仍有一些问题：

* ******使用门槛：CRM产品面向的用户主要是销售人员，在配置报表前需要他们掌握“指标”和“维度”等较为抽象的概念，还需要理解每个指标的具体含义，这尤为困难。******
* ******数据获取链路过长 **：** **即使用户已配置了一个满足业务需求的数据看板，但在实际使用中仍需根据不同的分析场景动态调整过滤条件来获得数据，例如：报表默认配置为“本周直销部的业绩数据”，若临时需要查看“上周渠道部的业绩数据”，需要找到【时间】与【统计成员】过滤条件后，切换条件值。********

以上问题也是BI自助分析阶段普遍存在的。因此，我们计划通过ChatBI这款产品，来解决这些问题。

#### 2、Text2SQL or Text2DSL ？

正如上文所述，ChatBI的效果与所选方案密切相关。在前期调研中，我们首选了Text2SQL方案，但经过多次测试，结果未达预期。举个例子：

在当前的自助分析BI产品中，用户可以在一张报表内配置多个跨数据集的指标，例如“通话次数”和“订单成交量”，这些指标存储在不同的数据集中，并且每个指标支持多个实时过滤条件。因此，在技术方案中，我们无法创建太多的大宽表，也无法进行大量的数据预聚合。![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPudzjOcMjLu90FA49o2puSib92T4iaQUewPxaiaUAl9E8nl84f5fRSmGsg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)▲支持跨数据集搭建报表

前端报表搭建实际上是对底层数据查询的业务封装。用户只需选择适合业务需求的指标来搭建报表，而跨数据集的查询等复杂业务则由业务方统一处理。

然而，这种业务特定的场景并不适用于Text2SQL方法，因为它需要将大量元数据整合到Prompt中以支持跨数据集查询。这在与大模型交互时可能会导致表字段识别不准和返回SQL响应时间过长。

最终团队确定采用Text2DSL方案，该方案本质上是在保持现有报表搭建框架的基础上，将用户手动选择参数和拖拉拽搭建报表的过程交由大模型完成。此外，对于常规SQL难以处理的特定业务查询，Text2DSL能够通过其原生的快速计算功能轻松实现。

#### 3、方案设计

选择了Text2DSL的方案后，接下来便是Chat BI的业务方案设计。刚开始我们遇到了一些问题：

* 由于涉及到7个数据域的400多个指标和50多个维度，若全部整合到Prompt中，可能会影响大模型语义理解和意图识别的准确性。
* 用户对数据指标的叫法各异，例如“订单成交量”可能被用户称为“成交业绩”或“订单转化”等。

因此，我们将BI元数据库和业务知识库进行了向量化存储。BI元数据库涵盖了指标、维度、过滤条件、图表类型、同环比和排序等基本信息；而业务知识库则包含了行业特定术语和针对特定业务的SQL编写规则等垂直领域知识（虽然维护知识库需要定期投入资源，但其对于提高语义解析和结果生成的准确性至关重要，这也是产品近年来沉淀的重要数据资产。）

ChatBI初步的业务流程如下：

① 知识召回：当用户提问时，先通过RAG技术对知识库进行相似度检索，以召回高匹配度的知识。

② 关键信息提取：针对召回的知识动态组装Prompt后让大模型进行语义理解与关键词识别，当无法识别关键词时可通过多轮对话的方式进行交互，直至提取用户提问中的关键信息，如指标、维度、过滤条件等。然后让大模型按照Prompt中的规则返回DSL。

③ SQL转换：接收大模型返回的参数后，进行参数合法性和数据权限校验，确认无误后将DSL转换为SQL以执行数据查询。

④ 数据可视化：查询结果后，进行数据报表组装，并在前端进行可视化展示。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPicotGLceK2LiaRj1frHhckuYBRVzNFrkNzCYO4LAmbuB9o15BhPibfl3A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲ChatBI业务流程图

我们在落实这一流程时很快又遇到了新的问题。这套方案在处理用户常规的结构化数据查询提问，如“昨天入库多少个客户”或“上周打了多少通电话"时表现良好，但在应对例如“总结欧汶本周新增客户的转化情况”或“欧汶最近的业绩表现如何”等要求非结构化回答时就显得力不从心。这是因为基于结构化SQL语言生成的回答显得过于生硬，有时甚至让用户一种答非所问的感觉。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPpXt6SlVUVDk2VQK4ShSicXEg3DbfAmCFoIYX0WnVeE7jHAddBSeDFNA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲生硬的回答

因此，我们引入了意图识别机制。大模型首先通过识别意图来区分用户是需要简单的数据查询，还是针对特定业务的分析。对于后者，我们会在SQL查询得到数据后，重新组装Prompt，再次利用大模型进行语义理解分析，并根据用户的提问提供针对性的回答。

意图识别后再走一次LLM

#### 4、产品效果

以下是有关Chat BI的一个初步演示效果

 **，时长**00:51

 [ ]

<video data-v-960a6ffa="" src="https://mpvideo.qpic.cn/0bc3feaaaaaavyalgzllgntvakodaauqaaaa.f10002.mp4?dis_k=3612ec1d8db29d18aa1de60731ad9213&dis_t=1737016584&play_scene=10120&auth_info=W4X+/bYsRl5PoMPp1VxaRnM9Fg1GfX80WThCFW0Nc2pgGk07OhZmP21LOTRIbUFYIDscVRondy5iTxULYFclPXweBTE=&auth_key=47ca2f53e7332dea79c128cc88fcd1e2&vid=wxv_3783585811606339586&format_id=10002&support_redirect=0&mmversion=false" poster="http://mmbiz.qpic.cn/sz_mmbiz_jpg/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPCaejl2wXcFMHvrVgYCKpejNtSorHvObaaQYSB1Db5sKPxv1g1bDGaA/0?wx_fmt=jpeg&wxfrom=16" webkit-playsinline="isiPhoneShowPlaysinline" playsinline="isiPhoneShowPlaysinline" preload="metadata" crossorigin="anonymous" controlslist="nodownload" class=""></video>

▲ChatBI演示

ChatBI主要聚焦于解答What（是什么）的数据查询，为了帮助用户探索Why（找原因）和How（给方案）的问题，以形成完整的业务闭环，我们从应用层面提供了以下方案。

* Why：在数据可视化环节，支持用户根据需要调整维度、指标和图表组件等，以便进行全面分析，并能够根据用户的设定进行“多维分析”以及“下钻分析”。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJPRp28kiczmbDAZYB8rf0JzTE4MWLNxk6JRFfbExFM0YrrH4jhMfg4CXQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)▲帮助用户找原因

* How：通过“推荐提问”的功能，针对不同指标数据异常的场景，提供针对性的解决方案，如：设置指标预警、调整销售目标、分配销售任务等。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/dTlTyMSkaIzXaNIf20ziap5AsUdibo0cJP91yx5xYava5ydjR1MBF4YOuxNiag721yjOo5Vh5uRyWiaMRqU7MkmQrQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲给用户提供方案

团队还通过嵌入式将 ChatBI 产品整合到多端。PC端用户可以直接在输入框中输入文字；移动端用户则可以通过语音方式描述数据需求，从而快速查询数据，增强用户体验。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/dTlTyMSkaIzulayXtFp0ic5Lrqo6DKIiaUb6stBUONKg6vGjJ9WO9iaBUk9UCoDS6Q83XSQOArlqRibCc2n2Js8cdA/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲移动端语音描述需求

********上文是笔者有关ChatBI产品实践的一个简要分享，在落地过程中还需要克服种种困难，如：报表配置参数的混合检索召回；用户提问的多轮会话识别；大模型参数提炼不准的应对措施等等，此处不做细说。********

********五、****************结语********

ChatBI是今年比较火的一个话题，同时也是ToB领域落地难度较大的Agent应用。但无论采用哪种技术方式落地，最终还是需要以用户价值为导向，核心目的始终是帮助用户更便捷、更低成本去解决业务上的问题，这也是“SaaS+AI”的初衷。
