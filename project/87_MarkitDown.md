# 自然语言处理:第八十七章 微软开源MarkitDown，RAG文档解析就这么解决了~

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


RAG有这么一个说法：“垃圾进，垃圾出”，文档解析与处理以获取高质量数据至关重要。近期，微软开源了 **MarkItDown** ，一款将各种文件转换为 Markdown 的实用程序（用于索引、文本分析等）。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/AE74ia62XricEVJvC2aFXEJR28VINY2yAiaX6sicUfmFViaGMICwtXE8ibNADoBzmlORqPxmVBK4hxcrWiaw2Onu59ddQ/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*https://x.com/shao__meng/status/1867348058662744236*

 **MarkItDown支持** ：

* PDF
* PPT
* Word
* Excel
* 图像（EXIF 元数据和 OCR）
* 音频（EXIF 元数据和语音转录）
* HTML
* 基于文本的格式（CSV、JSON、XML）
* ZIP 文件

**MarkItDown使用**

使用 pip: pip install markitdown。或者，从源代码安装它：pip install -e .

Python中的基本用法：

```
from markitdown import MarkItDown 
md = MarkItDown() result = md.convert("test.xlsx") 
print(result.text_content)
```

要使用大型语言模型进行图像描述，请提供llm_client和llm_model：

```
from markitdown import MarkItDown 
from openai import OpenAI client = OpenAI() 
md = MarkItDown(llm_client=client, llm_model="gpt-4o") 
result = md.convert("example.jpg") print(result.text_content)
```

**MarkItDown试用**

*https://www.html.zone/markitdown/*

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AE74ia62XricEVJvC2aFXEJR28VINY2yAiac5dqxGN9PN95fbsVjWhYGv0QmV7DAVAwCUEwZvU7eo3ur6Wl82Ujgg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
https://github.com/microsoft/markitdown
```
