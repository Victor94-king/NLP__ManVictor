# 自然语言处理:第一百零四章 生产环境vLLM 部署 DeepSeek，如何调优，看这里

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


### 之前有小伙伴想让我出一篇，生产环境如何部署deepseek，什么样的配置能生产可用，今天我用双4090，测试几个模型。大家看看。 非常感谢提供环境的朋友。

### vLLM 简单介绍

vLLM 是一个快速且易于使用的 LLM 推理和服务库。

vLLM（**V**ery **L**arge **L**anguage **M**odel **S**erving）是由加州大学伯克利分校团队开发的高性能、低延迟的大语言模型（LLM）推理和服务框架。它专为**大规模生产级部署**设计，尤其擅长处理超长上下文（如8k+ tokens）和高并发请求，同时显著优化显存利用率，是当前开源社区中**吞吐量最高**的LLM推理引擎之一。

* **高吞吐量** ：采用先进的服务器吞吐量技术。
* **内存管理** ：通过PagedAttention高效管理注意力键和值内存。
* **请求批处理** ：支持连续批处理传入请求。
* **模型执行** ：利用CUDA/HIP图实现快速模型执行。
* **量化技术** ：支持GPTQ、AWQ、INT4、INT8和FP8量化。
* **优化内核** ：包括与FlashAttention和FlashInfer的集成。
* **其他特性** ：支持推测解码、分块预填充。

vLLM 文档：https://docs.vllm.ai/en/latest/index.html

源码地址：https://github.com/vllm-project/vllm

性能测试：https://blog.vllm.ai/2024/09/05/perf-update.html

## 软硬件信息

我的环境是双4090，内存是192GB。我预想的8k上下文。最好16k。![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgmx9ibj2TsrZhfHJgp79rCs0clNprrFRkHW9wLjbL7slvVriaIXy7eXkQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### CUDA信息

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgwZHxzqjQWFfDXxxFAR5UEgo0M3dNNqgVwF7yoVYKISp8jzCgXMUSCQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Driver Version: 550.144.03
CUDA Version: 12.4

## 环境安装

```
#下载 miniconda3
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh

# 安装
sh Miniconda3-latest-Linux-x86_64.sh

# 添加国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/


#创建vllm环境
conda create -n vllm python=3.12 -y
#激活环境,注意，切换窗口一定要执行该命令
conda activate vllm
#设置国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
#安装vllm和 modelscope
pip  install vllm modelscope

#根据自己的cuda版本安装，具体适配哪个，可以问下kimi或deepseek，使用下面的方法比较慢，不推荐
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 设置镜像（推荐）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/

conda install pytorch torchvision torchaudio pytorch-cuda=12.4
#安装基准测试依赖包
pip install pandas datasets transformers
```

### 监控工具

```
pip install nvitop
# 查看
nvitop
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dg0PDCVgNFxSiawvB17lfnNyb0IrTiasOEgQxGicTndgfZ5qcQlh0IfHq8g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 模型选择

我以8k上下文为要求，让deepseek给我推荐测试模型的大小。

| 精度            | 推荐模型 | 单卡显存占用 | 剩余显存 | 适用场景                |
| --------------- | -------- | ------------ | -------- | ----------------------- |
| **16bit** | 7B-13B   | 13-19GB      | 4-10GB   | 高精度任务（代码/数学） |
| **8bit**  | 20B-30B  | 15-23GB      | 0.6-8GB  | 通用任务+大上下文       |
| **4bit**  | 40B-50B  | 12-21GB      | 3-11GB   | 低延迟+超大模型         |

### 基准测试

参考： https://modelscope.cn/docs/models/download

```
# 创建模型目录
mkdir -p /opt/models

# 官方基准测试代码
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks

```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgnsfnmOW7tcSFk2D1uk66MtbVVs9L6Vx7NbjWUUIs6npJjefuf2EKOA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

基准测试指标含义

| 指标                                | 含义                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Avg prompt throughput**     | **输入吞吐量** （Prompt Tokens/s），0.0 表示当前没有新的输入请求                             |
| **Avg generation throughput** | **生成吞吐量** （Generation Tokens/s），86.8 表示模型每秒生成**86.8 个 token**         |
| **Running**                   | **正在处理的请求数** （当前正在生成的请求）                                                  |
| **Swapped**                   | **被换出的请求数** （当显存不足时，某些请求会被移到 CPU）                                    |
| **Pending**                   | **等待中的请求数** （尚未处理的请求）                                                        |
| **GPU KV cache usage**        | **GPU KV Cache 使用率** ，表示当前 GPU 的 key-value cache 使用情况，数值越高表示显存消耗越多 |

#### DeepSeek-R1-Distill-Qwen-14B

```
modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' --local_dir '/opt/models/DeepSeek-R1-Distill-Qwen-14B'
```

基准测试代码

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/DeepSeek-R1-Distill-Qwen-14B" \
  --backend vllm \
  --input-len 2048 \
  --output-len 10000 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16 \  -- 半精度
  --tensor-parallel-size 2 \  -- 双cpu
  --gpu-memory-utilization 0.95 \
  --max-model-len 60000   --长上下文
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgVmh9Z8ibq7VfgRINPFh6ib8ibSUqvXOIt914414Q4yjsnAiaWQ0qFJFF4Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

刚开始600多tokens/s，并行能达到20多个

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dg1eVZWqssHtcwS8cMyNPY6ibiaLkOUicXTlT1e3sgABeWPX16O98KRf9HQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)几分钟以后，降到了290tokens/s，并发降到10多个

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgCy93qicGFsXxureCicky1a35pJlFgxpL5ibMia3uJmY4HfY8HB06qXOfHw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)最后降到了200tokens 左右，并发稳定到7个。

### Qwen/QwQ-32B

```
modelscope download --model 'Qwen/QwQ-32B' --local_dir '/opt/models/QwQ-32B'
```

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/QwQ-32B" \
  --backend vllm \
  --input-len 1024 \
  --output-len 3000 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16  \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgFZOmLrichxXA0AYwgxuv24Q3MvnGSvLhZ080zxEhQrTsSA01lV7vDDQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

直接内存溢出，加载不上。

### Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ

```
modelscope download --model 'Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ' --local_dir '/opt/models/deepseek-70b-awq'
```

基准测试脚本

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/deepseek-70b-awq" \
  --backend vllm \
  --input-len 1024 \
  --output-len 4096 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16  \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 5200 
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgQmEoicaSB3Khs61ZWBKtrvmyxwXC4I02ImvNuepia8pmJDpFJ1310UWg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)压测刚开始，还好一些，能达到6~9个并发，随着时间的挪移，逐步稳定到2个并发。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dg4uVQrBbKxbIIFKl4iaLeARKQjyUdicmPmeg3L2Vc1D54MVO8gGL1Vpxg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgNE2vAGLGJSWMQnWT0wgohpRvYM4gFAGkrka31ZyiciauBsOBGAOgGnfw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

利用内存，开启长上下文

deepseek推荐的两个参数.

| 参数                       | 作用机制                                                         | 适用场景                             | 性能影响                        |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------ | ------------------------------- |
| **--cpu-offload-gb** | 将部分模型权重**静态卸载**到CPU内存，形成"虚拟显存"        | **模型参数过大**导致OOM        | 高延迟（需频繁CPU-GPU数据传输） |
| **--swap-space**     | 将KV Cache**动态交换**到CPU/磁盘，处理长上下文时的显存溢出 | **长序列生成**导致KV Cache爆炸 | 中等延迟（按需交换）            |

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/deepseek-70b-awq" \
  --backend vllm \
  --input-len 4096 \
  --output-len 10000 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16  \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --swap-space 40
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dghYagk79jWZJkQjnpSd0fn7mmFxlDxCHCoMKaElCiar5zFtTEWmZGicibA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)在启动的过程中，一直报超过了10256，我以为这个参数可以直接通过CPU增加GPU的内存。

然后在官方的ai上问了下![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgibdHk4bEjia1kMQxg6o2rjNbt9gssjc1DFUE0xbhmSg6MAJAoAI1oMmg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

看这个意思，在并发的时候，才会利用，启动的时候，并不会利用。

然后我就又问下了，该如何配置启动参数。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgicaGFXeZmZriaxy0ib8Id5cGBsS5DOiahPsQ6NkrOAsnF5icespxibQJLJ2w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)官方的api给我了一个配置参数，用上。

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/deepseek-70b-awq" \
  --backend vllm \
  --input-len 4096 \
  --output-len 10000 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16  \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --cpu-offload-gb 10
    --enforce-eager 
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dge0nEOGs3nqcYpfvZrveg19ufzN2a0R3vwUyfoxkcia2ibCb4RsW2KWGg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)好不容易看到点希望，又挂了。继续调整脚本。

```
CUDA_VISIBLE_DEVICES=0,1 python benchmark_throughput.py \
  --model "/opt/models/deepseek-70b-awq" \
  --backend vllm \
  --input-len 4096 \
  --output-len 10000 \
  --num-prompts 50 \
  --seed 1100 \
  --dtype float16  \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --cpu-offload-gb 10 \
  --enforce-eager 
```

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgSDiaFQ94SeeDC8W0rlKIa9uhUpwkciclbhZo0FTDicLQ3HSEua7SdRvDQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)这次终于起来了，这速度。。。![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/AaKTtPGkoWIm1yMcfsUWWs40V6w0z2dgQ5j6YUr6j0W2MdQrcL5NmGkbrLPafIhCt2egFE1V7uhjichQ0fS65yw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)这cpu也没有利用起来。

## 后记

使用vllm还是老老实实的使用显存吧，使用cpu，这速度真的没法忍受。也可能是我的能力有限。有哪位高手看到了，可以指教下。

善于利用官方的ai工具真能省不少力气，感觉浪费了好几个小时。

生产环境看自己的要求，尽量减少上下文长度，十几个并发，重度使用，也就20来个人可用，轻度使用，几十个人。它不像普通程序，很快就返回，每次执行都是好多秒。

通过官方的基准测试，不断地调整参数，找到适合自己的配置，没有绝对。

相关软件都放到了网盘里。有需要的可以自己下载。想加群的小伙伴，直接加我微信，yxkong。我拉你们进群。

## 相关资料

清华DeepSeek相关资料

https://pan.quark.cn/s/5c1e8f268e02https://pan.baidu.com/s/13zOEcm1lRk-ZZXukrDgvDw?pwd=22ce

北京大学DeepSeek相关资料

https://pan.quark.cn/s/918266bd423ahttps://pan.baidu.com/s/1IjddCW5gsKLAVRtcXEkVIQ?pwd=ech7

零基础使用DeepSeek

https://pan.quark.cn/s/17e07b1d7fd0

https://pan.baidu.com/s/1KitxQy9VdAGfwYI28TrX8A?pwd=vg6g

ollama的docker镜像

https://pan.baidu.com/s/13JhJAwaZlvssCXgPaV_n_A?pwd=gpfq

deepseek和qwq模型（ollama上pull下来的）

https://pan.quark.cn/s/dd3d2d5aefb2

https://pan.baidu.com/s/1FacMQSh9p1wIcKUDBEfjlw?pwd=ks7c

dify相关镜像

https://pan.baidu.com/s/1oa27LL-1B9d1qMnBl8_edg?pwd=1ish

ragflow相关资料和模型

https://pan.baidu.com/s/1bA9ZyQG75ZnBkCCenSEzcA?pwd=u5ei

公众号案例

https://pan.quark.cn/s/18fdf0b1ef2ehttps://pan.baidu.com/s/1aCSwXYpUhVdV2mfgZfdOvA?pwd=6xc2

总入口（有时候会被屏蔽）：

https://pan.quark.cn/s/05f22bd57f47提取码：HiyL

https://pan.baidu.com/s/1GK0_euyn2LtGVmcGfwQuFg?pwd=nkq7
