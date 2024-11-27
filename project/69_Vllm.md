# 自然语言处理:第六十九章 大模型推理框架神器 - VLLM部署篇

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

官方文档: [Welcome to vLLM! — vLLM](https://docs.vllm.ai/en/latest/)

项目地址: [vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

VLLM和TGI一样也是大模型部署应用非常广泛的一个库，下面我以[蓝耘](https://cloud.lanyun.net/#/k8sStarter)平台为例，教学一次Vllm的使用，大家可以选择相似的云平台作为使用。

* 系统: Linux
* python: 3.8 - 3.12
* GPU: Nvidia - 4090
* Cuda: 12.1

<br>

### 1. VLLM安装

1. 用实例，这里我选择了个CUDA12.1.1 + Ubuntu22.04的系统，进去可以nvcc -V查看下cuda版本是否一致

   ![1732083807583](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732083807583.png)
2. 使用pip方法安装vLLM，记得配置下镜像源

   ```
   # (Recommended) Create a new conda environment.
   conda create -n myenv python=3.10 -y
   conda activate myenv

   # Install vLLM with CUDA 12.1.
   pip install vllm
   ```

   另外，如果你使用的也是蓝耘云，利用conda切换环境的时候会可能会遇到conda init 错误。蓝耘里conda init 有点问题，在.bashrc里把下面这一段配置文件加进去，然后再 `source ~/.bashrc` 就可以配置环境了

```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

如果安装完成的话 `pip list`查看下几个关键包是不是都装好了，这里装的0.6.4

![1732086079981](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732086079981.png)

3. 国内的话设置下modelscope，国外的话默认从huggingface下载，可以忽略:

   ```
   echo 'export VLLM_USE_MODELSCOPE=True' >> ~/.bashrc
   source ~/.bashrc
   pip install modelscope
   ```
4. 运行下面python脚本，测试下VLLm是否安装成功，

   ```
   from vllm import LLM
   prompts = [
       "Hello, my name is",
       "The president of the United States is",
       "The capital of France is",
       "The future of AI is",
   ]

   llm = LLM(model="Qwen/Qwen2-0.5B",trust_remote_code=True) 

   outputs = llm.generate(prompts)
   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   ```

   输出如下：

   ![1732086677767](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732086677767.png)

<br>

<br>

### 2. 启动VLLM服务

启动vLLM服务，以Qwen2-0.5B为例，其中chat-template 输入的是[template_chatml.jinja](https://github.com/vllm-project/vllm/blob/main/examples/template_chatml.jinja)是聊天模板，也可以不设置vLLM会调用默认的聊天模板，在vllm官方库中，可自行下载进行覆盖:

`vllm serve Qwen/Qwen2-0.5B-Instruct --chat-template ./examples/template_chatml.jinja --served-model-name Qwen --trust-remote-code --tensor-parallel-size 1`

* model: 模型路径，如果不是本地的话默认会从hf 或者modelscope下载
* chat-template: 聊天模板
* served-model-name: 服务器名称，后期访问的时候可以通过这个名称访问
* tensor-parallel-size: 几张卡放置模型
* trust-remote-code: 默认使用transforemer的远程模型代码

服务启动完成后会自动计算指标等，通过默认的8000端口即可访问，正常启动后如下图

![1732088637897](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732088637897.png)

<br>

<br>

### 3. 在线服务调用

1. 通过curl 调用，temperature/top_p/repetition_penalty/max_tokens 都是大模型参数。

```
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2-0.5B",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

输出结果如下

![1732090525725](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732090525725.png)

2. 通过Python 调用， 同样的是使用openai接口进行访问，运行如下脚本即可
   ```
   from openai import OpenAI
   # Set OpenAI's API key and API base to use vLLM's API server.
   openai_api_key = "EMPTY"
   openai_api_base = "http://localhost:8000/v1"

   client = OpenAI(
       api_key=openai_api_key,
       base_url=openai_api_base,
   )

   chat_response = client.chat.completions.create(
       model="Qwen/Qwen2-0.5B",
       messages=[
           {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
           {"role": "user", "content": "Tell me something about large language models."},
       ],
       temperature=0.7,
       top_p=0.8,
       max_tokens=512,
       extra_body={
           "repetition_penalty": 1.05,
       },
   )
   print("Chat response:", chat_response)
   ```

输出结果如下

![1732090525725](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732090525725.png)

<br>

最后 `nvidia-smi`看看显存占用

![1732090565380](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V0/1732090565380.png)

<br>

<br>

### 4. 压力测试

VLLM的压力测试代码只需要将 `get_tgi_response`函数替换成下面的 `get_vllm_response`函数修改即可，核心注意下url & 以及data里的stream 设置成True对应如下：

```
def get_vllm_response(query, context=None):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "EMPTY"
    }
    data = {
    "model": "Qwen",
    "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": query}
  ],
    "stream": True, # 这里设置成流式输出
    "max_tokens": 16,  #最大生产的token数量
}

    time_st = int(time.time() * 1000) # 请求开始时间
    response = requests.post(url, headers=headers, json=data, stream=True) 
    event_data = {} # 保存事件
    first_token_cost = None # 保存首字符时间

    if response.status_code == 200: 
        if response.headers.get('content-type') == 'text/event-stream; charset=utf-8': # 判断是否为流式响应
            for chunk in response:
                chunk = chunk.decode('utf-8',errors='ignore').strip() # 解析数据
                if first_token_cost is None: # 如果还没有记录首字符时间
                    first_token_cost = int(time.time() * 1000) - time_st  # 计算首包延迟，TTFT
        else:
            event_data = response.json() # 不是stream返回，直接解析json数据
 
    event_data['query'] = query #存储query
    event_data['first_token_cost'] = first_token_cost # 记录首字符的消耗
    if event_data.get('token'):
        event_data.pop('token') # 如果存在token数据，则移除
    return event_data
```

<br>

<br>

在VLLM推理框架下，Qwen2-0.5B-Instruct / Qwen2-1.5B-Instruct / Qwen2-7B-Instruct 的三者性能对比如下:

![1732460014532](https://file+.vscode-resource.vscode-cdn.net/f%3A/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2/%E6%B7%B1%E8%93%9D%E5%AD%A6%E9%99%A2_finetuning/project/image/project2_V2/1732460014532.png)

<br>

<br>

---

参考文档:

[文本生成推理 - Hugging Face 中文 (hugging-face.cn)](https://hugging-face.cn/docs/text-generation-inference/index)

[Qwen2.5: 基础模型大派对！ | Qwen (qwenlm.github.io)](https://qwenlm.github.io/zh/blog/qwen2.5/)

[TGI - Qwen](https://qwen.readthedocs.io/zh-cn/latest/deployment/tgi.html)
