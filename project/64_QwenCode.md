# 自然语言处理:第六十四章 Qwen2代码解析

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

原文地址：[微信公众平台](https://mp.weixin.qq.com/s/PVSPNfv0I8_cxgPTmOes5w)

项目地址: [QwenLM/Qwen2.5: Qwen2.5 is the large language model series developed by Qwen team, Alibaba Cloud.](https://github.com/QwenLM/Qwen2.5)

官网地址: [你好，Qwen2 | Qwen](https://qwenlm.github.io/zh/blog/qwen2/)  & [Qwen2.5: 基础模型大派对！ | Qwen](https://qwenlm.github.io/zh/blog/qwen2.5/)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

<br />

下面的源码内容来自transformers代码库中：`transformers-4.45.2/src/transformers/models/qwen2/modeling_qwen2.py`。

## 实验准备

首先我们下载一些Qwen2需要的配置数据。下载地址：https://hf-mirror.com/Qwen/Qwen2-0.5B/tree/main

```
# 下载配置相关的文件
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/config.json
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/generation_config.json
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/merges.txt
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/tokenizer.json
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/Qwen/Qwen2-0.5B/resolve/main/vocab.json
```

> 注：权重文件我们可以不下载，我们这里仅仅是做一些流程实验，所以权重可以使用随机初始化。

下载transformers源码，我们这里使用的是 `4.45.2`版本，理论上之后的版本也都支持。

### config.json文件修改

原始文件内容：

```
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 131072,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

我们这里修改 `num_hidden_layers`值为 `4`和 `use_cache`设置为 `false`，因为我们仅仅是实验一下，并不需要那么多层。其它内容保持不变。

### 文件结构

在transformers目录的examples目录里面新建一个Qwen2_learn目录，在Qwen2_learn下再建一个config文件夹，然后将上面下载的配置文件复制到config目录下。最终或得如下目录结构：

```
├── __init__.py
├── config
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
└── main.py
```

### 主要代码

下面是主体代码：

```
from src.transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from src.transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from src.transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

config = Qwen2Config.from_pretrained("./config")
tokenizer = Qwen2Tokenizer.from_pretrained("./config")
model = Qwen2ForCausalLM(config=config)
print("模型细节： ")
print(model)
print("*="*80)
print("文本编码：")
inputs = tokenizer(["你好啊", "简单的机器学习是为了让机器学习变得更简单而存在的"],
                add_special_tokens=True,
                max_length=10,
                padding=True,
                truncation=True,
                return_tensors="pt")
print(inputs)
print("*="*80)
print("模型输出：")
print(model(**inputs))
```

不出意外的话，你可以看到下面的输出内容：

```
模型细节： 
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 896)
    (layers): ModuleList(
      (0-3): 4 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          (v_proj): Linear(in_features=896, out_features=128, bias=True)
          (o_proj): Linear(in_features=896, out_features=896, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
          (up_proj): Linear(in_features=896, out_features=4864, bias=False)
          (down_proj): Linear(in_features=4864, out_features=896, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((896,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=896, out_features=151936, bias=False)
)
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
文本编码：
{'input_ids': tensor([[108386, 103924, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
         151643],
        [105172, 102182, 100134, 104802,  99258, 102182, 100134, 112606, 100405,
          68536]]), 'attention_mask': tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
模型输出：
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
CausalLMOutputWithPast(loss=None, logits=tensor([[[ 1.5059,  0.6765,  0.2425,  ...,  0.4329, -0.0789, -1.0450],
         [ 0.3321,  0.8809,  0.6826,  ...,  0.0330,  0.0865, -0.6893],
         [ 0.2471,  0.7115,  0.5307,  ..., -0.0703,  0.1209, -0.7370],
         ...,
         [ 0.3910,  0.7432,  0.3905,  ...,  0.0459,  0.2107, -0.6613],
         [ 0.3790,  0.7864,  0.4028,  ...,  0.0793,  0.2166, -0.6966],
         [ 0.3704,  0.8088,  0.4358,  ...,  0.0567,  0.2196, -0.7045]],

        [[ 1.4859, -0.7797,  0.9490,  ..., -0.0205, -0.2090, -0.7289],
         [ 1.5965, -0.2371,  0.7803,  ..., -0.8275, -0.1699, -0.0016],
         [ 1.2100, -0.2230,  0.8658,  ..., -0.0166, -0.0579, -0.5130],
         ...,
         [ 0.5131, -0.2756,  0.8019,  ..., -0.0026,  0.3006, -1.2691],
         [ 0.2210, -0.0853,  0.9619,  ..., -0.1808,  0.5546, -1.0678],
         [ 0.4743,  0.1699,  0.6723,  ..., -0.0445,  0.4406, -0.9143]]],
       grad_fn=<UnsafeViewBackward0>), past_key_values=None, hidden_states=None, attentions=None)

```

有了上面的内容，我们基本流程就是搭好了，下面就可以使用我们自己喜欢的IDEA进行各种内容的调试了。我这里使用的是 `pycharm`。

## Qwen2Model

`Qwen2ForCausalLM`主体主要是 `Qwen2Model`，所以我们主要来看一下 `Qwen2Model`中的输入输出部分。

### 输入

对于 `Qwen2Model`的输入主要是以下参数

```
input_ids: torch.LongTensor = None,
attention_mask: Optional[torch.Tensor] = None,
position_ids: Optional[torch.LongTensor] = None,
past_key_values: Optional[List[torch.FloatTensor]] = None,
inputs_embeds: Optional[torch.FloatTensor] = None,
use_cache: Optional[bool] = None,
output_attentions: Optional[bool] = None,
output_hidden_states: Optional[bool] = None,
return_dict: Optional[bool] = None,
cache_position: Optional[torch.LongTensor] = None,
```

* `input_ids`的shape是 `[bs, seq_len]`，即batch_size和序列的长度组成的二维矩阵。里面的元素值是token在词汇表中对应的**索引**信息。
* `attention_mask`的shape和 `input_ids` shape是一直的，也是 `[bs, seq_len]`，元素取值要么是1，要么是0，1表示 `input_ids`对应位置的元素是有效的，0则表示无效的，在后续计算attention时，只有为1的位置才会被真正的计算。
* `position_ids`的shape也是 `[bs, seq_len]`，表达元素的位置的信息。
* `past_key_values`：预先计算的隐藏状态（自注意力块和交叉注意力块中的键和值），可以用来加速序列解码。这通常包括模型在解码的前一阶段返回的 `past_key_values`，当 `use_cache=True`或 `config.use_cache=True`时。

  允许两种格式：

  模型将输出与输入相同的缓存格式。如果没有传递 `past_key_values`，将返回传统的缓存格式。

  如果使用了 `past_key_values`，用户可以选择性地只输入最后一个 `input_ids`（那些没有给这个模型提供过去键值状态的），形状为 `(batch_size, 1)`，而不是所有 `input_ids`的形状 `(batch_size, sequence_length)`。

  > 注：这个参数一般情况在推理的时候使用，训练的时候不用。
  >

  * 一个 `~cache_utils.Cache`实例，参见我kv缓存指南;
  * 一个长度为 `config.n_layers`的元组，其中每个元组包含两个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)`的 `torch.FloatTensor`张量。这也被称为传统的缓存格式。
* `inputs_embeds`：形状为 `(batch_size, sequence_length, hidden_size)`， 可选地，您可以选择不传递 `input_ids`，而是直接传递嵌入表示。这在您想要对如何将 `input_ids` 索引转换为相关向量有更多的控制权时很有用，而不是使用模型内部的嵌入查找矩阵。
* `use_cache`：如果设置为 `True`，则返回 `past_key_values` 键值状态，可以用来加速解码（参见 `past_key_values`）。
* `output_attentions`：是否返回所有注意力层的注意力张量。有关返回张量的更多详细信息，请参见返回张量中的 `attentions`。
* `output_hidden_states`：是否返回所有层的隐藏状态。有关返回张量的更多详细信息，请参见返回张量中的 `hidden_states`。
* `return_dict`：是否返回一个 `~utils.ModelOutput`而不是一个普通的元组。
* `cache_position`：描述输入序列标记位置的索引。与 `position_ids` 不同，这个张量不受填充（padding）的影响。它用于在正确的位置更新缓存，并推断完整的序列长度。

上面就是在 `forward`中所需要的所有参数。下面我们将结合代码的内容实现，以及参数的具体值来简单实验一下。通过实验过程来逐步理解代码逻辑。

```
["你好啊", "简单的机器学习是为了让机器学习变得更简单而存在的"]
```

这个样例产生的tokens结果为：

```
{'input_ids': tensor([[108386, 103924, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
         151643],
        [105172, 102182, 100134, 104802,  99258, 102182, 100134, 112606, 100405,
          68536]]), 'attention_mask': tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

即得到的shape为：`[2, 10]`

由上一节 `print(model)`内容：

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 896)
    (layers): ModuleList(
      (0-3): 4 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          (v_proj): Linear(in_features=896, out_features=128, bias=True)
          (o_proj): Linear(in_features=896, out_features=896, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
          (up_proj): Linear(in_features=896, out_features=4864, bias=False)
          (down_proj): Linear(in_features=4864, out_features=896, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((896,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=896, out_features=151936, bias=False)
)
```

我们看到 `Qwen2Model`主体是由 `embed_tokens + 4*(self_attn + mlp + input_layernorm + post_attention_layernorm) + norm + rotary_emb`组成的。

### 详情

**embed_tokens层：**

`embed_tokens`就是我们熟悉的 `nn.Embedding`初始化得到的层。即：

```
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
```

其中：`config.vocab_size=151936, config.hidden_size=896, self.padding_idx=config.pad_token_id=None`。当 `self.padding_idx`为 `None`时候，默认取值就为0。

对于shape为 `[2, 10]`的输入，经过 `embed_tokens`层，可获得shape为 `[2, 10, 896]`，记为 `inputs_embeds`。

**cache_position和position_ids：**

因为 `cache_position和position_ids`都是 `None`(注：正对本样例而言)，所以cache_position直接是通过传进来的序列长度计算得到的，即为：`tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])`，`position_ids`为：`tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])`

**causal_mask：**

`causal_mask`是由方法 `self._update_causal_mask`产生的，它将产生四维的矩阵数据，shape为 `[2, 1, 10, 10]`，即 `[bs, 1, seq_len, seq_len]`，我们这里展示一下最后两维的数据，分别是 `causal_mask[0,0][:5, :5]`和 `causal_mask[1,0][:5, :5]`，如下：

```
# causal_mask[0,0][:5, :5]
tensor([[ 0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38]])
# causal_mask[1,0][:5, :5]
tensor([[ 0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
```

可以看出下三角全为0，`causal_mask[0,0][:5, :5]`不全为0的原因是由attention_mask引起的，即pad部分是不用去计算的。

**rotary_emb：**

`rotary_emb`层只计算一次，然后运用到后面的各层，这一层是没有参数的，不参与训练。使用旋转位置编码**最直接的好处**有：

* 可以使用绝对位置编码来表示相对位置编码；
* 计算量是线性的；
* 通过配置，可以实现一定的长度往外延拓能力；

`rotary_emb`主要用于计算cos和sin的值，即计算公式：

```
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    # 这里会获取到一个shape为[config.hidden_size // config.num_attention_heads//2]的inv_freq
        # 因为是多头，所以实际上每个头的维度是config.hidden_size // config.num_attention_heads
        # 再除以2是由公式确定的，具体看下面的公式矩阵.
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
       """
       x: shape为[bs, seq_len, hidden_size]
       position_ids: shape为[1, seq_len]
       """
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        # self.inv_freq本身的shape为[32], 经过下面的操作可获得[1, 32, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # shape为[1, 1, seq_len]
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
           # 经过[1, 32, 1]和[1, 1, seq_len]矩阵乘法之后可以得到[1, 32, seq_len]
            # 再经过变换可以得到[1, seq_len, 32]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # [1, seq_len, 64]
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

原位置编码公式：

代码里面得到的是：

> 注：更多文档请参考Transformer升级之路：2、博采众长的旋转式位置编码^[1]^

经过 `rotary_emb`可以得到 `position_embeddings`，它是一个元组，分别是 `(cos, sin)`对应的值，它们的 `shape`都是 `[1, seq_len, 64]`。

**self_attn: **

这里使用 `Qwen2SdpaAttention`来计算 `self_attention`，下面我们仔细介绍一下这个模块。

首先是 `Qwen2SdpaAttention`继承自 `Qwen2Attention`，然后修改了其forward方法。而 `Qwen2Attention`初始化方案主要初始化了4个可训练参数权重，分别是 `self.q_proj、self.k_proj、self.v_proj、self.o_proj`，如下代码：

```
self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

* `self.hidden_size=config.hidden_size=896`；
* `self.num_heads=config.num_attention_heads=14`；
* `self.head_dim=self.hidden_size // self.num_heads=64`；
* `self.num_key_value_heads=config.num_key_value_heads=2`；
* 注意这里的 `q, k, v`偏置全部设为了 `True`，即 `bias=True`；

接着我们看一下 `Qwen2Attention`中的 `forward`部分：

```
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    参数shape说明：
     hidden_states: [bs, seq_len, hidden_size]
     attention_mask: [bs, 1, seq_len, seq_len]
     position_ids: [1, seq_len]
     cache_position: [seq_len]
     position_embeddings: 元组数据，即(cos, sin)，shape都是[1, seq_len, self.head_dim]
    """
    bsz, q_len, _ = hidden_states.size()
  
    # [bs, seq_len, hidden_size]=[bs, seq_len, 896]
    query_states = self.q_proj(hidden_states)
    # [bs, seq_len, self.num_key_value_heads * self.head_dim]=[bs, seq_len, 128] 
    key_states = self.k_proj(hidden_states)
    # [bs, seq_len, self.num_key_value_heads * self.head_dim]=[bs, seq_len, 128]
    value_states = self.v_proj(hidden_states)
  # [bs, self.num_heads, seq_len, self.head_dim]=[bs, 14, sql_len, 64]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bs, self.num_key_value_heads, seq_len, self.head_dim] = [bs, 2, sql_len, 64]
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # [bs, self.num_key_value_heads, seq_len, self.head_dim] = [bs, 2, sql_len, 64]
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
       # cos: [1, seq_len, self.head_dim]=[1, seq_len, 64]
        # sin: [1, seq_len, self.head_dim]=[1, seq_len, 64]
        cos, sin = position_embeddings
    # 针对query_states和key_states运用旋转位置编码，即使用下面的公式。
    # 得到的shape为[bs, self.num_heads, seq_len, self.head_dim]=[bs, 14, seq_len, 64]
    # 和 [bs, self.num_key_value_heads, seq_len, self.head_dim] = [bs, 2, seq_len, 64]
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    # 这里的self.num_key_value_groups=self.num_heads // self.num_key_value_heads=7
    # num_key_value_groups作用请看下面注释。
    
    # 得到的shape为[bs, self.num_heads, seq_len, self.head_dim]=[bs, 14, seq_len, 64]
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
  
    # 计算attn_weights，其shape是[bs, self.num_head, seq_len, seq_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # [bs, self.num_head, seq_len, seq_len]矩阵乘以[bs, self.num_head, seq_len, head_dim]
    # 得到[bs, self.num_head, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
  
    # [bs, seq_len, self.num_head, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # [bs, seq_len, self.hidden_size]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
  # [bs, seq_len, self.hidden_size]
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
```

> 注：
>
> 在Transformer模型中，`num_key_value_groups` 是分组查询注意力（Grouped-query attention, GQA）的一个概念。分组查询注意力是多头注意力的一种改进形式，它在保持一定数量的query头的同时，减少key和value头的数量，以此来提高计算效率。
>
> 具体来说，`num_key_value_groups` 表示的是将key和value头分组的数量。在标准的多头注意力中，每个query头都会与一个对应的key和value头配对。但在分组查询注意力中，多个query头会共享一组key和value头。这样做可以减少模型的参数数量和计算量，从而提高效率。
>
> 例如，如果我们有8个query头，但在分组查询注意力中，我们可能只有4个key-value组，那么 `num_key_value_groups` 就是4。这意味着每两个query头会共享一个key和value头。在实际计算中，这组key和value会被复制（或者说广播）到与query头相同的数量，以便进行注意力权重的计算。
>
> 这种方法在保持多头注意力的优势的同时，减少了参数数量和计算复杂度，有助于提升模型的推理速度，尤其是在解码阶段。但是，它也需要仔细设计，以避免对模型性能产生负面影响。
>
> 在实际的代码实现中，`num_key_value_groups` 通常是通过将总的query头数除以key-value头数来计算的。例如，如果 `num_heads`（query头的数量）是8，而 `num_key_value_heads`（key-value头的数量）是4，那么 `num_key_value_groups` 就是2，意味着每两个query头共享一个key-value头。

**mlp：**

对于这一层，其实直接看代码就可以理解了，没有特别难的内容在里面。所以这里就不进行介绍了。


## 总结

本篇文章主要集中在介绍数据在流转的过程中，各个矩阵的shape，通过shape的变化，来理解整个过程。其实如果对Bert本身有理解的情况下，整篇内容只需要理解**旋转位置编码的实现以及分组查询注意力**的理解就好了，其它内容和Bert相比，并没有本质的变化（除了attention_mask部分）。
