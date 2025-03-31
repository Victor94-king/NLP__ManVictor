# 自然语言处理:第一百零九章 单卡4090微调DeepSeek-R1-32B

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


在 24G 显存的单卡 4090 上微调训练 deepseek-ai/DeepSeek-R1-Distill-Qwen-32B；即使该模型的权重文件大小已经达到 62G，这是因为 unsloth 和 lora 的量化微调和部分参数微调优化可以大幅节约显存占用。

因为设置了**max_steps=**60 限制了只执行60步以便快速完成实验。去掉这个参数后，SFTTrainer 即可根据数据量自动计算微调步数了。这次进行了 **FreedomIntelligence**/medical-o1-reasoning-SFT 数据集 24772 条数据的全量微调，epoch为默认值3，微调结果如下：

* **训练总步数 (Total steps) : 9288 步**
* **训练总轮次 (Epochs) : 3.0 轮 **
* **每轮数据量: 24,772 条数据**
* **训练时间: 总计 28小时28分37秒（102517.8411 秒）**

本次训练在贝联云算力平台( https://cloud.lccomputing.com )上完成。

完整训练代码如下：

```
import wandb
# 登录 wandb.ai 用于实验跟踪
wandb.login(key="放置你的wandb.ai网站上的token")
# 初始化wandb项目
run = wandb.init(
    project='Lora-R1-Distill-Qwen on Medical COT Dataset',
    job_type="training",
    anonymous="allow"
)


####################################################################################################
# 1.加载模型


# 使用 unsloth 优化的 FastLanguageModel 加载模型
from unsloth import FastLanguageModel
max_seq_length = 4096 # 最大序列长度
dtype = None          # 数据类型，None表示自动选择
load_in_4bit = True   # 使用4bit量化加载模型以节省显存


# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    model_name = "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    local_files_only=True,  # 避免联网
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #token = hf_token, 
)
print(model)


####################################################################################################
# 2. 定义提示模板，并在微调前做一次推理


prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
  请写出恰当完成该请求的回答。
  在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。
  ### Instruction:
  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
  请回答以下医学问题。
  ### Question:
  {}
  ### Response:
  <think>{}"""
train_prompt_style = prompt_style + """
  </think>
  {}"""


# 测试用医学问题
question = "一名70岁的男性患者因胸痛伴呕吐16小时就医，心电图显示下壁导联和右胸导联ST段抬高0.1~0.3mV，经补液后血压降至80/60mmHg，患者出现呼吸困难和不能平卧的症状，体检发现双肺有大量水泡音。在这种情况下，最恰当的药物处理是什么？"


# 设置模型为推理模式
FastLanguageModel.for_inference(model) 
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")


# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("### 微调前模型推理结果：")
print(response[0].split("### Response:")[1])


####################################################################################################
# 3. 处理数据集


EOS_TOKEN = tokenizer.eos_token  # 添加结束符标记
#格式化提示函数,用于处理数据集中的示例
def formatting_prompts_func(examples):
    # 从examples中提取问题、思维链和回答
    inputs = examples["Question"]      # 医学问题列表
    cots = examples["Complex_CoT"]     # 思维链列表 
    outputs = examples["Response"]     # 回答列表
    
    # 存储格式化后的文本
    texts = []
    
    # 遍历每个示例,将问题、思维链和回答组合成指定格式
    for input, cot, output in zip(inputs, cots, outputs):
        # 使用train_prompt_style模板格式化文本,并添加结束符
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
        
    # 返回格式化后的文本字典
    return {
        "text": texts,
    }


# 加载数据集并应用格式化
from datasets import load_dataset,load_from_disk
dataset = load_dataset(
    "json",  # 指定数据格式为 JSON
    data_files="/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
    #split="train[0:500]",  # 只取前 500 条数据
    trust_remote_code=True  # 兼容 remote code 的行为
)


# 如果返回的是 DatasetDict，则取出 "train" 这一部分
if isinstance(dataset, dict):  
    dataset = dataset["train"]
    
dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset)  # 查看数据集结构


####################################################################################################
# 4. 配置训练参数启动训练


model = FastLanguageModel.get_peft_model(
    model, 
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",    
    ],
    lora_alpha=16,
    lora_dropout=0,  
    bias="none",  
    use_gradient_checkpointing="unsloth", 
    random_state=8137,
    use_rslora=False,  
    loftq_config=None,
)
print(model)


# 配置训练参数和初始化训练器
from trl import SFTTrainer  
from transformers import TrainingArguments  
from unsloth import is_bfloat16_supported  


# 初始化 SFT 训练器
trainer = SFTTrainer(
    model=model,  
    tokenizer=tokenizer,  
    train_dataset=dataset,  
    dataset_text_field="text",  # 数据集字段的名称
    max_seq_length=max_seq_length,  
    dataset_num_proc=2,  # 数据集处理的并行进程数，提高CPU利用率
    args=TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,   
        warmup_steps=5,  # 预热步数,逐步增加学习率
        learning_rate=2e-4,  # 学习率
        lr_scheduler_type="linear",  # 线性学习率调度器
        # max_steps=200,    # 最大训练步数（一步 = 处理一个batch的数据）
        fp16=not is_bfloat16_supported(),  # 如果不支持bf16则使用fp16
        bf16=is_bfloat16_supported(),      # 如果支持则使用bf16
        logging_steps=10,  # 每10步记录一次日志
        optim="adamw_8bit",  # 使用8位AdamW优化器节省显存，几乎不影响训练效果
        weight_decay=0.01,   # 权重衰减系数,用于正则化，防止过拟合
        seed=8137,  # 随机数种子
        output_dir="outputs",  # 保存模型检查点和训练日志
        run_name="medical-o1-sft-experiment",  # 显式设置 wandb 运行名称，避免警告
    ),
)


# 开始训练
print(f"trainer.args.max_steps: {trainer.args.max_steps}")
print(f"trainer.args.num_train_epochs: {trainer.args.num_train_epochs}")
trainer.train()
print(f"Total training steps: {trainer.state.max_steps}")
print(f"Total epochs: {trainer.state.epoch}")


####################################################################################################
# 5. 微调后的模型做一次推理


FastLanguageModel.for_inference(model)  
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")


# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids, # 输入token的id序列
    attention_mask=inputs.attention_mask,  # 注意力掩码,用于标记有效输入位置
    max_new_tokens=1200, # 生成的最大新token数量
    use_cache=True, # 是否使用KV缓存加速生成
)


response = tokenizer.batch_decode(outputs)
print("### 微调后模型推理结果：")
print(response[0].split("### Response:")[1])


####################################################################################################
# 6. 保存模型


new_model_local = "DeepSeek-R1-Medical-COT-Qwen-32B"
model.save_pretrained(new_model_local) 
tokenizer.save_pretrained(new_model_local)


# 保存合并后的16bit模型
model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)


# 保存为 GGUF 模型
# model.save_pretrained_gguf("DeepSeek-R1-Qwen-32B-Medical-COT-GGUF", tokenizer,)
```

完整日志如下：

```
ounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(lineounter(line
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.6
wandb: Run data is saved locally in /workspace/wandb/run-20250212_150918-mvocwedu
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ruby-wind-2
wandb: ⭐️ View project at https://wandb.ai/xlxkming-none/Lora-R1-Distill-Qwen%20on%20Medical%20COT%20Dataset?apiKey=edb4e5ad4f056c86bc64f3ea1d5b327e88378327
wandb: 🚀 View run at https://wandb.ai/xlxkming-none/Lora-R1-Distill-Qwen%20on%20Medical%20COT%20Dataset/runs/mvocwedu?apiKey=edb4e5ad4f056c86bc64f3ea1d5b327e88378327
wandb: WARNING Do NOT share these links with anyone. They can be used to claim your runs.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
INFO 02-12 15:09:30 __init__.py:190] Automatically detected platform cuda.
==((====))==  Unsloth 2025.2.4: Fast Qwen2 patching. Transformers: 4.48.3.
   \\   /|    GPU: NVIDIA GeForce RTX 4090. Max memory: 23.65 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.5.1+cu121. CUDA: 8.9. CUDA Toolkit: 12.1. Triton: 3.1.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post1. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|██████████| 8/8 [00:16<00:00,  2.07s/it]
Unsloth 2025.2.4 patched 64 layers with 64 QKV layers, 64 O layers and 64 MLP layers.
/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B does not have a padding token! Will use pad_token = <|vision_pad|>.
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 5120, padding_idx=151654)
    (layers): ModuleList(
      (0-63): 64 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear4bit(in_features=5120, out_features=5120, bias=True)
          (k_proj): Linear4bit(in_features=5120, out_features=1024, bias=True)
          (v_proj): Linear4bit(in_features=5120, out_features=1024, bias=True)
          (o_proj): Linear4bit(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear4bit(in_features=5120, out_features=27648, bias=False)
          (up_proj): Linear4bit(in_features=5120, out_features=27648, bias=False)
          (down_proj): Linear4bit(in_features=27648, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
        (post_attention_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
      )
    )
    (norm): Qwen2RMSNorm((5120,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=5120, out_features=152064, bias=False)
)
### 微调前模型推理结果：
  <think><think>
嗯，这个问题看起来有点复杂，但我会一步一步来分析。首先，我需要理解患者的情况和检查结果，然后确定可能的诊断，最后选择合适的药物治疗。
患者是一位70岁的男性，因胸痛和呕吐16小时来就医。心电图显示下壁导联和右胸导联的ST段抬高，这可能提示心肌梗死，特别是下壁和右室梗死。因为下壁心肌梗死通常与右冠状动脉阻塞有关，而右胸导联的ST抬高可能涉及右心室。
接下来，患者在补液后血压降至80/60 mmHg，这可能意味着低血压，但补液后血压反而下降，这可能是因为心脏功能受损，无法有效泵血，导致心源性休克。同时，患者出现呼吸困难和不能平卧，体检发现双肺有大量水泡音，这可能提示肺水肿，尤其是心源性肺水肿，因为心脏无法有效泵血，导致液体积聚在肺部。
现在，我需要确定患者的具体情况。下壁和右室梗死可能导致心脏泵血功能下降，特别是右心室功能不全，影响心脏的输出，导致低血压和肺水肿。这种情况下，患者的血流动力学状态可能不稳定，需要紧急处理。
接下来，考虑药物治疗。通常，对于心肌梗死，我们会使用抗血小板药物（如阿司匹林）、抗凝药物（如肝素或替格瑞洛），以及β受体阻滞剂、ACEI或ARB类药物来改善心脏功能和减少心脏负荷。然而，患者现在血压低，可能不适合使用ACEI，因为ACEI可能会进一步降低血压，导致低血压加重。
此外，患者出现肺水肿，可能需要利尿剂来减轻肺部液体积聚。但利尿剂可能会导致血容量进一步减少，从而加重低血压，这可能不太适合当前的情况。
考虑到患者的低血压和肺水肿，可能需要使用正性肌力药物，如多巴胺或多巴酚丁胺，来增强心脏收缩力，改善心脏输出，从而提升血压和减轻肺水肿。同时，可能需要调整其他药物的使用，以避免进一步影响血压。
另外，患者可能需要机械通气支持，特别是如果呼吸困难严重，无法平卧，可能需要无创通气或插管。但这可能超出了当前药物处理的范围。
综上所述，患者的情况可能涉及下壁和右室心肌梗死，导致心源性休克和肺水肿。在这种情况下，最恰当的药物处理可能包括使用正性肌力药物（如多巴胺或多巴酚丁胺）来改善心脏功能，同时继续抗血小板和抗凝治疗，但需谨慎调整以避免低血压加重。可能还需要利尿剂来减轻肺水肿，但需在监测下使用，以防止血容量过低。
当然，具体情况可能需要进一步评估，如心脏超声检查，以确定右心室的功能和是否存在机械并发症，如室间隔穿孔或乳头肌功能不全。此外，可能需要介入治疗，如冠状动脉造影和支架植入，以恢复血流，改善心脏功能。
但根据问题，主要是在药物处理方面，因此重点应放在使用正性肌力药物和支持性治疗上，同时监测和调整其他药物的使用。
</think>
针对该患者的情况，最恰当的药物处理如下：
1. **抗血小板和抗凝治疗**：继续使用阿司匹林和氯吡格雷（或替格瑞洛），并给予肝素抗凝，以防止血栓进一步形成。
2. **正性肌力药物**：使用多巴胺或多巴酚丁胺，以增强心脏收缩力，改善心脏输出，提升血压，并减轻肺水肿。
3. **利尿剂**：在监测下使用利尿剂（如呋塞米）以减轻肺水肿，但需注意避免血容量过低。
4. **避免使用ACEI或ARB**：由于患者血压低，暂时避免使用ACEI或ARB，以防止进一步降低血压。
5. **监测和支持治疗**：密切监测患者的生命体征，必要时进行机械通气支持，并考虑介入治疗（如冠状动脉造影和支架植入）以恢复血流。
综上所述，药物处理的重点在于使用正性肌力药物和支持性治疗，同时继续抗血小板和抗凝治疗，以改善心脏功能和血流动力学状态。<｜end▁of▁sentence｜>
Dataset({
    features: ['Question', 'Complex_CoT', 'Response', 'text'],
    num_rows: 24772
})
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(152064, 5120, padding_idx=151654)
        (layers): ModuleList(
          (0-63): 64 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=5120, out_features=5120, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=5120, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=5120, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
             。。。
)
...
```

训练过程和结果日志：

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 24,772 | Num Epochs = 3
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
\        /    Total batch size = 8 | Total steps = 9,288
 "-____-"     Number of trainable parameters = 268,435,456


trainer.args.max_steps: -1
trainer.args.num_train_epochs: 3.0
{'loss': 2.149, 'grad_norm': 0.16384364664554596, 'learning_rate': 0.00019989227620381345, 'epoch': 0.0}
{'loss': 1.5362, 'grad_norm': 0.07211203873157501, 'learning_rate': 0.0001996768286114403, 'epoch': 0.01}
{'loss': 1.4647, 'grad_norm': 0.07446285337209702, 'learning_rate': 0.00019946138101906713, 'epoch': 0.01}
...
{'loss': 1.39, 'grad_norm': 0.08653779327869415, 'learning_rate': 0.0001977378002800819, 'epoch': 0.04}
...
{'loss': 1.2627, 'grad_norm': 0.1181635782122612, 'learning_rate': 0.00013590434126898633, 'epoch': 0.96}
...
{'loss': 1.1951, 'grad_norm': 0.11674296855926514, 'learning_rate': 0.00013224173219864268, 'epoch': 1.02}
...
{'loss': 1.071, 'grad_norm': 0.1962611824274063, 'learning_rate': 3.1843154152752344e-05, 'epoch': 2.52}
...
{'loss': 0.9945, 'grad_norm': 0.17683860659599304, 'learning_rate': 2.2794355273079824e-05, 'epoch': 2.66}
...
{'loss': 1.1104, 'grad_norm': 0.21208912134170532, 'learning_rate': 6.032532586448347e-07, 'epoch': 2.99}
{'loss': 1.0957, 'grad_norm': 0.2164667695760727, 'learning_rate': 3.8780566627167944e-07, 'epoch': 2.99}
{'loss': 1.101, 'grad_norm': 0.21290326118469238, 'learning_rate': 1.723580738985242e-07, 'epoch': 3.0}


100%|██████████| 9288/9288 [28:28:37<00:00, 11.04s/it]


{'train_runtime': 102517.8411, 'train_samples_per_second': 0.725, 'train_steps_per_second': 0.091, 'train_loss': 1.210533706973484, 'epoch': 3.0}
Total training steps: 9288
Total epochs: 2.999192636848054
### 微调后模型推理结果：


  <think>患者是个70岁的男性，他因为胸痛和呕吐来到医院，这让我首先想到可能和心脏有关。心电图显示下壁导联和右胸导联的ST段抬高，嗯，这可能意味着下壁心肌梗死。接下来，他的血压在补液后降到了80/60mmHg，这很低，而且他还出现了呼吸困难，不能平卧，肺里有水泡音，这些症状让我怀疑他有心源性休克或者急性心衰。


心源性休克和急性心衰通常需要快速处理，因为这会危及生命。首先想到的是要稳定他的血流动力学状态。通常，这种情况下我们会使用正性肌力药物，比如多巴酚丁胺，因为它可以增加心肌收缩力，提高心输出量，帮助改善低血压和呼吸困难的症状。


不过，等等，患者有低血压和肺水肿，这让我觉得可能不仅仅是一个单纯的心源性休克，还可能有液体过载的问题。如果是液体过多，使用利尿剂如呋塞米可能更有效，因为它能帮助排除多余的液体，减轻肺水肿，同时降低心脏的负担。


再想想，患者的低血压情况很严重，而且不能平卧，这可能提示心脏泵功能非常差。在这种情况下，使用正性肌力药物来提升心脏收缩力可能更合适。多巴酚丁胺可以增加心输出量，同时改善低血压，这可能是目前更好的选择。


哦，对了，患者的心电图显示下壁ST段抬高，这可能提示右心室梗死。右心室梗死可能导致心源性休克，需要特别注意。在这种情况下，使用多巴酚丁胺来增强心脏收缩力和提高心输出量可能是更合适的。


综上所述，考虑到患者严重的低血压、呼吸困难和右心室梗死的可能性，使用多巴酚丁胺来迅速改善血流动力学状态是最恰当的。嗯，这应该是一个明智的选择。
  </think>
  在这种情况下，患者表现出低血压、呼吸困难、不能平卧以及肺部有水泡音，这些症状提示可能存在心源性休克或急性心力衰竭。心电图显示下壁导联和右胸导联的ST段抬高，提示可能有下壁心肌梗死，甚至可能涉及右心室梗死。


对于这种情况，最恰当的药物处理是使用正性肌力药物来改善心脏的泵功能，提高心输出量，从而改善低血压和呼吸困难的症状。多巴酚丁胺（dobutamine）是一种常用的正性肌力药物，可以增加心肌收缩力，提高心输出量，同时在一定程度上扩张血管，降低心脏后负荷，有助于改善患者的血流动力学状态。


因此，考虑到患者目前的血流动力学不稳定和可能的右心室梗死，使用多巴酚丁胺是合理且必要的选择。这种药物干预可以迅速帮助稳定患者的病情，为后续的治疗争取时间。<｜end▁of▁sentence｜>
Unsloth: Merging 4bit and LoRA weights to 16bit...
Unsloth: Will use up to 303.83 out of 503.72 RAM for saving.
Unsloth: Saving model... This might take 5 minutes ...
  0%|          | 0/64 [00:00<?, ?it/s]
We will save to Disk and not RAM now.
100%|██████████| 64/64 [01:34<00:00,  1.47s/it]
Unsloth: Saving tokenizer... Done.
Done.
wandb:
wandb: 🚀 View run ruby-wind-2 at: https://wandb.ai/xlxkming-none/Lora-R1-Distill-Qwen%20on%20Medical%20COT%20Dataset/runs/mvocwedu?apiKey=edb4e5ad4f056c86bc64f3ea1d5b327e88378327
wandb: Find logs at: wandb/run-20250212_150918-mvocwedu/logs
```

峰值资源占用。这时这次 lora rank 32的：

```
ounter(lineounter(lineounter(lineounter(lineounter(line
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
| 76%   64C    P2             392W / 450W |  24176MiB / 24564MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

之前 lora rank 8 也测过，和 rank 32 比资源占用并没少多少：

```
ounter(lineounter(lineounter(lineounter(lineounter(line
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
| 82%   65C    P2             394W / 450W |  21246MiB / 24564MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+-------------
```
