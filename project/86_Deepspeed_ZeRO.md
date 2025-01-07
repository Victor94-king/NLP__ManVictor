# 自然语言处理:第八十六章 Deepspeed各阶段配置你了解么？

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />


## 简单介绍

DeepSpeed是由微软基于PyTorch研发的开源深度学习优化库，主要目的是降低大模型训练的门槛，提升训练效率，并帮助开发者更有效地管理和优化大模型的训练和部署任务。它支持多种训练优化策略，包括：

1. **3D并行** ：数据并行、模型并行、流水线并行以及三者的混合使用。
2. **Zero Redundancy Optimizer（零冗余优化器）** ：包括ZeRO-0、ZeRO-1、ZeRO-2、ZeRO-3、ZeRO-Infinity等不同级别的优化。
3. **ZeRO-Offload** ：支持将数据、梯度、优化器状态等下沉到CPU和NVMe，以减少GPU内存压力。
4. **自定义混合精度训练** ：包括动态精度缩放（Dynamic Loss Scaling）和混合精度优化器（Mixed Precision Optimizer）。

此外，DeepSpeed还提供了许多大模型相关的工具，如分布式训练管理、内存优化和模型压缩等，以帮助开发者更好地管理和优化大规模深度学习训练任务。DeepSpeed在自然语言处理（NLP）和多模态等领域有许多成功的应用案例，可以极大提升大模型的训练速度、降低训练门槛以及训练成本，并因具备完整健康的社区生态，提升大模型的可用性。


## 特点和优势

DeepSpeed的主要特点和优势包括：

* **高效的并行化策略** ：支持多种并行化方法，能够显著提高训练速度和可扩展性。
* **内存优化技术** ：通过ZeRO技术减少内存占用，使得在有限的内存资源下训练更大的模型成为可能。
* **混合精度训练支持** ：减少内存占用和计算时间，降低能耗。
* **易用性和兼容性** ：与PyTorch等主流深度学习框架紧密集成，提供了易用的API和丰富的文档支持。

DeepSpeed的架构主要分为四个板块：Training、Inference、Compression、Science，每个板块都提供了相应的API和工具来支持深度学习的不同阶段和需求。通过使用DeepSpeed，开发者可以更加专注于模型的研究与改进，而将训练效率和性能的提升交给DeepSpeed来实现。

## ZeRO-1~ZeRO-Infinity

DeepSpeed的ZeRO技术通过在多个GPU之间分片模型的状态（包括优化器状态、梯度和参数），来消除数据并行进程中的内存冗余，从而提升训练效率和显存利用率。以下是ZeRO技术的工作机制：

1. **ZeRO-1（Stage 1）** ：在这个阶段，优化器状态被分片存储在不同的GPU上，每个GPU只更新它自己的那部分优化器状态。这减少了内存使用量，因为每个GPU不再需要存储完整的优化器状态。更新后的参数通过All-gather操作同步到所有GPU。
2. **ZeRO-2（Stage 2）** ：在ZeRO-1的基础上，ZeRO-2进一步将梯度也进行分片处理。每个GPU只保留与其优化器状态相对应的梯度部分。这样，每个GPU在反向传播后只需要合并自己的那部分梯度，然后更新模型参数，其余的梯度会被释放。
3. **ZeRO-3（Stage 3）** ：ZeRO-3在ZeRO-2的基础上进一步将模型参数也进行分片。在前向传播和反向传播过程中，每个GPU只保存部分参数。在需要时，通过All-Gather操作收集分布在其他GPU上的参数，以获得完整的模型参数进行计算。计算完成后，非本地维护的参数会被释放。
4. **ZeRO-Infinity** ：ZeRO-3的扩展，允许将模型状态卸载到CPU和NVMe内存中，以进一步减少GPU内存的使用。

ZeRO技术的核心思想是“万物皆可切，万物皆可抛”，即通过分片和卸载操作，减少每个GPU上的内存占用，同时保持计算和通信效率。这种方法使得在有限的硬件资源下训练更大的模型成为可能，并且可以显著提高显存效率和计算效率。通过这种方式，ZeRO技术能够在保持数据并行的计算粒度和通信效率的同时，显著降低内存占用，使得训练更大模型成为可能。以下是DeepSpeed的ZeRO-1、ZeRO-2、ZeRO-3以及ZeRO-Infinity的配置文件示例：

### ZeRO-1 配置文件示例

DeepSpeed ZeRO-1 是一种分布式训练优化技术，旨在减少分布式训练中的内存占用和通信开销。下面是一个DeepSpeed ZeRO-1的json文件配置样例，以及各个配置参数的含义：

```
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "use_multi_rank_bucket_allreduce": true,
    "allgather_partitions": true,
    "load_from_fp32_weights": true,
    "elastic_checkpoint": false
  }
}
```

配置参数含义：

1. `train_batch_size`: 训练批次大小，即每个GPU上每个训练步骤处理的样本数。
2. `gradient_accumulation_steps`: 梯度累积步数，用于模拟更大的批次大小。
3. `optimizer`: 定义优化器类型及其参数。

   * `type`: 优化器类型，如"Adam"。
   * `params`: 优化器参数，如学习率 `lr`和权重衰减 `weight_decay`。
4. `fp16`: 配置混合精度训练。

   * `enabled`: 是否启用混合精度训练。
5. `zero_optimization`: ZeRO优化配置。

   * `stage`: ZeRO优化的阶段，1表示ZeRO-1。
   * `reduce_bucket_size`: 梯度压缩时的桶大小，用于限制内存使用。
   * `allgather_bucket_size`: 参数聚合时的桶大小，用于限制内存使用。
   * `overlap_comm`: 是否尝试重叠梯度压缩与反向传播计算。
   * `contiguous_gradients`: 是否将梯度复制到连续缓冲区以避免内存碎片。
   * `reduce_scatter`: 是否使用reduce或reduce scatter代替allreduce来平均梯度。
   * `use_multi_rank_bucket_allreduce`: 是否将不同等级的reduce桶组合起来做All-Reduce，而不是多个Reduce操作。
   * `allgather_partitions`: 选择使用allgather collective还是一系列broadcast collectives来从所有GPU收集更新后的参数。
   * `load_from_fp32_weights`: 从fp32权重初始化fp32主权重，以避免精度损失。
   * `elastic_checkpoint`: 启用时可以加载由不同GPU数量的作业保存的检查点，但目前已不再支持。

这个配置文件为DeepSpeed ZeRO-1提供了一个基本的配置框架，可以根据具体需求调整参数以优化训练性能和资源利用。

### ZeRO-2 配置文件示例

ZeRO-2在ZeRO-1的基础上，进一步对梯度进行分片。以下是一个ZeRO-2配置文件示例：

```
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
```

配置参数含义：

1. `fp16`: 配置混合精度训练。

   * `enabled`: 是否启用混合精度训练。
   * `loss_scale`: 损失缩放值。
   * `loss_scale_window`: 损失缩放窗口大小。
   * `initial_scale_power`: 初始缩放幂。
   * `hysteresis`: 缩放调整的滞后性。
   * `min_loss_scale`: 最小损失缩放值。
2. `optimizer`: 定义优化器类型及其参数。

   * `type`: 优化器类型，如"AdamW"。
   * `params`: 优化器参数，包括学习率 `lr`、betas、eps和权重衰减 `weight_decay`。
3. `scheduler`: 定义学习率调度器类型及其参数。

   * `type`: 学习率调度器类型，如"WarmupLR"。
   * `params`: 学习率调度器参数，包括最小学习率 `warmup_min_lr`、最大学习率 `warmup_max_lr`和预热步数 `warmup_num_steps`。
4. `zero_optimization`: ZeRO优化配置。

   * `stage`: ZeRO优化的阶段，2表示ZeRO-2。
   * `offload_optimizer`: 指定优化器状态卸载到CPU的配置。
   * `allgather_partitions`: 是否使用分区的AllGather。
   * `allgather_bucket_size`: AllGather操作的桶大小。
   * `overlap_comm`: 是否重叠通信和计算。
   * `reduce_scatter`: 是否使用Reduce Scatter代替AllReduce。
   * `reduce_bucket_size`: Reduce操作的桶大小。
   * `contiguous_gradients`: 是否使用连续的梯度以减少内存碎片。
5. `steps_per_print`: 每多少步打印一次日志信息。
6. `wall_clock_breakdown`: 是否打印详细的时间分解信息。

这个配置文件为DeepSpeed ZeRO-2提供了一个基本的配置框架，可以根据具体需求调整参数以优化训练性能和资源利用。

### ZeRO-3 配置文件示例

以下是一个DeepSpeed ZeRO-3的json文件配置样例，以及各个配置参数的含义：

```
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

配置参数含义：

1. `fp16`: 配置混合精度训练。

   * `enabled`: 是否启用混合精度训练。
   * `loss_scale`: 损失缩放值。
   * `loss_scale_window`: 损失缩放窗口大小。
   * `initial_scale_power`: 初始缩放幂。
   * `hysteresis`: 缩放调整的滞后性。
   * `min_loss_scale`: 最小损失缩放值。
2. `optimizer`: 定义优化器类型及其参数。

   * `type`: 优化器类型，如"AdamW"。
   * `params`: 优化器参数，包括学习率 `lr`、betas、eps和权重衰减 `weight_decay`。
3. `scheduler`: 定义学习率调度器类型及其参数。

   * `type`: 学习率调度器类型，如"WarmupLR"。
   * `params`: 学习率调度器参数，包括最小学习率 `warmup_min_lr`、最大学习率 `warmup_max_lr`和预热步数 `warmup_num_steps`。
4. `zero_optimization`: ZeRO优化配置。

   * `stage`: ZeRO优化的阶段，3表示ZeRO-3。
   * `offload_optimizer`: 指定优化器状态卸载到CPU的配置，包括设备类型 `device`和是否使用pin内存 `pin_memory`。
   * `offload_param`: 指定参数卸载到CPU的配置，包括设备类型 `device`和是否使用pin内存 `pin_memory`。
   * `overlap_comm`: 是否尝试重叠梯度压缩与反向传播计算。
   * `contiguous_gradients`: 是否将梯度复制到连续缓冲区以减少内存碎片。
   * `sub_group_size`: 参数处理的块大小，用于适应超大模型。
   * `reduce_bucket_size`: 梯度压缩时的桶大小，用于限制内存使用。
   * `stage3_prefetch_bucket_size`: 参数预取的桶大小。
   * `stage3_param_persistence_threshold`: 不分区参数的最小大小阈值。
   * `stage3_max_live_parameters`: 允许在GPU上保持的最大参数数量。
   * `stage3_max_reuse_distance`: 参数重用的最大距离。
   * `stage3_gather_16bit_weights_on_model_save`: 是否在模型保存时收集16位权重。
5. `gradient_accumulation_steps`: 梯度累积步数。
6. `gradient_clipping`: 梯度裁剪值。
7. `steps_per_print`: 每多少步打印一次日志信息。
8. `train_batch_size`: 训练批次大小。
9. `train_micro_batch_size_per_gpu`: 每个GPU上的微批次大小。
10. `wall_clock_breakdown`: 是否打印详细的时间分解信息。

以上配置文件为DeepSpeed ZeRO-3提供了一个基本的配置框架，可以根据具体需求调整参数以优化训练性能和资源利用。

### ZeRO-Infinity 配置文件示例

ZeRO-Infinity允许将模型状态卸载到CPU和NVMe。以下是一个ZeRO-Infinity配置文件示例：

DeepSpeed ZeRO-Infinity 是一种用于大规模分布式深度学习的内存优化技术，它通过将模型状态（包括优化器状态、梯度和参数）在数据并行进程中进行分区，而不是复制，从而提高内存效率。以下是DeepSpeed ZeRO-Infinity 的一个JSON配置样例及其各个配置参数的含义：

```
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

参数含义如下：

1. **fp16** ：

* `enabled`: 是否启用半精度浮点数（16位浮点数）计算。
* `loss_scale`: 损失缩放值，用于防止半精度下的数值下溢。
* `loss_scale_window`: 调整损失缩放值的窗口大小。
* `initial_scale_power`: 损失缩放的初始指数。
* `hysteresis`: 调整损失缩放值之前可以容忍的连续步数。
* `min_loss_scale`: 损失缩放的最小值。

1. **optimizer** ：

* `type`: 优化器类型，例如AdamW。
* `params`: 优化器参数，包括学习率（`lr`）、betas（`betas`）、eps（`eps`）和权重衰减（`weight_decay`）。

1. **zero_optimization** ：

* `device`: 卸载设备，`cpu`表示CPU。
* `pin_memory`: 启用钉住内存，以提高数据传输效率。
* `device`: 卸载设备，`cpu`表示CPU。
* `pin_memory`: 启用钉住内存，以提高数据传输效率。
* `stage`: ZeRO优化的阶段，3表示启用ZeRO-3。
* `offload_optimizer`: 将优化器状态卸载到CPU内存中的配置。
* `offload_param`: 将模型参数卸载到CPU内存中的配置。
* `overlap_comm`: 启用通信重叠，以减少通信延迟。
* `contiguous_gradients`: 确保梯度数据在内存中是连续的。
* `sub_group_size`: 控制参数更新的粒度，防止内存不足。
* `reduce_bucket_size`: 自动设置reduce_bucket_size。
* `stage3_prefetch_bucket_size`: 自动设置stage3_prefetch_bucket_size。
* `stage3_param_persistence_threshold`: 自动设置stage3_param_persistence_threshold。
* `stage3_max_live_parameters`: GPU上保留的参数数量上限。
* `stage3_max_reuse_distance`: 参数重用的距离阈值。
* `stage3_gather_16bit_weights_on_model_save`: 在模型保存时启用16位权重的收集。

1. **gradient_accumulation_steps** ：

* 累积梯度的步数，`auto`表示自动设置。

1. **gradient_clipping** ：

* 梯度裁剪，`auto`表示自动设置。

1. **steps_per_print** ：

* 每多少步打印一次日志。

1. **train_batch_size** ：

* 训练批量大小，`auto`表示自动设置。

1. **train_micro_batch_size_per_gpu** ：

* 每个GPU上的微批量大小，`auto`表示自动设置。

1. **wall_clock_breakdown** ：

* 是否打印详细的时间分解日志，`false`表示不打印。

## 总结

理解各阶段参数，才能在显存不足或多 GPU 场景充分利用 GPU。
