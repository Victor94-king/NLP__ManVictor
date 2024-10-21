# 自然语言处理:第五十五章 RAG应用 FastGPT 快速部署

**代码：** [labring/FastGPT: FastGPT is a knowledge-based platform built on the LLMs, offers a comprehensive suite of out-of-the-box capabilities such as data processing, RAG retrieval, and visual AI workflow orchestration, letting you easily develop and deploy complex question-answering systems without the need for extensive setup or configuration. (github.com)](https://github.com/labring/FastGPT)

官网： [FastGPT](https://fastgpt.cn/en)

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

<br />

<br />


## 简介

> FastGPT 是一个基于 LLM 大语言模型的 **知识库问答系统** ，提供开箱即用的数据处理、模型调用等能力。同时可以通过可视化进行 **工作流编排** ，从而实现复杂的问答场景！

它界面简洁美观，功能完备强大。本文将介绍如何基于 Docker 快速部署 FastGPT，该方案非常适合个人或者小型团队。

## 系统要求

本方案已经在 Linux 上验证通过，笔者也建议选择 Linux 作为运行 FastGPT 的操作系统。

Intel 芯片的 MacOS 则没问题，可以正常运行 FastGPT docker 容器。

Apple 芯片的 MacOS 则不能正常运行 FastGPT docker 容器，因为部分 docker 镜像暂时不支持 Apple 芯片。

Windows 系统未验证，不过根据一些用户反馈，是能正确运行的，就是需要一定的动手和 Debug 能力。

## 硬件配置

因为 FastGPT 知识库功能需要上传的文档进行向量化并存储到向量数据库中，所以对 CPU、内存和存储有一定要求。

| 数据量      | 最低配置    | 推荐配置     |
| ----------- | ----------- | ------------ |
| 开发测试用  | 2c2g        | 2c4g         |
| 100w 组向量 | 4c8g 50GB   | 4c16g 50GB   |
| 500w 组向量 | 8c32g 200GB | 16c64g 200GB |

## 依赖软件

* Docker
* Git
* Ollama（可选）

Linux 下安装 docker 可参考下面的命令，或者 docker 官网安装文档^[1]^

```
# 安装 Docker
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
systemctl enable --now docker
# 安装 docker-compose
curl -L https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
# 验证安装
docker -v
docker-compose -v
# 如失效，自行百度~
```

## FastGPT 安装

架构图如下：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/ib3G4kbB3s926G8r7mAJsXf2EiaKnicNeSAjLmbtpE5zCgfhicc8ibTNWX1VJqbqWiazG1ByjcrORcibU1BctQdKwCibUQ/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Docker 运行 FastGPT 参考了 FastGPT 官方文档 ^[2]^ ，但是由于 docker-compose.yaml 文件中的配置经常变导致一些组件不兼容部署失败，所以笔者专门新建一个仓库 fastgpt-deploy ^[3]^ ，把验证通过的 compose 文件传上去，确保成功部署。

1. git 克隆仓库
   ```
   git clone https://github.com/dockerq/fastgpt-deploy.git
   ```
2. 下载 docker 镜像。由于网络原因这步可能耗时较长或者失败。
   ```
   cd fastgpt-deploy
   docker compose pull
   ```
3. 运行 fastgpt 服务
   ```
   docker compose up -d
   ```

## 更改 FastGPT 配置

参考 如何自定义配置文件？ ^[4]^ 。

每次变更配置都要重新运行 FastGPT 容器。

1. 变更配置文件 `config.json` 中的配置
2. 删除 fastgpt 容器
   ```
   docker compose down fastgpt
   ```
3. 重新运行 fastgpt 容器
   ```
   docker compose up -d fastgpt
   ```

## ollama 地址（可选）

如果你使用 Ollama 运行开源大模型，这部分要注意下，因为 FastGPT 都是运行在 Docker 容器中的，所以在 FastGPT/OneAPI 中配置 Ollama 地址时，要写对：

1. 如果在 Linux 系统运行，地址默认填 `http://172.17.0.1:11434`
2. 如果在 MacOS 系统运行，地址默认填 `http://host.docker.internal:11434`

### 参考资料

[1]docker 官网安装文档: *https://docs.docker.com/engine/install/*

[2]FastGPT 官方文档: *https://doc.fastgpt.in/docs/development/docker/#%E5%BC%80%E5%A7%8B%E9%83%A8%E7%BD%B2*

[3]fastgpt-deploy: *https://github.com/dockerq/fastgpt-deploy*

[4]如何自定义配置文件？: *https://doc.fastgpt.in/docs/development/docker/#%e5%a6%82%e4%bd%95%e8%87%aa%e5%ae%9a%e4%b9%89%e9%85%8d%e7%bd%ae%e6%96%87%e4%bb%b6*
