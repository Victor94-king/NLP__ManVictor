# 自然语言处理:第九十九章 私有化RAG封神组合：Dify+RagFlow知识库

**本人项目地址大全：[Victor94-king/NLP__ManVictor: CSDN of ManVictor](https://github.com/Victor94-king/NLP__ManVictor)**

<br />

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

***写在前面: 笔者更新不易，希望走过路过点个关注和赞，笔芯!!!***

<br />

上一次分享[Dify+Fastgpt知识库](https://blog.csdn.net/victor_manches/article/details/146138714?spm=1001.2014.3001.5502)，收到部分朋友的反馈--想了解dify外接ragflow的效果。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPsQ1Pu8VePthbg3M0qd9BIe0pPQXuZIcNLGfeUBibf2oSrXxeibMnYxbw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPNX7CwlSRewoFV8KuwI5bibwjldQ1HC1hEsIevbSQZ8T2umgq9YB9PNw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

以及，还有一些朋友反馈说dify v1.0.0存在不少问题，所以大家都回退到之前版本了。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPjyPcRbEVcfXw2xpz9fNPlicLxbtIBhoyML8TiayCDIYCMAJGbTABaUlg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

好消息是：dify最近已经更新到了v1.0.1版本（更新/修复内容如下）

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPQcdjxyeJicOKwiaoFhnFAHRpZRQibHu07jaoxnpLdxZhlibpkTEY3CEEsw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

所以，响应大家的号召，今天就给大家带来dify外接ragflow知识库的详细步骤，一起看看接入之后效果到底怎么样~

顺便带大家一起把本地的dify升级到最新的v1.0.1版本。

本期使用的dify和ragflow都是使用docker本地部署的

[dify本地部署](https://mp.weixin.qq.com/s?__biz=MzkwMzE4NjU5NA==&mid=2247506421&idx=1&sn=240d895a1b03de12d1035cc186e142ab&scene=21#wechat_redirect)、[ragflow本地部署](https://mp.weixin.qq.com/s?__biz=MzkwMzE4NjU5NA==&mid=2247506528&idx=1&sn=f22fa5347d7c1aadf95f235c37174c59&scene=21#wechat_redirect)在往期文章都分享过，这次就不再赘述了。

![图片](https://mmbiz.qpic.cn/mmbiz_png/fRp5p4jMuDQjdXQXUMBDtPtLS0iaiaxVKblUBecgRUn30Lv2liaIUfnwcVib2D28Om4F0LpOd4oiah0psOJlRBHqewA/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

dify v1.0.0升级到最新v1.0.1

![图片](https://mmbiz.qpic.cn/mmbiz_png/jLdw7EZFJmIjAic1276gZeyjcsS9UMqa3VkvD2WgU11EyJAoVCSagkO3Kmia89jgusIXDficZIgTTb6ia32cibxVKgQ/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们先把本地的dify升级成最新v1.0.1版本（我的dify目前还是v1.0.0）

*PS：本次升级仅适用于docker部署方式*

先进入dify源码所在根目录/docker目录下，把docker-compose.yaml文件备份一个副本。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPACfBNZ51Os0Lwk3ibe5P8FGWJ1NAS9QCstR0hM49Sw5TNeYfM4Vjwxw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

然后去github上面下载dify v1.0.1最新的docker-compose.yaml文件。

把docker目录下旧的docker-compose.yaml替换掉

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPh41h8va8LmhK0cvMibNIj5t4vvMUqxsC879iaRDaYxBjNuxwpOAvk2icg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

没有科学上网的朋友也可以在公众号后台私信：“dify1.0.1” 获取最新v1.0.1版本的docker-compose.yaml文件

![](http://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CSrZbBrhRM4UPLqxenYcUGa4pyfWAODhWd5WLep1YrAgovib5EGO0TdTHOAZlhGdYDqZ1Qjj42USA/300?wx_fmt=png&wxfrom=19)

**袋鼠帝AI客栈**持续分享AI实践干货，走超级个体崛起之路

**134篇原创内容**

公众号

替换之后在地址栏输入cmd 回车，进入docker-compose.yaml所在目录的控制台。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPZqT516XIzM6FR5nUiajHOg6OJtOs7tib2fwRnib2TATbEnGU32OgBLWNQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在控制台执行docker-compose up -d

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPO6qnjoFWu6vZIdmJHOzsfCKoG1K7ufeibgUNvmicxnfC5GibBNjjzVcYA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

直到出现如下日志，就代表升级启动成功啦

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP4jyZGicNbKjFlRnwNfddLBISLQp0ia7rDc01kS0MgF0ec5Em5ahzjn5A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这时候我们访问dify页面：127.0.0.1

点击右上角头像，下拉框中可以看到已经成功升级到v1.0.1了

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPiayP8kr9iaouBvKcnIks5jzFba2ZPARRnVmiarqgslC077IVXc83V7kSQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/fRp5p4jMuDQjdXQXUMBDtPtLS0iaiaxVKblUBecgRUn30Lv2liaIUfnwcVib2D28Om4F0LpOd4oiah0psOJlRBHqewA/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

dify外接ragflow知识库

![图片](https://mmbiz.qpic.cn/mmbiz_png/jLdw7EZFJmIjAic1276gZeyjcsS9UMqa3VkvD2WgU11EyJAoVCSagkO3Kmia89jgusIXDficZIgTTb6ia32cibxVKgQ/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

首先我们需要解决一个问题，就是在本地的ragflow和dify的页面默认的访问端口是有冲突的（用的都是80和443端口）。

如果不解决这个问题就会导致某一方无法正常启动。

我的解决方案是修改ragflow的默认端口，可以参考我的ragflow配置（如下：在docker-compose.yml里面把ragflow映射到主机的端口改掉，改成容器的80端口映射到主机的8000端口，433端口映射到主机的4333端口）

这样就不会和dify的主机端口冲突了

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP1qOj1TibflyNfxWRzeicWTL2O80Fz2bhFy9NkfMickljJ9a1qicnWia9ZRw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

docker-compose.yml有改动的话

需要重新执行docker-compose up -d来重置服务使配置生效（执行位置还是要在docker-compose.yml所在的当前路径）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP2MrK84Ff8xZzztZyYvXllviaLCaILx8V3iblYc2L1MeD0hBZFPyAFSGw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

重置ragflow之后，我们就可以通过：127.0.0.1:8000 来访问ragflow的页面了。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPxx8ycdhIoNRIPDqSlM4GicNpeaLB5ARZqqLbLP9UxR6drzHKvKxhQdQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

并且ragflow官方已经提供了dify外接知识库的接口，所以不需要像之前dify外接fastgpt那样自己开发一套适配程序了。

接下来的整个对接过程非常丝滑~

首先，我们需要去拿到ragflow的三个要素：

知识库的api地址、apikey、知识库id

点击ragflow右上角头像->API->API KEY->创建新密钥（复制备用）

并把 API服务器地址 复制备用

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPy9XusdRiaINPxyHdncKoWoYCp0EPKfic3qVvOxEmdicYLOhAtc9W5Turg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我准备把下面 ragflow中的内存条商品表知识库，外接到dify

点击进入

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPkLAhcyoqhWNyLhEN8W9dME0HzIn8myNgic1vdpMBTlicatZMzsq9AsuA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在路径栏中复制知识库id 备用

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPoocb0Az2icqvKamjVnBiaTicB9be2YdEm9MCM8RGc0GdIA3KewUQkDrIg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

回到dify这边

在知识库->外部知识库->添加外部知识库

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPvUia1znj8ib4TickOU64Uqic9AiaTn72ARCkRcoeAVY5aI7zqUAicTzWiaT8w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

name随便填一个

API Ednpoint：填写http://<ragflow地址>:9380/api/v1/dify

apikey填写刚才在ragflow创建好的apikey，点击保存

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPYgZ4hTQHMh4ViaUExY86Libqpc5fwx6QJib8fRQiayHHd6D5rQ9TpceoNQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*PS：由于我的dify和ragflow都部署在同一个主机的docker中，所以dify可以通过主机的内网ip访问ragflow的知识库。*

windows可以通过在控制台输入ipconfig找到本机内网ip

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPtl2ibEibY1DibHFl5ic4XwXCLCjibfGs6lCFg29N6NhfXMfpxYI8EASWFhw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Linux可以输入ifconfig找到本机内网ip

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPPKTM1EQckh8rTd2ibLl7n9t3Cs9YhBKfXiaNRvYGgDVBxciaL4leKkMibQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

保存成功的话，会有成功的提示，并且在外部知识库中会增加一条

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP0lAia0YbGnwMoYtSZ3TN4x1bacq4icnNQlp2T3xahy5gaovwB0iar8Log/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

点击连接外部知识库

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPEboE9EneCNUIHZQ0XH1ibVSSCZC9ribbZReiaoHmhQ8ObrseD6M363AwQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

按照下图，填写好信息

TopK 和 相似度阈值自行根据需要调整，最后点击连接

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP0hUp43aSXfv6qAPvLNQHIDGbZbUECyCsBcbD0xLzLp9JibIbpE5Viaew/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

就创建成功啦

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPk3jpa1G5V2HqdxMZngAIbuD2VDaTtKLGHhCZVLLFR0nfSBxw8CJiaDQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

接下来我们测试看看~

创建一个空白应用，关联刚刚创建的 ragflow-内存条商品表知识库

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPRx6UxMHtSRdnn6UXS0VgmDibSHLu9D8ed1hUQkkqjOaxzGmRRljb9kw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这里我没有开启重排(不过大家后续使用也可以开启，测试效果)

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaP7Z1ZVXAZwQu7TMQVOkPR04eIjpUPRicP2GMy0LAabDEHiaQlLBD0msfA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我在ragflow和dify两边都创建了测试应用(参数都调整到相同状态)

测试了一下（下图，左边dify，右边ragflow）

![图片](https://mmbiz.qpic.cn/mmbiz_png/y3cJRRodqQ6JUhJuC8CadYxg5ludQCvdAzH8zukzJicZPY51nosEuvBRtrxyeTMgSYoKFhayk8xKMYKFxXgrsSg/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

AUTUMN

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPFbJjRmeibKTVxStAcZDFYmjkXibTKLBeqpOYjYkCTqSkBKWRcdM9EXCA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPFqzzOu90pl1fibs7zG4DGboW3Ensp0eb5CEAKUbD3v01iboVaoV4icssQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我检查了原表格数据，确实表格里面所有海盗船内存条中仅有一种是32G的。

![图片](https://mmbiz.qpic.cn/mmbiz_png/3QzcPBL9P1CJdfviaMG9M1dhHLTSMibtaPjjobDOFYCmrRpTdGhgrumIGQibcFxsuK5jRR8gtfMcNLb6pfV3gz9ZQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

回答挺准的，两边效果一致！目标达成。

*本次仅测试知识库问答效果，不对内存条产生购买建议*

dify借助ragflow很大程度弥补了知识库解析、知识库问答效果的不足，最方便的是ragflow官方本身就支持了dify的外部知识库API。

这样接入还挺丝滑的，非常推荐！
