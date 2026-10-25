# 知识语料与许可核对

核对日期：2026-10-25。当前第三方语料仅来自本机已安装的 `pvlib==0.16.1` Python 发行包中的函数文档字符串；不是抓取未核对许可的网页，也未复制其参考文献中的论文正文。

## 第三方材料

- 来源仓库：[pvlib/pvlib-python](https://github.com/pvlib/pvlib-python)，文档定位及源模块 SHA-256 逐条记录在 [manifest.json](manifest.json)。版本来自发行包 metadata，内容从对应已安装模块提取；本轮没有另行联网核实 Git tag 与发行包逐字节一致。
- 许可证：**BSD-3-Clause**。已读取发行包中的完整许可：允许源码或二进制再分发，须保留版权、条件、免责声明，且不得使用贡献者名称背书。
- 完整版权与许可保存在 [licenses/pvlib-LICENSE.txt](licenses/pvlib-LICENSE.txt)，SHA-256 为 `a02e12ddaada3cf0d5dbdd8affdd577c2eec640758a95842a304f7513b8c0be4`。每个条目携带该许可路径、哈希及 attribution。
- 版权人：pvlib python Contributors、PVLIB python Development Team、Sandia National Laboratories。保留发行包原许可中的年份与完整措辞。

| 条目 | 实际收录范围 | 适用边界 |
|---|---|---|
| pvlib-soiling-hsu | `pvlib.soiling.hsu` 函数完整文档字符串 | HSU 模型及雨水清洗输入，不是本站积灰真值 |
| pvlib-shading-masking | `pvlib.shading.masking_angle_passias` 函数完整文档字符串 | 行间天空遮蔽几何，不覆盖所有真实遮挡机制 |
| pvlib-pvwatts-dc | `pvlib.pvsystem.pvwatts_dc` 函数完整文档字符串 | 模型定义、辐照/温度/额定功率输入，不提供本站参数 |
| pvlib-pvwatts-ac | `pvlib.inverter.pvwatts` 函数完整文档字符串 | 逆变效率及 AC 削顶模型，不证明硬件失效 |
| pvlib-faiman | `pvlib.temperature.faiman` 函数完整文档字符串 | 温度与经验散热参数，默认值不等于本站实测 |

## 项目编写的 Playbook

`../playbooks/F1.md` 至 `F6.md` 是本项目编写的 SKILL.md 格式候选诊断流程，manifest 明确标记 `kind=project_playbook`、`license=project-authored`，不冒充第三方公开指南或独立专家标注。其可再分发条件随项目未来确定的许可证；本记录不为整个项目新增开源授权。第三方材料的 BSD 许可不自动延伸至这些流程。

流程包含数据质量、取证、混淆因素和人工核实建议；它们不是设备测量证据或自动操作权限。F3/F4/F6 的真实硬件根因仍需现场或日志核实。

## 复现与摄取边界

在已有依赖环境下运行 `.venv/bin/python -m pvdiag.knowledge.build_corpus` 可离线重建。构建器仅接受已核对的 pvlib 版本和许可哈希，以及固定的六份 Playbook；不搜索 `data/`、`bench/`、诊断报告或基准答案。语料变更后需重建 manifest 并重启 MCP 服务；已启动服务保持不可变快照。

`KnowledgeStore` 只读 manifest 指定的 `kb/excerpts/`、`kb/licenses/` 与同级 `playbooks/` 文件，校验内容哈希、重复 ID、体积和路径边界。整个 manifest 和检索配置共同构成知识快照指纹，与观测快照指纹分开记录。

BM25 与 `hybrid_bge` 已在本机实跑。混合模式使用 BM25 + 本地向量余弦检索、RRF 融合与本地 BGE CrossEncoder 重排；推理时缺失依赖或经过校验的本地模型即显式失败，不下载、不转用在线服务、不静默回退。下载仅通过显式执行 `pvdiag.knowledge.model_assets` 进行。Ragas 与正式 A3 仍未实评。

本地加载参数先对照 sentence-transformers v5.1.2 的官方 [SentenceTransformer](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/SentenceTransformer.py) 与 [CrossEncoder](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/cross_encoder/CrossEncoder.py) 构造函数核对，随后在实际安装的 v5.1.2 上完成离线推理及 stdio MCP 验证。官方源码没有作为知识语料再分发。

开发检索集位于 `../bench/kb_qa_dev.json`，含项目编写的 30 个问题、参考回答和相关文档 ID。它仅供离线检索对比，未经过独立专家复核，也不包含电站场景真值；运行时知识服务不会加载它。纯代码文档指标不能替代 Ragas 的上下文/回答评测。

## 本地模型权重许可与版本

2026-10-25 已读取以下固定版本的模型卡 YAML 及 License 段落。两份卡片均标记 `license: mit`，并明确说明发布模型可免费商用，链接到 FlagEmbedding 的 MIT 许可。链接的完整许可已在固定 GitHub commit 下读取并保留于 [licenses/FlagEmbedding-LICENSE.txt](licenses/FlagEmbedding-LICENSE.txt)，版权为 `Copyright (c) 2022 staoxiao`，SHA-256 为 `587a673933425dbc36ec61268d3b954051b2d3ef3c9b322ede357976055ffdd5`；再分发时保留该许可及版权。

| 用途 | 模型卡及已固定 revision | 权重 SHA-256 |
|---|---|---|
| 512 维中文嵌入 | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620) | `354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026` |
| 中英双语重排 | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base/tree/2cfc18c9415c912f9d8155881c133215df768a70) | `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd` |

[model_assets.json](model_assets.json) 固定模型、revision、完整许可出处及各文件大小/哈希；LFS 权重和 tokenizer 大文件的 SHA-256 来自相应 revision 的 Hugging Face 文件元数据，并在下载后独立重算；其余小文件在固定 revision 下读取后计算哈希。仅选择 safetensors 权重及本地配置/tokenizer，不下载 pickle 权重、ONNX 副本或远程代码。

运行文件放在忽略的 `../data/models/`，本次合计下载 1,230,814,896 字节，不将权重提交 Git。每个目录附有模型卡、完整 MIT 许可及固定清单收据。下载器不会覆盖已有不匹配文件；中断后再次运行会核验并跳过已完成文件，重新下载未完成文件。推理加载再次核对许可、收据、配置、tokenizer 和权重哈希，并使用 `local_files_only=True`、`trust_remote_code=False`、`use_safetensors=True`。

可选依赖单独记录在 `../requirements/knowledge.in` 与 macOS arm64 / Python 3.11 的 `knowledge-macos-arm64.txt`。当前实际环境为 sentence-transformers 5.1.2、Transformers 4.57.6、PyTorch 2.11.0；未改动原有未跟踪的工程配置。该锁定环境尚未验证 Linux 部署，原核心 uv 配置也不管理这些可选包。
