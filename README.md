# 首个单台 DGX Spark 可用的 Qwen3.8-Flash-Next Abliterated NVFP4-PLE 模型

把一个为多机环境组装的 abliterated（uncensored）版 Qwen3.8-Flash-Next NVFP4-PLE 量化模型，改造为**单台 DGX Spark（GB10，128GB 统一内存）可用**：加载通过、推理正常、MTP 投机解码生效。

在此之前，单台 Spark 上只有官方量化版（hibrid47），**没有 abliterated NVFP4 版本**——唯一存在的 abliterated 组装产物（lychee，多机环境）直接放到单台 Spark 上：加载能过、health 200，推理输出全是垃圾（退化 argmax、logprobs 含 NaN、MTP 接受率 0%）。本仓库把这个模型改造成了单台可用，并完整记录改造过程。

## 成果

| 指标 | lychee 原版在单台 Spark | 改造后 |
|------|------------------------|--------|
| 推理输出 | 垃圾（无限重复 token） | 中文/英文/数学/多轮/工具调用全部正常 |
| logprobs | NaN | 48 值全部有限 |
| MTP 投机解码接受率 | 0% | 54.9%（平均接受长度 2.65） |
| 流式 TTFT | — | 0.3s |
| 运行形态 | — | 单台 Spark，8000 端口对外服务 |

## 为什么原版在单台 Spark 上是坏的

lychee 组装时量化元数据烂了三处（叠加生效，少修一个输出照样是垃圾）：

| # | 根因 | 修复 |
|---|------|------|
| 1 | L35 rogue `weight_scale` 键（BF16 [12288,1]，index 无记录）流入未量化 QKV 融合模块 | `fixes/orca_repair.py` 反量化为 BF16（e4m3 网格值 × 逐行 scale → BF16 原值），删除 rogue scale 键；该层 config 未声明量化，引擎按普通 BF16 Linear 构建 |
| 2 | MTP 层零量化声明 + 1536 个 input_scale 键缺失，引擎按未量化 FusedMoE 构建，NVFP4 scale 无处落 | `fixes/orca_fix2.py` 抄 h47 的 6 条 mtp 声明 + 回填 input_scale |
| 3 | **weight_scale_2 存成倒数** `2688/amax`（应为 `amax/2688`），73728 个标量膨胀 5 亿倍 → logits NaN | `fixes/fix3.py` 原位 4 字节补丁翻转全部 73728 个 scale_2 |
| 4 | **主模型 73728 个专家 input_scale 标量键整体缺失**，FLASHINFER_CUTLASS 后端拿未初始化内存当激活 scale | `fixes/fix5.py` 从 h47 提取并注入全部 73728 个标量 |
| 5 | **PLE 表 global scale 错 3400 倍**（0.6646，FP8 源的错误幅度约定），每次查表向 42 个 linear_attn 层灌放大 3400 倍的值 | `fixes/fix6_ple_global.py` LSQ 拟合正确值 1.954e-04，表 std 26.7 → 0.00745（corr 0.996） |

关键判断：**量化载荷本身是健康的**（PLE packed 字节与源逐元素 corr=1.0000，专家权重修完 scale 后数值与金标准对齐），坏的只是元数据。因此不做整体反量化再重量化——那会在 abliterated 权重上再糊一层量化误差、整片重写 43GB×17 分片；4 字节原位标量补丁零误差半径。唯二例外：L35 反量化为 BF16（该层本就未声明量化）、MTP 从裸 BF16 真量化（lychee 根本没量化它）。

修复金标准：同架构官方量化版 `Qwen3.8-Flash-Next-hibrid47`（键布局、量化格式、数值幅度全部对齐）。

其他关键细节：
- PLE global 不能直抄 h47 的 3.32e-05：orca 表内容幅度不同（g=1 时 std 40.1），直抄会小 5.9 倍，必须 LSQ 对齐反量化幅度。
- vLLM modelopt 加载器按 **safetensors 文件 header 枚举键**，不走 index——所有修复必须改文件本体，改 index 无效（两次实证）。

## 仓库结构

```
assembly/    lychee 量化组装管线（多机原版产物怎么来的，bug 就在这）
             assemble_orca.py  quant_ple.py  split_mtp.py  rename_experts.py
             dequant_g0.py  dequant_qsa.py  scan_fast.py  pipeline_orca.sh
fixes/       修复链（按执行顺序）
             orca_repair.py   根因1：L35 rogue 键 + MTP 重量化 + PLE 剥重复 global + index 重写
             orca_fix2.py     根因2：MTP 贴 h47（config 声明 + 1536 input_scale）
             fix3.py          根因3：scale_2 倒数原位翻转（73728 标量，JSON 备份）
             fix5.py          根因4：注入 73728 主模型专家 input_scale + PLE global 首版
             fix6_ple_global.py 根因5：PLE global LSQ 终版 1.954e-04
diagnosis/   诊断链（症状 → 根因的证据收集）
             diag2-6/A.py     分层级联探针（主模型专家 vs MTP vs PLE 表数值）
             scan8/9.py       全层键分布扫描（定位缺失/异常键）
             code_a-f.py      vLLM 源码消费路径分析 + h47/orca 结构 diff
             lattn_check.py   linear_attn 层量化状态核对
             probe_mtp.py     MTP 阶段加载崩溃探针
validate/    端到端验证
             restart-validate.sh  清缓存 → 重启 → health → 5 项质量探针总控
             orca-val.py           质量/TTFT/MTP 接受率/logprobs 探针
             recon.sh/recon_check.py 重启前预检（缓存状态、修复完整性）
launch/      部署启动
             orca-trial-8000.sh  单台 Spark 启动器（vLLM mmap v2 recipe）
             launch-orca.sh / launch-flashnext.sh / vllm-stack-start.sh
```

## 方法论（可复用到"多机产物 → 单机可用"的同类改造）

1. **找金标准**：同架构、已知能在目标机器上跑的量化版本做键布局 + 数值幅度参照系。
2. **先定症状边界**：流式空 content？logprobs NaN？MTP 接受率？每个症状对应不同的嫌疑层。
3. **逐层标量体检**：每层每专家解包 `weight / weight_scale / weight_scale_2 / input_scale`，比 std/amax/相关性。NaN 和量级异常（膨胀 5 亿倍、错 3400 倍）在这一步现形。
4. **读引擎源码确认消费路径**：加载器怎么枚举键、哪个 kernel 消费哪个 scale（如 FLASHINFER_CUTLASS 显式要 input_scale）——决定"缺的键"是不是致命的。
5. **改文件本体**：加载器不走 index，一切修复落到 safetensors 文件；原位 4 字节补丁比整文件重写安全（不碰其他 tensor）。
6. **每步备份 + 自洽检查**：修复后立即回读验证（std/corr 对齐金标准），重启前跑预检；fix5 的错误 global 值就是自洽检查在重启前抓住的。
7. **切换 checkpoint 必清 mmap 缓存**：稀疏空壳缓存会静默复用旧数据。

## 环境

- 目标机器：单台 DGX Spark（GB10，128GB 统一内存），Docker 容器执行所有权重文件操作（权限隔离）
- 推理栈：vLLM（modelopt_mixed，NvFp4 MoE backend = FLASHINFER_CUTLASS，PLE mmap gather）
- 模型：Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE（25 分片：17 主 + 1 MTP + 7 PLE，~43GB 主分片）

## 使用

```bash
# 1. 诊断（按顺序）
python3 diagnosis/diag2.py    # 分层级联：哪一层哪类张量数值坏
python3 diagnosis/scan9.py    # 全层键分布 vs 金标准
python3 diagnosis/code_c.py   # 引擎消费路径 + 结构 diff

# 2. 修复（按顺序，每步后回读验证）
python3 fixes/orca_repair.py
python3 fixes/orca_fix2.py
python3 fixes/fix3.py
python3 fixes/fix5.py
python3 fixes/fix6_ple_global.py

# 3. 重启 + 端到端验证
bash validate/restart-validate.sh
```

所有脚本内 API key / 密码已替换为占位符（`YOUR_API_KEY_HERE` 等），使用前替换。
