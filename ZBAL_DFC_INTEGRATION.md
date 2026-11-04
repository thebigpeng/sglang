# ZBAL DFC 接入分析、方案与实施记录

更新日期：2026-11-04。

- 目标仓库：`D:\work\memfabric_hybrid\sglang`。
- 目标基线：`dsv41-a5` / `3c333ac7099f95159fa471c7b8cc28f396cb6ce9`。
- 对照 ZBAL 源码：外层仓库 `br_fp4_dfc` / `9db36bfa`。
- 本次目标：通用 ModelSlim W8A8 INT8、离线 W8A8 MXFP8 E4M3、W4A4 MXFP4。
- 本文区分源码接通、CPU 契约验证和待完成的 NPU 验证；没有设备精度或性能结论。

## 1. 原有 ZBAL 接入流程

### 1.1 初始化、内存与通信

1. `managers/scheduler.py::init_zbal_on_npu` 在模型加载前调用
   `hardware_backend/npu/utils.py::init_zbal`。
2. `SGLANG_ZBAL_LOCAL_MEM_SIZE > 0` 启用 ZBAL，单位为 MiB；
   `SGLANG_ZBAL_BOOTSTRAP_URL` 指定 bootstrap 地址。
3. 非 mix 模式直接初始化 GVA；mix 模式先切换 allocator，权重/KV 仍使用 DMA VMM。
   `model_executor/model_runner_components/attention_backend_setup.py` 在 attention backend
   构造后调用 `lazy_init_zbal_gva_mem`，再进入图捕获。
4. `platforms/device_mixin.py` 将 NPU 分布式 backend 从 `hccl` 选为 `zbal`。
5. `layers/moe/token_dispatcher/deepep.py` 在 NPU 且 ZBAL 启用时导入
   `zbal.zbal_buffer.Buffer` 和 ZBAL 的 DeepEP-compatible Config。

上述初始化和分离 dispatch/combine 路径已经存在。本次复用它们，不改 ZBAL C++ ABI 或 kernel。
原有 lazy GVA 估算仍含 `total_memory = 61.2` GiB 的设备容量假设，部署时需核对设备规格和实际 GVA 预算；
该既有容量策略不属于本次算子适配变更。

### 1.2 MoE 和已有融合调用链

| 层次 | 调用链 | 覆盖范围 |
| --- | --- | --- |
| 分离通信 | `FusedMoE.forward_impl -> dispatcher.dispatch -> run_moe_core -> dispatcher.combine` | ZBAL 可替换 dispatch/combine；专家计算仍在 SGLang |
| 专家计算 | `quant_method.apply -> MoeRunner.run -> AscendRunnerCore.run` | GMM1、激活、GMM2 |
| 计算内融合 | `AscendRunnerCore -> apply_fused_gmm1_swiglu -> npu_grouped_matmul_swiglu_quant_v2` | 部分量化格式的 GMM1/SwiGLU/requant，通信仍分离 |
| runner 注册融合 | `FusedOpPool.get_fused_func(a2a_backend, runner_backend)` | 输入已是 dispatch output，不能直接放入整层 DFC |
| 整层融合 | `FusedMoE.forward -> forward_fuseep -> Buffer.fused_deep_moe` | 现有 Ascend FuseEP 入口，可承载 DFC 旁路 |

DFC 已包含 routing quant、dispatch、两次 GMM、SwiGLU/requant 和 combine。
将它放进 dispatcher 或 AscendRunnerCore 会重复执行通信或专家计算，因此接入整层 `forward_fuseep` 入口。

## 2. 适配方案和实际修改

### 2.1 选择规则

| 配置 | 路径 |
| --- | --- |
| `--moe-a2a-backend deepep`，ZBAL 启用 | 原有分离 dispatch / compute / combine |
| `--moe-a2a-backend ascend_fuseep`，ZBAL 未启用 | 原生 `fused_deep_moe` |
| `--moe-a2a-backend ascend_fuseep --fuseep-mode 2`，ZBAL 启用 | 新增 `dispatch_ffn_combine` |
| ZBAL 启用且选择 FuseEP mode 1 | 初始化报错，避免把另一种融合语义传给 DFC |

复用现有 backend 枚举，使模型的 EP 判定、TopK、shared expert 和外层 collective 路由保持一致。
当前分支将参数定义迁移到了 `arg_groups/fields/exec_.py`，帮助文字修改在该文件，未改旧位置。

### 2.2 权重加载

新增 `hardware_backend/npu/moe/zbal_dfc.py`，在
`NPUW8A8Int8MoEMethod`、`NPUMXFP8MoEMethod`、`NPUW4A4MXFP4MoEMethod`
的 `process_weights_after_loading` 开头分流，早于通用转置/NZ/scale 重排。

`process_zbal_dfc_weights` 只做 checkpoint 到 DFC 输入契约的离线布局转换，不重新量化权重。
按 `w13`/`w2` 分别标记处理完成，避免第二次加载后处理重复转换。

令 E 为本 rank 专家数，H 为 hidden size，D 为每专家 intermediate size；ModelSlim 的 W13 为 `[gate, up]`。

| 格式 | DFC W13 | DFC W2 | scale |
| --- | --- | --- | --- |
| INT8 | `[E,H,2D]`，NZ 物理格式 | `[E,D,H]`，NZ | FP32 位模式解释为 INT32，再数值扩展至 INT64；每输出通道一个值 |
| MXFP8 | `[E,2D,H]`，ND E4M3 | `[E,H,D]`，ND E4M3 | E8M0 原始 uint8 字节，逐输出行补齐 K tile |
| MXFP4 | `[E,2D,H/2]`，native FP4 x2 | `[E,H,D/2]`，native FP4 x2 | 同上，scale 数量从逻辑 K 计算 |

MX scale 从 `[E,N,K/32]` 补齐为 `[E,N,ceil(K/256)*8]`，再 flatten 最后两维。
例如 K=192，六个有效 scale 字节补齐为八个；FP4 的 payload 为 96 字节，scale 仍按 192 计算。
FP4 仅 `view(torch.float4_e2m1fn_x2)`，不交换 nibble、不做 NZ 转换。
INT8 在加载时完成 scale 打包，避免 forward 中触发 ZBAL wrapper 的 CPU/numpy 转换。

### 2.3 Buffer 生命周期和前向调用

`FusedMoE` 在创建权重之后、创建 runner 之前校验 DFC 支持的层语义。
`ModelRunner.init_attention_backends` 在 GVA 就绪后、任何 graph capture 之前创建并缓存 DFC Buffer。
当前分支 SSM backend 构造可能提前返回，新增初始化检查会补齐该路径的 lazy GVA 初始化。
正常路径不重复初始化 GVA。

Buffer 创建前检查原生 pybind 类确实导出 `dispatch_ffn_combine`。
仅检查 Python wrapper 不够：裁剪构建或旧 wheel 可能有 wrapper 而没有 native 算子。
进程组通过当前分支的 `get_parallel()` 获取，不引入已废弃的分布式 getter。

```text
模型 gate / TopK
  -> FusedMoE.forward
  -> forward_fuseep
  -> forward_zbal_dfc
  -> zbal.zbal_buffer.Buffer.dispatch_ffn_combine
  -> deepep adaptor Buffer::dispatch_ffn_combine
  -> zbal_dispatch_ffn_combine / ZBALOpDispatchFFNCombine
  -> dtype + scenario 对应的 launcher
  -> routing quant / dispatch -> GMM1 -> SwiGLU + requant -> GMM2 -> combine
```

前向保持 BF16/FP16 输入 dtype，将 TopK IDs 转为 INT32、路由权重转为 FP32，并保证连续存储。
共享专家和模型外层 scaling 仍由现有模型逻辑负责；不将 shared expert 塞入 DFC。

### 2.4 多 rank 容量、idle rank 和图执行

DFC 根据 `max_output_size` 推导远端 combine 地址，因此各 EP rank 必须使用同一个容量。

```text
C = max(1, 本次执行各 rank 公共输入行数上界)
max_output_size = C * moe_ep_size * top_k
```

- Eager：先运行当前分支的 `_prepare_eager_forward_batch`；有 DP 同步时读取补齐后的公共行数表，
  否则读取当前 batch 行数。不用 DeepEP 的固定 token 环境变量限制 prefill。
- Decode graph：使用公共 capture bucket；上下文放在 `run_once` 内，保留回调的 debug 执行也能重新进入。
  NPU graph runner 继承该路径，不需另写一套 NPU 捕获逻辑。
- Prefill graph：在统一 prefill forward context 及独立 multimodal dummy 路径设置阶段与容量。
- Base warmup：在实际 model callback 外设置对应阶段与容量。
- `require_mlp_tp_gather` 为 DFC 启用现有 DP 公共 bucket 机制，保留其新接口参数。
- idle rank：提交一行零输入和零路由权重参与通信/专家计算，完成后返回零行输出。
- 批次信息写入 `ForwardFlags` 的 graph-visible scoped 字段，退出或抛异常后恢复，避免跨 batch 泄漏。

## 3. 遇到的问题及处理

| 问题 | 原因 | 本次处理 |
| --- | --- | --- |
| 沿用分离算子入口会重复通信 | DFC 覆盖整层 | 使用现有 FuseEP 整层旁路 |
| 通用量化后处理破坏 DFC 权重 | INT8 与 MX 的布局契约不同 | 在三种 ModelSlim kernel 的加载入口提前分流 |
| FP4 的 K 与 K/2 混淆 | 存储字节数不是逻辑维度 | shape 和 scale 分别按 payload / logical K 处理 |
| 每 rank 用本地 token 数导致远端偏移不同 | workspace 地址依赖容量 | eager 公共上界、graph 公共 bucket |
| idle rank 提前返回会缺席通信 | DFC 要求所有 rank 参与 | dummy 零路由调用后截回空输出 |
| 按 token 数猜 prefill/decode | 单 token prefill、混合 DP、图 padding 均可能误判 | runner 显式传入场景 |
| SSM backend 提前返回 | 原有 GVA lazy init 未执行 | target runner 检查 GVA，再创建 Buffer |
| 旧 wheel 或裁剪构建缺 DFC | Python wrapper 不是 native 能力证明 | 图捕获前检查 native 方法并明确报错 |
| 旧参数路径和 getter 已迁移 | 新仓重构了参数和 runtime context | 修改实际字段文件，使用 `get_parallel()` |
| MoE draft 使用独立 speculative graph runner | 不走本次公共捕获入口 | 精确拒绝具有 DFC 权重标记的 draft；dense draft 保持原路径 |
| DSV4 W4A8 与普通 W4A4 不等价 | DFC 的 FP4 W1 选择 A4，且无 clamp ABI | 明确拒绝，未改 DSV4 专用实现 |

支持边界：EP 覆盖 WORLD，EP=1～16，PP=1、MoE TP=1；专家均分且连续放置，每 rank 1～128 个。
使用普通 gated SiLU、输出端路由权重；两组权重使用相同量化方案。
MX 路径要求 Ascend A5 architecture 35 和 native FP4 dtype（FP4 情况）。

初始化时拒绝：SBO/TBO、EPLB/动态放置/冗余专家、融合 shared expert、非零 bias/offset、
非 gated SiLU、alpha/beta/clamp、输入端 router weight、no-combine、DFC balance、
非 ModelSlim 量化类、在线 MXFP8、W4A8 或混合 W1/W2 格式。
`gate_up_interleaved` 在当前仓库默认 True，但普通 ModelSlim SiLU 仍加载 split gate/up；
没有据此错误拒绝默认模型，特殊 GPT-OSS 激活已由上述 alpha/clamp 限制排除。

原生 host 仍负责 MX 输入行尾空间等 kernel 形状限制，例如
`Align256(payloadBytes) + Align32(ceil(H/32)) <= payloadBytes + 512`。
常见维度不代表所有自定义模型形状均已验证。

## 4. 启用与验证

### 4.1 Linux / Ascend 环境

先使用目标 torch/torch_npu/CANN 环境构建并安装包含 DFC 的 ZBAL wheel；MX 路径需要对应 FKL 库。
不同 torch_npu 次版本分别构建 wheel。ZBAL 的构建脚本会清理其构建/输出目录。

```bash
cd /path/to/memfabric_hybrid/app/zbal
bash script/build.sh --release
python3 -m pip install --force-reinstall output/memfabric_zbal*.whl
python3 -c 'from zbal.zbal.deepep_adaptor import Buffer; print(hasattr(Buffer, "dispatch_ffn_combine"))'
```

在已安装本 checkout 的 SGLang 运行环境中，先关闭 prefill/decode graph 验证 eager：

```bash
export MODEL_PATH=/path/to/modelslim-checkpoint
export SGLANG_ZBAL_LOCAL_MEM_SIZE=YOUR_GVA_BUDGET_MIB
export SGLANG_ZBAL_BOOTSTRAP_URL=tcp://YOUR_BOOTSTRAP_HOST:24699
export ZBAL_DFC_BALANCE_TOKEN=0

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --device npu --quantization modelslim \
  --tp-size 8 --ep-size 8 \
  --attention-backend ascend \
  --moe-a2a-backend ascend_fuseep --moe-runner-backend ascend --fuseep-mode 2 \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph
```

将 GVA 预算和 bootstrap 主机占位值替换为实际部署参数。EP 数需按模型专家数和设备数配置。
新分支用 `--disable-prefill-cuda-graph` / `--disable-decode-cuda-graph` 分别控制图执行。
启用 attention DP 时使用当前参数 `--attn-dp-size`，保持 EP 覆盖 WORLD。

### 4.2 本地检查及证据边界

```bash
cd /path/to/memfabric_hybrid/sglang
python3 test/manual/test_zbal_dfc_runner.py -v
python3 test/manual/test_zbal_dfc.py -v
git diff --check
```

2026-11-04 本机结果：runner 契约测试 **7/7 通过**；13 个新增/修改 Python 文件的语法编译、
仓库采用的 Ruff 检查项及 `git diff --check` 通过。22 个张量契约用例在导入 PyTorch 阶段被阻止，
未执行到测试主体。

- runner 契约测试不依赖 torch/CANN；抽取并执行实际 runner 方法，覆盖初始化顺序、eager/DP/idle、
  decode 保留回调、prefill、warmup、scope 恢复和 MoE draft 拒绝。
- 张量契约测试需要 CPU PyTorch，验证实际权重 payload/scale 转换、dtype、容量和调用分流。
  NZ 操作和 native DFC 被 mock，因此即使通过，也不代表设备计算正确。
- 本次 Windows 环境运行张量测试时，应用控制阻止加载 `torch.dll`，报 `WinError 4551`；
  沙箱外重试同样失败。不能将此前其他 checkout 的测试结果计为本次通过。
- 本次未执行 CANN/NPU 构建、设备精度、图捕获/重放或性能测试。

### 4.3 设备验收矩阵

在相同模型权重、路由、输入和并行配置下，以原分离路径作为对照，依次验证：

1. INT8、MXFP8、MXFP4 各自的 BF16/FP16 输入，单 rank 与多 rank 的 eager prefill/decode。
2. 大 prefill、DP 不均衡、idle rank、graph bucket padding，以及 prefill/decode 连续切换。
3. 开启 decode graph，再开启 prefill graph，检查捕获、重放和内存稳定性。
4. shared expert 保持模型外层计算，比较最终输出或模型精度，确认没有重复 scaling。
5. 对 FP4 记录 logical K、payload bytes、E8M0 scale 布局以及设备参考精度；不要用性能输出代替正确性。
6. 正确性通过后，再用相同 shape/routing/timing 配置测量性能并记录中位数及失败尝试。

源码接通不等于该矩阵已通过。当前未提交或推送本次 SGLang 修改。
