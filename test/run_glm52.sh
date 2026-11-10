#!/usr/bin/env bash
# run_glm52_single.sh — GLM-5.2 (W8A8_MXFP8, ModelSlim) 单机 8 卡部署。
# 后台启动(setsid) + 等 /health 就绪后打印 DEPLOY_READY 返回，不占调用方。
# 说明: 本版不接 ZBAL DFC（不设 SGLANG_ZBAL_POOL_MB），走框架自带的 deepep a2a 路径。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/weights/GLM-5.2-W8A8C8-mxfp8}"
SERVED_NAME="${SERVED_NAME:-glm52}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_PREFILL="${MAX_PREFILL:-8192}"
# chunked-prefill-size 是「每条请求每步最多 prefill 多少 token」；要让 N 条 8k 请求一步吃完，
# 这个值要 >= 单条输入长度，同时 --max-prefill-tokens 要 >= N × 输入长度。
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
PID_FILE="${PID_DIR:-${SCRIPT_DIR}}/glm52_server.pid"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/glm52_$(date +%Y%m%d_%H%M%S).log"

# ===== 1. 环境 =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING || true
# DFC 开关: ENABLE_DFC=1 时整模型的 MoE 专家权重 + dispatch_ffn_combine 的常驻 workspace 全部
# 落在 ZBAL 对称池里（池外只剩注意力/embedding/KV/激活）。池大小 = 权重 + workspace：
#   每层每卡专家权重 = w13[32,4096,6144] + w2[32,6144,2048] + e8m0 scale ≈ 1.16 GiB，78 层 ≈ 90.5 GiB
#   workspace 的行数由 --max-prefill-tokens 精确决定（zbal_dfc._max_tokens，不再有独立 env）：
#   balance 打开时约 374 KiB/bound token，另加 32 个权重预取槽 ≈ 1.25 GiB。
#   例：MAX_PREFILL=4096 → ws ≈ 3.1 GiB → 池 ≈ 92 GiB；MAX_PREFILL=16384 → ws ≈ 7.9 GiB → 池 ≈ 98 GiB。
# 单 rank 池上限当前是固定常量 128 GiB（SmemLocalHbmSizeMax()/MF_SMEM_LOCAL_HBM_MAX_MB 已删），
# 下面导出 MF_SMEM_LOCAL_HBM_MAX_MB 只为零风险兼容旧分支，当前分支忽略它。
if [[ "${ENABLE_DFC:-0}" = "1" ]]; then
  # 当前分支的开关是 SGLANG_ZBAL_ENABLE_DFC(默认 False)；SGLANG_ZBAL_POOL_MB 已废弃(池大小由
  # zbal_fuseep._pool_bytes 按 moe_layers 精确算出)，保留导出只为零风险兼容旧脚本。
  export SGLANG_ZBAL_ENABLE_DFC=1
  # GLM-5.2: num_hidden_layers=78, first_k_dense_replace=3, moe_layer_freq=1 -> MoE 层 75(无 MTP)。
  # 这里故意多给 1 层(76)：加载期每层先按原始 ND 布局在池里 create_weights，repack 又要在池里
  # 分配一份新副本，第 N 层的 raw 与已 repack 的 N-1 层短暂共存，峰值比"权重+workspace"高一层。
  # 实测 moe_layers=75(池 90.12 GiB) 会在第 75 层 create_weights 处 NPU OOM，76(91.28 GiB) 正常。
  export SGLANG_ZBAL_DFC_MOE_LAYERS="${SGLANG_ZBAL_DFC_MOE_LAYERS:-76}"
  export SGLANG_ZBAL_POOL_MB="${ZBAL_POOL_MB:-95232}"
  export MF_SMEM_LOCAL_HBM_MAX_MB="${MF_SMEM_LOCAL_HBM_MAX_MB:-131072}"
  export ZBAL_DFC_BALANCE_TOKEN="${ZBAL_DFC_BALANCE_TOKEN:-256}"
  unset SGLANG_ZBAL_LOCAL_MEM_SIZE ZBAL_NPU_ALLOC_CONF || true
  DFC_BANNER=" + zbal DFC (moe_layers=${SGLANG_ZBAL_DFC_MOE_LAYERS}, bound=${MAX_PREFILL} tok from --max-prefill-tokens, balance ${ZBAL_DFC_BALANCE_TOKEN}; weights+ws in pool)"
else
  unset SGLANG_ZBAL_ENABLE_DFC SGLANG_ZBAL_DFC_MOE_LAYERS
  unset SGLANG_ZBAL_POOL_MB SGLANG_ZBAL_LOCAL_MEM_SIZE MF_SMEM_LOCAL_HBM_MAX_MB \
    ZBAL_DFC_BALANCE_TOKEN ZBAL_NPU_ALLOC_CONF || true
fi

# CANN 的 set_env.sh 里有未定义变量引用(ZSH_VERSION 等)，必须在 set -u 关闭时 source。
set +u
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# DFC 全量入池需要 >64 GiB 的对称池，只有本分支重编的 libmf_smem 支持（旧包仍是 64 GiB 常量）。
# zbal 在 MEMFABRIC_HYBRID_LIBRARY_PATH 未设时会退回 pip 装的 memfabric_hybrid 包里的库，因此显式指过去。
# 注意这个变量是 **目录**（zbal 会 realpath 该目录再拼 /libmf_smem.so），不是 .so 全路径。
# libmf_smem 依赖 libmf_hybm_core 且自身没有 RPATH，两个目录都要写进 LD_LIBRARY_PATH。
MF_BUILD_OUT="${MF_BUILD_OUT:-/home/w00802660/memfabric_hybrid/output}"
if [[ -f "${MF_BUILD_OUT}/smem/lib64/libmf_smem.so" ]]; then
  export MEMFABRIC_HYBRID_LIBRARY_PATH="${MEMFABRIC_HYBRID_LIBRARY_PATH:-${MF_BUILD_OUT}/smem/lib64}"
  export LD_LIBRARY_PATH="${MF_BUILD_OUT}/smem/lib64:${MF_BUILD_OUT}/hybm/lib64:${LD_LIBRARY_PATH:-}"
elif [[ "${ENABLE_DFC:-0}" = "1" ]]; then
  echo "ERROR: ENABLE_DFC=1 needs the >64GiB capable libmf_smem, none found under ${MF_BUILD_OUT}" >&2
  exit 1
fi
echo "memfabric lib dir: ${MEMFABRIC_HYBRID_LIBRARY_PATH:-<pip package default>}"

# zbal 建通信子走 SMA 静态内存，要求不开 expandable_segments（否则 zbal_mem_allocator.cpp:129 断言）；
# 可用 PYTORCH_NPU_ALLOC_CONF="" 覆盖（置空即不带选项）。
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF-expandable_segments:True}"
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export TRANSFORMERS_VERBOSITY=error
# NPU graph 捕获不支持 TASK_QUEUE_ENABLE=2（torch_npu 直接抛
# "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture"），入图时必须 1/0。
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-400}"
export DEEPEP_HCCL_BUFFSIZE="${DEEPEP_HCCL_BUFFSIZE:-1800}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=150
# MXFP8 权重不能按 int8 做 dispatch 量化(否则激活 scale 形状与融合 GMM 契约不符)
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-0}"
export DEEP_USE_ALLTOALL_MODE=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-data0.3001}"
# 本镜像 torch 2.13 + torch_npu 2.13 的 dynamo 规则表冲突，导入期 @torch.compile 会抛
# Duplicate torch object，导致模型注册表扫不全；本部署不用 torch.compile。
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="/home/w00802660/sglang/python:${PYTHONPATH:-}"

[[ -d "$MODEL_PATH" ]] || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }

ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --device npu
  --attention-backend ascend
  --tp-size "$TP_SIZE" --nnodes 1 --node-rank 0
  --trust-remote-code
  --quantization modelslim
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests 64
  --chunked-prefill-size "${CHUNKED_PREFILL:-4096}"
  --max-prefill-tokens "$MAX_PREFILL"
  --cuda-graph-bs-decode ${CUDA_GRAPH_BS_DECODE:-1 2 4 8 16}
  --moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE:-normal}"
  --attn-dp-size "${ATTN_DP_SIZE:-1}"
  --enable-metrics
)

# decode 图对照臂用：显式给 backend（full/breakable/disabled），缺省不传，走框架默认(full)。
if [[ -n "${CUDA_GRAPH_BACKEND_DECODE:-}" ]]; then
  ARGS+=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
fi

# FuseEP 的权重 ABI 只覆盖 routed experts，共享专家必须走独立通路（zbal_fuseep.prepare_weights
# 见到 num_fused_shared_experts != 0 直接 raise）。GLM-5.2 n_shared_experts=1。
if [[ "${ENABLE_DFC:-0}" = "1" ]]; then
  ARGS+=(--disable-shared-experts-fusion)
fi

echo "============================================================"
echo "GLM-5.2 single-node deployment"
echo "model      : $MODEL_PATH"
echo "tp         : $TP_SIZE   attn-dp: ${ATTN_DP_SIZE:-1}   mem-fraction: $MEM_FRACTION"
echo "port       : $PORT      max-prefill: $MAX_PREFILL   chunked-prefill: ${CHUNKED_PREFILL:-4096}"
echo "a2a        : deepep${DFC_BANNER:-}"
echo "============================================================"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: server already running (pid $(cat "$PID_FILE")); stop it first" >&2
  exit 1
fi
rm -f "$PID_FILE"

cd /home/w00802660/sglang 2>/dev/null || true
setsid bash -c 'echo $$ > "$0"; exec python3 -m sglang.launch_server "$@"' \
  "$PID_FILE" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &

for _ in $(seq 1 30); do
  [[ -s "$PID_FILE" ]] && break
  sleep 1
done
SERVER_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: launch_server did not start (pid='${SERVER_PID}')" >&2
  tail -n 40 "$LOG_FILE" >&2
  exit 1
fi
echo "Launched pid=$SERVER_PID log=$LOG_FILE"

waited=0
while (( waited < HEALTH_TIMEOUT )); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited during startup (pid $SERVER_PID)" >&2
    tail -n 40 "$LOG_FILE" >&2
    exit 1
  fi
  if python3 -c "import sys,urllib.request;urllib.request.urlopen(sys.argv[1],timeout=5)" \
      "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "DEPLOY_READY pid=$SERVER_PID port=$PORT log=$LOG_FILE"
    exit 0
  fi
  sleep 10
  waited=$((waited + 10))
done

echo "ERROR: health not ready after ${HEALTH_TIMEOUT}s (pid $SERVER_PID)" >&2
tail -n 40 "$LOG_FILE" >&2
exit 1