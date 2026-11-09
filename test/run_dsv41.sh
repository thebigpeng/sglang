#!/usr/bin/env bash
#set -Eeuo pipefail

# ============================================================
# SGLang DSV4.1 单机 8 卡部署（非 PD 分离）
#
# 部署方式：
#   默认使用 ZBAL 完整 DFC；可切换 deepep 做非融合对照。
#   额外 SGLang 参数可追加在脚本后，例如：
#   bash test/run_dsv41.sh --disable-decode-cuda-graph
#
# 默认并行拓扑假设：
#   每节点 8 张 NPU
#   TP_SIZE = 8
#   DP_SIZE = 8
#   启用 DP Attention，MoE EP 覆盖 WORLD。
#
# 重要：
#   当前固定 TP/DP 配置仅用于单机；多机需要另行核对拓扑和 bootstrap。
#   ZBAL DFC 当前不支持 DSPARK。
# ============================================================

# ===== 1. CPU / System =====

if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
#sysctl -w kernel.sched_migration_cost_ns=50000

# ===== 2. Cleanup =====

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY || true

# 不要默认 kill 全机 Python/SGLang 进程，避免影响其他任务。
# 如需清理旧进程，请在确认进程归属后手动执行。

# Set ASCEND_LAUNCH_BLOCKING=1 explicitly when diagnosing asynchronous failures.

# ===== 3. Ascend Environment =====

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export TRANSFORMERS_VERBOSITY=error

export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-data0.3001}"

export SGLANG_SET_CPU_AFFINITY=1
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

export HCCL_BUFFSIZE=400
export DEEPEP_HCCL_BUFFSIZE=1800

# ===== 4. Custom Operator Environment =====

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 保留原脚本的 MemFabric 环境初始化。
# 不再配置 PD 专用的 device_urma 传输协议及 Store URL。
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then
  source /usr/local/memfabric_hybrid/set_env.sh
fi

# ===== 5. Model / SGLang Environment =====

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=150


# 保留原脚本中的性能/算子开关。
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_DSV4_FP4_DEQUANT=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export SGLANG_DSV4_NATIVE_ATTN=0
export SGLANG_DSV4_DBG_ATTN=1
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0
export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1
export SGLANG_OPT_USE_HC_TRITON=0
export SGLANG_OPT_USE_HC_TILELANG=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
if [[ ! -f "$SGLANG_ROOT/python/sglang/srt/hardware_backend/npu/moe/zbal_dfc.py" ]]; then
  echo "ERROR: missing ZBAL DFC integration under SGLANG_ROOT=$SGLANG_ROOT" >&2
  exit 1
fi
export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"

# zbal
unset PYTORCH_NPU_ALLOC_CONF
# MiB; budget weights, KV cache and DFC workspace together on the target device.
export MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-ascend_fuseep}"
if [[ "$MOE_A2A_BACKEND" == ascend_fuseep ]]; then
  export SGLANG_ZBAL_LOCAL_MEM_SIZE="${SGLANG_ZBAL_LOCAL_MEM_SIZE:-58368}"
else
  export SGLANG_ZBAL_LOCAL_MEM_SIZE="${SGLANG_ZBAL_LOCAL_MEM_SIZE:-0}"
fi
export ZBAL_DFC_BALANCE_TOKEN=0
if [[ "$MOE_A2A_BACKEND" == ascend_fuseep && ! "$SGLANG_ZBAL_LOCAL_MEM_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ZBAL DFC requires positive SGLANG_ZBAL_LOCAL_MEM_SIZE (MiB)" >&2
  exit 1
fi
# zbal if use mix alloc
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export SGLANG_ZBAL_BOOTSTRAP_URL="${SGLANG_ZBAL_BOOTSTRAP_URL:-tcp://127.0.0.1:34669}"


# ===== 6. DSPARK: Optional =====
#
# 默认关闭推测解码，先验证统一服务和多机通信。
#
# 确认当前 SGLang 分支、模型及自定义算子支持 DSPARK 后，
# 启动时设置 ENABLE_DSPARK=1。
#
# 原 Decode 使用的 DSPARK 参数不保证能直接用于非 PD 模式。

ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
if [[ "$MOE_A2A_BACKEND" == ascend_fuseep && "$ENABLE_DSPARK" == 1 ]]; then
  echo "ERROR: ZBAL DFC does not support DSPARK; MOE_A2A_BACKEND=$MOE_A2A_BACKEND ENABLE_DSPARK=$ENABLE_DSPARK" >&2
  exit 1
fi

export SGLANG_DSV4_GRAPH_MAX_SEQ=131072

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  export SGLANG_ENABLE_SPEC_V2=1
  export SGLANG_RAGGED_VERIFY_MODE=static
  export SGLANG_DSPARK_FAST_KERNEL=0

  #DSPARK_SO="/mnt/share/l00850654/package/extra-ops/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
  #DSPARK_OPP="/mnt/share/l00850654/package/extra-ops/vendors/custom_transformer/"
  #DSPARK_LIB="/mnt/share/l00850654/package/extra-ops/vendors/custom_transformer/op_api/lib"

  #if [[ ! -f "$DSPARK_SO" ]]; then
  #  echo "ERROR: DSPARK custom op not found: $DSPARK_SO" >&2
  #  exit 1
  #fi

  #export SGLANG_DSPARK_A5_EXTRA_OPS_SO="$DSPARK_SO"
  #export ASCEND_CUSTOM_OPP_PATH="${DSPARK_OPP}:${ASCEND_CUSTOM_OPP_PATH:-}"
  #export LD_LIBRARY_PATH="${DSPARK_LIB}:${LD_LIBRARY_PATH:-}"
fi

# ===== 7. Model Config =====

export MODEL_PATH="${MODEL_PATH:-/mnt/share/l00850654/weights/dsv4.1-1017}"
#export MODEL_PATH="/mnt/share/weight/DeepSeek-V4.1-Flash-1017/"
SERVED_MODEL_NAME="dsv4"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

# ===== 8. Cluster Config =====

NUM_NPUS_PER_NODE=8

# 按当前服务器设置 IP 和网卡；本脚本的固定 TP/DP 配置仅用于单机。
NODE_IPS=(
  "141.61.49.195"
)

NODE_IFS=(
  "lo"
)

PORT=30000
DIST_PORT=5567

# 单机 TP8/DP8，MoE EP 由后端设置为 TP size；不能仅增加 NODE_IPS 扩展成多机。

NNODES="${#NODE_IPS[@]}"

if (( NNODES < 1 )); then
  echo "ERROR: NODE_IPS must contain at least one IP" >&2
  exit 1
fi

if [[ "$MOE_A2A_BACKEND" == ascend_fuseep ]] && (( NNODES != 1 )); then
  echo "ERROR: this DFC launcher requires one node with TP8/DP8; got NNODES=$NNODES" >&2
  exit 1
fi

if (( ${#NODE_IFS[@]} != NNODES )); then
  echo "ERROR: NODE_IFS count must equal NODE_IPS count" >&2
  exit 1
fi

TOTAL_NPUS=$((NNODES * NUM_NPUS_PER_NODE))

DP_SIZE=8
TP_SIZE=8
#TP_SIZE=$TOTAL_NPUS

MASTER_ADDR="${NODE_IPS[0]}"
DIST_INIT_ADDR="${MASTER_ADDR}:${DIST_PORT}"

# ===== 9. Detect Local Node =====

LOCAL_IPS="$(hostname -I 2>/dev/null || true)"
NODE_RANK=-1
HCCL_IF=""

for i in "${!NODE_IPS[@]}"; do
  for ip in $LOCAL_IPS; do
    if [[ "$ip" == "${NODE_IPS[$i]}" ]]; then
      NODE_RANK="$i"
      HCCL_IF="${NODE_IFS[$i]}"
      break 2
    fi
  done
done

if (( NODE_RANK < 0 )); then
  echo "ERROR: local IP [$LOCAL_IPS] not found in NODE_IPS" >&2
  exit 1
fi

export HCCL_SOCKET_IFNAME="$HCCL_IF"

echo "============================================================"
echo "SGLang Ascend unified deployment"
echo "Deployment mode : non-PD"
echo "Local IPs       : $LOCAL_IPS"
echo "Node rank       : $NODE_RANK"
echo "NNODES          : $NNODES"
echo "NPU/node        : $NUM_NPUS_PER_NODE"
echo "Total NPUs      : $TOTAL_NPUS"
echo "TP_SIZE         : $TP_SIZE"
echo "DP_SIZE         : $DP_SIZE"
echo "Master          : $DIST_INIT_ADDR"
echo "HCCL interface  : $HCCL_SOCKET_IFNAME"
echo "GLOO interface  : $GLOO_SOCKET_IFNAME"
echo "DSPARK          : $ENABLE_DSPARK"
echo "============================================================"

# ===== 10. Build Launch Arguments =====

LAUNCH_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"

  --tp-size "$TP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --dist-init-addr "$DIST_INIT_ADDR"

  --page-size 128
  --trust-remote-code
  --attention-backend dsv4
  --device npu

  --enable-dp-attention
  --dp-size "$DP_SIZE"
  --enable-dp-lm-head

  --watchdog-timeout 9000
  --max-running-requests 200
  --mem-fraction-static 0.83
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-2048}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
  --kv-cache-dtype auto

  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}'
  --reasoning-parser deepseek-v4
  --language-model-only
  --moe-a2a-backend "$MOE_A2A_BACKEND"
  --moe-runner-backend auto
  --fuseep-mode 2
  --disable-shared-experts-fusion
  --enable-metrics
  --cuda-graph-bs-decode 1 2 4 8 16 25
)
#--language-model-only
#--json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}'
#--disable-cuda-graph
#--cuda-graph-bs-decode 1 2 4 6 8 16 24
# ===== 11. Optional DSPARK Arguments =====
#
# 这些参数来自原 Decode 配置。
# 必须确认它们与当前非 PD SGLang 启动路径兼容。

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  LAUNCH_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-attention-backend ascend
    --speculative-num-draft-tokens 6
    --speculative-dspark-block-size 5
  )
fi

# ===== 12. Launch Unified Service =====

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/sglang_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $LOG_FILE"
if ! python3 - >"$LOG_FILE" 2>&1 <<'PY'
import inspect
import json
import os
from pathlib import Path

import sglang
from sglang.srt.configs.deepseek_v41 import normalize_deepseek_v41_config

config_path = Path(os.environ["MODEL_PATH"]) / "config.json"
config = normalize_deepseek_v41_config(json.loads(config_path.read_text(encoding="utf-8")))
print("SGLang:", sglang.__file__)
print("Model config:", config_path)
print("MoE backend:", os.environ["MOE_A2A_BACKEND"])
print("Model MoE parameters:", {name: config.get(name) for name in (
    "hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "swiglu_limit",
)})
if os.environ["MOE_A2A_BACKEND"] == "ascend_fuseep":
    from zbal.zbal_buffer import Buffer
    from zbal.zbal.deepep_adaptor import Buffer as NativeBuffer
    from sglang.srt.hardware_backend.npu.moe import zbal_dfc

    print("ZBAL DFC integration:", zbal_dfc.__file__)
    print("ZBAL Buffer:", inspect.getfile(Buffer))
    print("DFC signature:", inspect.signature(Buffer.dispatch_ffn_combine))
    if "swiglu_limit" not in inspect.signature(Buffer.dispatch_ffn_combine).parameters:
        raise RuntimeError("Rebuild/install ZBAL with DSV4 W4A8 swiglu_limit support")
    if not callable(getattr(NativeBuffer, "dispatch_ffn_combine", None)):
        raise RuntimeError("Installed native ZBAL Buffer has no dispatch_ffn_combine")
PY
then
  cat "$LOG_FILE"
  echo "ERROR: preflight failed; MODEL_PATH=$MODEL_PATH MOE_A2A_BACKEND=$MOE_A2A_BACKEND SGLANG_ROOT=$SGLANG_ROOT" >&2
  exit 1
fi
cat "$LOG_FILE"

# 每台机器只启动一个 SGLang 分布式服务。
# 不启动独立 Prefill/Decode 进程，不配置 PD transfer backend。

set +e
python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
PIPE_STATUS=( "${PIPESTATUS[@]}" )
set -e

SERVER_RC="${PIPE_STATUS[0]:-1}"
TEE_RC="${PIPE_STATUS[1]:-1}"

if (( SERVER_RC != 0 )); then
  echo "ERROR: SGLang exited with code $SERVER_RC" >&2
  exit "$SERVER_RC"
fi

if (( TEE_RC != 0 )); then
  echo "ERROR: tee exited with code $TEE_RC" >&2
  exit "$TEE_RC"
fi
