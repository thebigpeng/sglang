#!/usr/bin/env bash
#set -Eeuo pipefail

# ============================================================
# SGLang Ascend 多机统一部署（非 PD 分离）
#
# 部署方式：
#   所有 NODE_IPS 中的机器启动同一个 SGLang 分布式服务
#
# 默认并行拓扑假设：
#   每节点 8 张 NPU
#   TP_SIZE = 8
#   DP_SIZE = NNODES
#   启用 DP Attention + DeepEP
#
# 重要：
#   TP/DP 参数语义、DeepEP 跨节点通信和 DSPARK 兼容性，
#   必须根据当前使用的 SGLang 分支进行验证。
# ============================================================

# ===== 1. CPU / System =====

if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
#sysctl -w kernel.sched_migration_cost_ns=50000

# ===== 2. Cleanup =====

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING || true

# 不要默认 kill 全机 Python/SGLang 进程，避免影响其他任务。
# 如需清理旧进程，请在确认进程归属后手动执行。

export ASCEND_LAUNCH_BLOCKING=1

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

#export PYTHONPATH="/mnt/share/l00850654/code/sglang/python:${PYTHONPATH:-}"
export PYTHONPATH="/home/p00801009/sglang/python:${PYTHONPATH:-}"

# zbal
unset PYTORCH_NPU_ALLOC_CONF
export SGLANG_ZBAL_LOCAL_MEM_SIZE=58368
# zbal if use mix alloc
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:34669"


# ===== 6. DSPARK: Optional =====
#
# 默认关闭推测解码，先验证统一服务和多机通信。
#
# 确认当前 SGLang 分支、模型及自定义算子支持 DSPARK 后，
# 启动时设置 ENABLE_DSPARK=1。
#
# 原 Decode 使用的 DSPARK 参数不保证能直接用于非 PD 模式。

ENABLE_DSPARK="${ENABLE_DSPARK:-0}"

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

export MODEL_PATH="/mnt/share/l00850654/weights/dsv4.1-1017"
#export MODEL_PATH="/mnt/share/weight/DeepSeek-V4.1-Flash-1017/"
SERVED_MODEL_NAME="dsv4"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

# ===== 8. Cluster Config =====

NUM_NPUS_PER_NODE=8

# 只修改这里增加/删除机器。
# 所有节点必须使用一致的 NODE_IPS 和 NODE_IFS 配置。
NODE_IPS=(
  "141.61.49.195"
)

NODE_IFS=(
  "lo"
)

PORT=30000
DIST_PORT=5567

# 保留原脚本的 DP Attention + DeepEP 设计。
#
# 默认拓扑假设：
#   每个 DP 副本对应一台机器
#   每个副本内部 TP=每台机器的 NPU 数量
#
# 如果当前分支要求全局 TP 或其他 TP/DP 组合，
# 必须按该分支的参数语义调整，不能只改这两个变量。

NNODES="${#NODE_IPS[@]}"

if (( NNODES < 1 )); then
  echo "ERROR: NODE_IPS must contain at least one IP" >&2
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
  --max-prefill-tokens 131072
  --chunked-prefill-size 131072
  --kv-cache-dtype auto

  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}'
  --reasoning-parser deepseek-v4
  --language-model-only
  --moe-a2a-backend deepep
  --deepep-mode auto
  --enable-metrics
  --cuda-graph-bs-decode 1 2 4 8 16 25
  #--moe-a2a-backend ascend_fuseep

)
# --moe-a2a-backend ascend_fuseep
# --moe-runner-backend ascend
# --fuseep-mode 2
# --disable-shared-experts-fusion


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

# 每台机器只启动一个 SGLang 分布式服务。
# 不启动独立 Prefill/Decode 进程，不配置 PD transfer backend。

set +e
python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
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
