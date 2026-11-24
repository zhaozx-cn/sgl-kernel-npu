# causal_conv1d

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 与 `ascendc` 使用同一入口。

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d

def causal_conv1d(
    x, weight, conv_states,
    bias=None, query_start_loc=None, cache_indices=None,
    has_initial_state=None, num_accepted_tokens=None,
    activation_mode=0, pad_slot_id=-1, run_mode=0,
    *, output_layout="TOKEN_FIRST", head_num=0, impl="auto",
):
    ...
```

## 功能说明

按通道进行因果一维卷积，可选加偏置和 SiLU，支持 prefill 与集成 update。
设 `Z` 为 `K-1` 行历史与当前输入按时间顺序的拼接：

```text
u[t,d] = sum(Z[t+j,d] * weight[j,d], j=0..K-1) + bias[d]
y[t,d] = u[t,d]                              activation_mode == 0
         u[t,d] / (1 + exp(-u[t,d]))          其他情况
```

缺省 bias 时省略偏置项。返回值转换为 `x` 的 dtype，`conv_states` 原地更新，
保存原始输入和历史，不保存卷积或激活结果。

## 参数与返回值

`B` 为序列数，`L` 为共同序列长度，`T` 为拼接后的 token 数，`D` 为通道数，
`K` 为卷积宽度，`S` 为每个缓存槽的状态容量。输入采用 token 在前的布局。

| 参数 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `x` | `[T,D]` 或 `[B,L,D]` | 连续 BF16 或 FP16。 |
| `weight` | `[K,D]` | 连续，dtype 与 `x` 相同；`K` 为 2、3 或 4。 |
| `conv_states` | `[slots,S,D]` | 连续，dtype 与 `x` 相同；原地回写状态。 |
| `bias` | `[D]` 或 `None` | 可选通道偏置，dtype 与 `x` 相同，默认 `None`。 |
| `query_start_loc` | `[B+1]` 或 `None` | 拼接序列边界，转换为 int64。2D 输入必填；3D 输入忽略。 |
| `cache_indices` | `[B]` 或 `None` | 序列到缓存槽的映射，转换为 int64；默认序列 `b` 使用槽 `b`。 |
| `has_initial_state` | `[B]` 或 `None` | 转为 int64 后非零时 prefill 读取历史；默认不读取。update 忽略。 |
| `num_accepted_tokens` | `[B]` 或 `None` | 转为 int32，仅在 update 且 `K == 4` 时选择推测更新的历史起点。 |
| `activation_mode` | `int`，默认 `0` | 任意非零值应用 SiLU。 |
| `pad_slot_id` | `int`，默认 `-1` | 显式 cache index 等于此值时跳过序列。 |
| `run_mode` | `int`，默认 `0` | `1` 选择集成 update，其他值选择 prefill。 |
| `impl` | `str`，默认 `"auto"` | 仅限关键字，可选 `auto`、`ascendc`、`torch_native`。 |

| 返回值 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `y` | 与 `x` 相同 | 与 `x` 相同 | 新分配输出；状态通过 `conv_states` 回写。 |

空的可选 Tensor 按缺省处理；2D 输入仍必须提供有效序列边界。索引、边界和缓存容量
必须有效，使用的槽至少容纳 `K-1` 行历史；不同序列应使用独立缓存槽。
pad 序列不修改状态，输出行未初始化；默认槽映射不检查 `pad_slot_id`。

## 状态更新

- prefill：初始状态标志非零时读取 `state[slot,:K-1]`，否则使用零历史；
  将拼接后最后 `K-1` 行写回 state 的前 `K-1` 行。
- 普通 update：始终读取已有历史，写回规则与 prefill 相同。
- update 且 accepted 非空、`K == 4`：令 `o = min(max(accepted[b]-1,0),S-3)`，
  读取旧 `state[slot,o:o+3]`。若 `S >= 2+L`，将
  `[旧state[o+1],旧state[o+2],X]` 写入前 `2+L` 行，否则采用普通写回。
  accepted 不截断计算长度；`K == 2/3` 使用普通 update。

未写入的状态行和未使用槽保持原值。update 由本算子内部实现。

## ascendc 后端

- 要求所有 Tensor 位于同一 NPU，`x`、`weight` 和 `conv_states` 连续。
- 可选 bias 和元数据按 dtype 转换后的物理连续区间读取，不按逻辑 stride 重排；
  调用方应提供容量充足的连续缓冲。

## torch_native 后端

- 使用独立 PyTorch 运算，以 FP32 累加，再处理 bias、SiLU 和输出类型转换。
- 保持相同的历史选择、状态回写和可选参数语义。

## 调用示例

沿用现有 BF16 prefill 配置及其对应的普通 update 配置，两后端各自维护状态池。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d

x = torch.randn(2, 6, 4096, device="npu", dtype=torch.bfloat16)
weight = torch.randn(4, 4096, device="npu", dtype=x.dtype)
state = torch.randn(8, 3, 4096, device="npu", dtype=x.dtype)
state_ref = state.clone()
inputs = dict(
    x=x, weight=weight,
    cache_indices=torch.tensor([0, 1], device="npu", dtype=torch.int64),
    has_initial_state=torch.tensor([False, False], device="npu"),
)
y = causal_conv1d(**inputs, conv_states=state)  # auto: ascendc, prefill
y_ref = causal_conv1d(**inputs, conv_states=state_ref, impl="torch_native")
print(y.shape, y_ref.shape)  # torch.Size([2, 6, 4096])

# Integrated update starts from the state written by each backend's prefill.
y_update = causal_conv1d(**inputs, conv_states=state, run_mode=1)
y_update_ref = causal_conv1d(
    **inputs, conv_states=state_ref, run_mode=1, impl="torch_native"
)
print(y_update.shape, y_update_ref.shape)
```

## Prefill 输出布局

默认 `output_layout="TOKEN_FIRST", head_num=0` 保持已有输出形状与行为，
兼容原有 prefill、decode 和投机验证调用。

仅 `run_mode=0` 可以设置 `output_layout="HEAD_FIRST", head_num=H`：
`[T,D]` 输入返回连续的 `[H,T,D/H]`，`[B,L,D]` 输入返回连续的
`[B,H,L,D/H]`。H 必须为正且整除 D，D/H 必须是 16 的倍数。
非法布局、head 参数和 head-first update 请求在修改状态之前报错。
AscendC kernel 直接按目标布局写回，不额外启动转置算子。

输入、权重、bias、序列边界和 conv_states 的布局不变，卷积数值、激活与
状态更新语义不变。变长序列共享完整的 T 轴；普通 batch 各有独立的
[H,L,D/H] 存储段。padding 对应的输出元素仍未定义。

GDN/KDA 的 Q/K/V head_dim 相同时，可以用本地 TP 的 Hq+Hk+Hv 作为 H，
将 packed 输出按第 0 维切成连续的 Q/K/V 视图。head_dim 不一致时使用默认路径。
B>1 的普通 batch 按 head 维切分后不保证各分量整体连续。
本接口不自动改变下游 gate 的布局，也不自动接入模型。
