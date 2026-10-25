# tool-loop

[![tests](https://github.com/tiantianw/tool-loop/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/tiantianw/tool-loop/actions/workflows/test.yml)

一个能一口气读完的 tool-calling agent 循环。[English](README.md) · **中文**

大多数 agent 框架把最关键的部分藏了起来。装一个,你拿到的是一个循环,但失败处理散落在十几个模块里,背后的判断依据完全不可见。这个仓库是反过来的:一个循环、三个只读工具、两个依赖,以及一份把"工具出错时到底发生什么"讲清楚的 README。

循环在 `agent.py`,约 280 行,其中三分之一是注释——解释的是**为什么**,不是**做了什么**。

```
python demo.py        # 不需要 API key,不联网
pytest                # 36 个测试
```

---

## 不需要 API key 就能跑

```bash
pip install httpx
python demo.py
```

demo 把模型换成了脚本,所以完全离线、每次输出都一样。它会故意踩四个坑,然后一个个爬出来:

```
step 1: read_file  src/agent.py   -> not_found: no such file
step 2: list_dir   src            -> not_found: no such directory
step 3: list_dir   .              -> agent.py, llm.py, tools.py, ...
step 4: grep       "class ToolError["  -> invalid_argument: unterminated character set
step 5: grep       "class ToolError"   -> agent.py:35: class ToolError(Exception):
step 6: read_file  agent.py       -> lines 1-200 of 277 (截断)
step 7: 回答
```

第 1、2 步是最好看的一对:第 1 步返回的 hint 指向 `src`,而 `src` 也不存在,模型必须再退一级。**一条 hint 不等于一次修复**,循环没有假装它是。

完整 trace 见 [examples/transcript.md](examples/transcript.md)。

## 唯一重要的那个判断

> **工具失败是数据,不是异常。**

模型能修错路径、能修写错的正则、能修类型不对的参数。但它修不了 `429`,也修不了过期的 key。这是两类不同的问题,属于两个不同的层:

- **传输层失败**(429、5xx、超时)在 `llm.py` 里做指数退避加 jitter 重试,循环层根本看不到。重试耗尽就抛致命错误,整个运行停下。
- **工具层失败**(文件不存在、正则非法、参数错误)被序列化成工具结果回注给模型,运行继续,由模型自己决定重试、换路子,还是认输。

如果工具错误是用 `raise` 抛出去的,`tests/test_recovery.py` 里没有一个用例能跑完。这就是这个设计的全部理由。

## 循环长什么样

```
                 ┌──────────────────┐
                 │  构建 messages   │◄──────────────┐
                 └────────┬─────────┘               │
                          ▼                         │
                 ┌──────────────────┐               │
                 │     调用模型     │ 429/5xx 重试  │
                 └────────┬─────────┘ 在 llm.py 里  │
                          ▼                         │
                 ┌──────────────────┐               │
                 │   解析 response  │               │
                 └────┬────────┬────┘               │
          无 tool_calls│        │有 tool_calls      │
                      ▼        ▼                    │
              ┌──────────┐  ┌──────────────────┐    │
              │   回答   │  │    执行工具      │────┘
              └──────────┘  │ 错误 → 结果      │
                            └──────────────────┘
```

| 文件 | 行数 | 内容 |
|---|---|---|
| `agent.py` | ~280 | 主循环、`ToolRegistry`、参数校验、循环层防护 |
| `llm.py` | ~190 | HTTP 传输、重试策略、`ScriptedLLM` |
| `tools.py` | ~220 | `list_dir`、`read_file`、`grep`,以及它们的失败模式 |
| `main.py` | ~50 | 命令行入口 |
| `demo.py` | ~110 | 脚本化运行与 trace 渲染 |

## 失败处理

| 层 | 失败 | 谁处理 | 结果 |
|---|---|---|---|
| 传输 | 429、5xx、连接重置、超时 | `llm.py` | 退避 + jitter 重试,最多 3 次 |
| 传输 | 401、400、模型名错 | `llm.py` | 不重试,立即 `FatalError` |
| 传输 | 响应体不是合法 JSON | `llm.py` | 视为瞬时错误,重试 |
| 协议 | `tool_calls` 里某一项没有 name | `llm.py` | 视为瞬时错误,重试 |
| 工具 | 参数不是合法 JSON | `ToolRegistry` | `bad_json`,提示不要包代码块 |
| 工具 | 缺字段或类型错 | `ToolRegistry` | `invalid_arguments`,**点名是哪个字段** |
| 工具 | 工具名不存在 | `ToolRegistry` | `unknown_tool`,列出真实存在的工具 |
| 工具 | 文件或目录不存在 | `tools.py` | `not_found`,提示该去列哪个目录 |
| 工具 | 路径越出工作区 | `tools.py` | `invalid_argument`,碰盘之前就拒绝 |
| 工具 | 二进制文件 | `tools.py` | `invalid_argument`,建议改用 `grep` |
| 工具 | 文件过大 | `tools.py` | 截断,但 header 报告真实行数 |
| 工具 | 正则编译失败 | `tools.py` | `invalid_argument`,带上 `re` 的原始信息 |
| 工具 | 没有匹配 | `tools.py` | 正常结果,不是错误 |
| 工具 | 工具自己崩了 | `ToolRegistry` | `tool_crashed`,"重复调用没有用" |
| 循环 | 完全相同的调用重复出现 | `agent.py` | 第 3 次停止运行,要求总结 |
| 循环 | 步数上限 | `agent.py` | 强制收尾,不给工具 |
| 循环 | token 预算耗尽 | `agent.py` | 强制收尾,不给工具 |

两个容易被忽略、但值得单独说的细节:

**校验要报出字段名。** `"'start_line' must be integer, got str"` 一轮就能改对。`"invalid arguments"` 改不对——模型只能猜。

**收尾要没收工具。** `_finalize()` 传的是 `tools=None`,而不是"请不要再调用工具了"。工具调用在物理上变得不可能,回答就成了唯一剩下的动作。

## 四个决策,以及各自的代价

**不用框架。** 没有 LangChain,没有 LlamaIndex。*代价:* 没有流式、没有记忆、没有多 agent、没有可视化追踪。*理由:* 这些是循环正确之后可以往上加的东西,而循环本身才是通常出错的地方。

**手写参数校验。** 约 30 行,覆盖必填字段和基本类型,不引 JSON Schema 库。*代价:* 不支持 `oneOf`、不支持 `format`、不校验嵌套对象。*理由:* 模型只犯两种错——漏字段和类型错,两种都覆盖了;而且错误信息可以写给模型看,而不是写给日志看。

**脚本化的模型是一等公民。** `ScriptedLLM` 不是最后补上的测试夹具,而是 demo 和全部 36 个测试的运行方式。*代价:* 行为变了,剧本要跟着改。*理由:* 一个失败如果复现一次要花钱,你就不会去测它。

**只读工具。** 只有 `list_dir`、`read_file`、`grep`。*代价:* agent 改不了任何东西,所以它不是一个 coding agent。*理由:* 一个能写文件的公开仓库,就是某个人会拿去对着自己主目录跑的仓库。只读同样能造出所有有意思的失败——路径不存在、权限、超大文件、编码、坏正则——但爆炸半径小得多。

## 接真实模型

任何 OpenAI 兼容的端点都能用:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini          # 可选
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可选

python main.py "哪个模块负责重试,为什么?" --root .
```

代码用 `httpx` 直接打 `/chat/completions`,不用 SDK,所以请求形状没有任何东西挡着,也没有绑定任何厂商。

## 测试

```
pytest -q
36 passed
```

| 文件 | 覆盖 |
|---|---|
| `tests/test_loop.py` | 终止条件、步数上限、重复调用检测、token 预算、消息历史形状 |
| `tests/test_failures.py` | 每一种错误载荷,以及重试、退避增长、401 与 429 的区别 |
| `tests/test_recovery.py` | 模型出错、读到错误、然后自己改正 |

`tests/test_failures.py` 替换了 `httpx.Client.post`,所以重试行为可以在不联网、不真睡的情况下被测到。

## 已知局限

直说,因为一个宣称什么都能做的 sample,是一个没人会信的 sample:

- 工具是**串行**执行的。模型要读三个文件,就要付三次往返。
- 消息历史**无上限增长**。任务一长就会超出上下文窗口,而且没有压缩。
- 没有流式输出,一轮不结束就什么都看不到。
- 重试次数固定为 3,不会根据剩余 token 预算自适应。
- 单工作区、单 agent、无持久化。进程挂了,这次运行就没了。
- `grep` 用 `rglob` 遍历目录,上限 2000 个文件。它是演示工具,不是 ripgrep。

## 上生产还差什么

- **结构化 trace**,不是 `print`。每次模型调用、每次工具调用各一个 span,带 token 数和延迟,导出到 OTel。
- **工具调用幂等键**,避免重试时把副作用做两遍。
- **审批闸门**给所有会改状态的操作,以及真正的沙箱给所有会执行代码的操作。
- **上下文压缩**——在窗口填满之前摘要旧轮次,而不是运行时才发现上限。
- **并发执行**互不依赖的工具调用。
- **一套 eval。** 这里的恢复测试证明机制是通的,不证明某个具体模型会用。

## License

MIT
