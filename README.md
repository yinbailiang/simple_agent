# simple_agent

基于 [`infinity_agent`](https://pypi.org/project/infinity-agent/) 的极简编码 Agent，只有两类能力：

| 工具 | 说明 |
| --- | --- |
| `bash` | 在 workspace 中执行 shell 命令，返回退出码 / stdout / stderr |
| `read_file` | 读取文本文件，带行号 |
| `write_file` | 创建或覆盖文件，自动建父目录 |
| `edit_file` | 精确字符串替换，适合局部修改 |

所有文件操作都被限制在 `config.infd` 指定的 `workspace` 内。

## 配置

`config.infd`：

```infd
!env import DEEPSEEK_API_KEY as ds_key

Config {
    model = "deepseek-v4-flash"
    api_key = $ds_key
    base_url: url = "https://api.deepseek.com"

    workspace = p"."
    log_dir = p"log"
}
```

密钥从环境变量读取，未设置时启动会报配置错误：

```bash
export DEEPSEEK_API_KEY=sk-...
```

## 运行

```bash
# 交互模式（/help 查看命令，/reset 清空历史，/exit 或 Ctrl-D 退出）
uv run main.py

# 单次任务
uv run main.py 看看项目结构，然后给 main.py 补一段说明

# 指定其它配置文件
uv run main.py -c other.infd
```

每次会话的完整消息记录会写入 `log/session-<时间戳>.jsonl`。

## 结构

```text
main.py
├─ load_app_config()   编译 config.infd（infinity_data）
├─ WorkspaceTools      bash / read_file / write_file / edit_file
└─ SimpleAgent         流式对话 + 工具调用循环（infinity_agent）
```
