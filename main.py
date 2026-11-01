"""simple_agent —— 基于 infinity_agent 的极简编码 Agent。

只提供两类能力：
- bash：在 workspace 中执行 shell 命令
- 文件：读取 / 写入 / 精确替换

用法：
    uv run main.py                    # 交互模式
    uv run main.py 看看项目里有什么      # 单次任务
    uv run main.py -c other.infd       # 指定配置文件
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from types import TracebackType
from typing import Annotated, Any, Self, cast

from infinity_agent import (
    Message,
    StreamChunk,
    TextChunk,
    ToolCall,
)
from infinity_agent.clients import (
    DeepSeekConfig,
    DeepSeekRequestConfig,
    LLMClient,
    LLMError,
    create_client,
)
from infinity_agent.models import ThinkingChunk
from infinity_agent.tools import Raw, ToolRegistry
from infinity_data import SandboxConfig, Severity, load

DEFAULT_CONFIG = Path(__file__).with_name('config.infd')
MAX_STEPS = 40
MAX_TOOL_OUTPUT = 20_000
BASH = shutil.which('bash') or '/bin/sh'

SYSTEM_PROMPT = """你是一个运行在用户终端里的编码 Agent。

工作目录（workspace）：{workspace}

可用工具：
- bash：在 workspace 中执行 bash 命令，返回退出码、stdout、stderr
- read_file：读取文本文件，返回带行号的内容
- write_file：创建或覆盖文件（自动创建父目录）
- edit_file：精确替换文件中的一段内容（修改已有文件时优先使用）

工作原则：
1. 先了解现状再动手：用 bash 或 read_file 查看目录与文件内容。
2. 修改文件后尽量运行命令验证（测试、编译、运行脚本等）。
3. 所有路径都相对于 workspace，不要访问 workspace 之外的位置。
4. 回答简洁，使用与用户相同的语言。"""


# ---------------------------------------------------------------------------
# 终端输出
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty()


def _style(code: str, text: str) -> str:
    return f'\033[{code}m{text}\033[0m' if _COLOR else text


def _dim(text: str) -> str:
    return _style('2', text)


def _cyan(text: str) -> str:
    return _style('36', text)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f'{text[:limit]}\n…（已截断，共 {len(text)} 字符）'


def _print_tool_call(call: ToolCall) -> None:
    args = json.dumps(call.function.arguments, ensure_ascii=False)
    print(f'\n{_cyan("▶ " + call.function.name)} {_clip(args, 600)}', flush=True)


def _print_tool_result(message: Message) -> None:
    content = message.content if isinstance(message.content, str) else ''
    print(_dim(f'  ↳ {_clip(content, 1200)}'), flush=True)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppConfig:
    """config.infd 中 Config 块的运行时视图。"""

    model: str
    api_key: str
    base_url: str
    workspace: Path
    log_dir: Path


def _field(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise SystemExit(f'config: 缺少字段 {key!r}')
    return data[key]


def _as_str(data: Mapping[str, Any], key: str) -> str:
    value = _field(data, key)
    if not isinstance(value, str):
        raise SystemExit(f'config: 字段 {key!r} 必须是字符串，实际为 {type(value).__name__}')
    return value


def _as_path(data: Mapping[str, Any], key: str, base: Path) -> Path:
    value = _field(data, key)
    if isinstance(value, PurePath):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise SystemExit(f'config: 字段 {key!r} 必须是路径字面量 p"...", 实际为 {type(value).__name__}')
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def load_app_config(path: Path) -> AppConfig:
    """编译 config.infd 并取出 Config 块。"""
    if not path.is_file():
        raise SystemExit(f'配置文件不存在: {path}')

    # development 沙盒：允许读取 OS 环境变量、导入本地文件/模板
    result = load(path, sandbox=SandboxConfig.development())

    for diagnostic in result.diagnostics:
        if diagnostic.severity is not Severity.ERROR:
            print(f'config 警告 {diagnostic.location}: {diagnostic.message}', file=sys.stderr)
    if result.has_errors:
        details = '\n'.join(
            f'  [{d.code}] {d.message}'
            for d in result.diagnostics
            if d.severity is Severity.ERROR
        )
        raise SystemExit(f'配置文件解析失败: {path}\n{details}')

    block = result.value.get('Config')
    if not isinstance(block, dict):
        raise SystemExit(f'config: 缺少 Config {{ ... }} 块: {path}')
    block = cast(dict[str, Any], block)

    base = path.resolve().parent
    config = AppConfig(
        model=_as_str(block, 'model'),
        api_key=_as_str(block, 'api_key'),
        base_url=_as_str(block, 'base_url'),
        workspace=_as_path(block, 'workspace', base),
        log_dir=_as_path(block, 'log_dir', base),
    )
    if not config.workspace.is_dir():
        raise SystemExit(f'workspace 目录不存在: {config.workspace}')
    return config


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _format_bash_result(exit_code: int | None, stdout: str, stderr: str) -> str:
    parts = [f'exit_code: {exit_code}']
    if stdout.strip():
        parts.append(f'--- stdout ---\n{stdout.rstrip()}')
    if stderr.strip():
        parts.append(f'--- stderr ---\n{stderr.rstrip()}')
    if len(parts) == 1:
        parts.append('（命令无输出）')
    return '\n'.join(parts)


class WorkspaceTools:
    """绑定到某个 workspace 根目录的工具集合。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def register(self, registry: ToolRegistry) -> None:
        for func in (self.bash, self.read_file, self.write_file, self.edit_file):
            registry.register(func)

    # -- 路径安全 ----------------------------------------------------------

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f'路径超出 workspace 范围: {raw!r}（workspace={self.root}）')
        return resolved

    # -- bash --------------------------------------------------------------

    async def bash(
        self,
        command: Annotated[str, '要执行的 bash 命令'],
        timeout: Annotated[int, '超时时间（秒），默认 120'] = 120,
    ) -> Raw:
        """在 workspace 根目录执行 bash 命令，返回退出码、stdout 与 stderr。"""
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            executable=BASH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return f'命令超时（>{timeout}s），已终止: {command}'

        stdout = _clip(out.decode('utf-8', errors='replace'), MAX_TOOL_OUTPUT)
        stderr = _clip(err.decode('utf-8', errors='replace'), MAX_TOOL_OUTPUT)
        return _format_bash_result(proc.returncode, stdout, stderr)

    # -- 文件 --------------------------------------------------------------

    def read_file(
        self,
        path: Annotated[str, '相对 workspace 的文件路径'],
        start_line: Annotated[int, '起始行号（从 1 开始）'] = 1,
        max_lines: Annotated[int, '最多读取的行数'] = 400,
    ) -> Raw:
        """读取文本文件，返回带行号的内容，便于后续精确编辑。"""
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f'文件不存在: {path}')

        lines = target.read_text(encoding='utf-8', errors='replace').splitlines()
        total = len(lines)
        start = max(start_line, 1)
        end = min(start - 1 + max(max_lines, 1), total)

        header = f'{path}（共 {total} 行）'
        if total == 0:
            return f'{header}\n（空文件）'
        if start > total:
            return f'{header}\n（起始行 {start} 超出文件末尾）'

        body = '\n'.join(f'{i:>6}\t{lines[i - 1]}' for i in range(start, end + 1))
        if end < total:
            header += f'；已显示 {start}-{end}，继续读取请设置 start_line={end + 1}'
        else:
            header += f'；已显示 {start}-{end}'
        return f'{header}\n{body}'

    def write_file(
        self,
        path: Annotated[str, '相对 workspace 的文件路径'],
        content: Annotated[str, '写入的完整文件内容（会覆盖已有内容）'],
    ) -> Raw:
        """创建或覆盖文件，自动创建缺失的父目录。"""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding='utf-8')
        action = '已覆盖' if existed else '已创建'
        lines = content.count('\n') + (0 if content.endswith('\n') or not content else 1)
        return f'{action}: {path}（{len(content)} 字符，{lines} 行）'

    def edit_file(
        self,
        path: Annotated[str, '相对 workspace 的文件路径'],
        old_string: Annotated[str, '要被替换的原文，必须与文件内容完全一致（含缩进）'],
        new_string: Annotated[str, '替换后的新内容'],
        replace_all: Annotated[bool, '是否替换全部匹配项，默认仅替换唯一匹配'] = False,
    ) -> Raw:
        """在文件中精确替换一段字符串，适合局部修改已有文件。"""
        if not old_string:
            raise ValueError('old_string 不能为空')

        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f'文件不存在: {path}')

        text = target.read_text(encoding='utf-8')
        count = text.count(old_string)
        if count == 0:
            raise ValueError(f'未找到要替换的内容，请先用 read_file 确认原文: {path}')
        if count > 1 and not replace_all:
            raise ValueError(
                f'在 {path} 中匹配到 {count} 处，请提供更多上下文，或设置 replace_all=true'
            )

        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        target.write_text(updated, encoding='utf-8')
        return f'已更新: {path}（替换 {count if replace_all else 1} 处）'


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SimpleAgent:
    """对话循环 + 工具调度。"""

    def __init__(self, config: AppConfig, client: LLMClient, registry: ToolRegistry) -> None:
        self._config = config
        self._client = client
        self._registry = registry
        self._request_config = DeepSeekRequestConfig(tools=registry.definitions)
        stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
        self._log_path = config.log_dir / f'session-{stamp}.jsonl'
        self._messages: list[Message] = [
            Message.system(SYSTEM_PROMPT.format(workspace=config.workspace))
        ]

    @property
    def workspace(self) -> Path:
        return self._config.workspace

    @property
    def log_path(self) -> Path:
        return self._log_path

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    def reset(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        del self._messages[1:]

    async def run_task(self, task: str) -> None:
        """执行一次用户任务，直到模型不再请求工具。"""
        self._messages.append(Message.user(task))
        try:
            for _ in range(MAX_STEPS):
                assistant = await self._stream_once()
                self._messages.append(assistant)

                tool_calls = assistant.tool_calls or []
                if not tool_calls:
                    return

                for call in tool_calls:
                    _print_tool_call(call)
                    result = await self._registry.invoke(call)
                    _print_tool_result(result)
                    self._messages.append(result)

            print(_dim(f'\n[已达到最大工具调用轮数 {MAX_STEPS}，本次任务中止]'))
        finally:
            self._save_log()

    # -- 内部 --------------------------------------------------------------

    async def _stream_once(self) -> Message:
        chunks: list[StreamChunk] = []
        thinking = False
        text = False

        async for chunk in self._client.stream_chat(self._messages, self._request_config):
            chunks.append(chunk)
            if isinstance(chunk, ThinkingChunk):
                if not thinking:
                    print(_dim('\n💭 '), end='', flush=True)
                    thinking = True
                print(_dim(chunk.text), end='', flush=True)
            elif isinstance(chunk, TextChunk):
                if thinking:
                    print()
                    thinking = False
                if not text:
                    print('\n🤖 ', end='', flush=True)
                    text = True
                print(chunk.text, end='', flush=True)

        if thinking or text:
            print()
        return LLMClient.chunks_to_message(chunks)

    def _save_log(self) -> None:
        try:
            self._config.log_dir.mkdir(parents=True, exist_ok=True)
            with self._log_path.open('w', encoding='utf-8') as fh:
                for message in self._messages:
                    fh.write(json.dumps(message.model_dump(mode='json'), ensure_ascii=False) + '\n')
        except OSError as exc:
            print(f'[warn] 日志写入失败: {exc}', file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP_TEXT = """命令：
  /reset   清空对话历史
  /exit    退出（也可用 Ctrl-D）"""


async def repl(agent: SimpleAgent) -> None:
    print(f'workspace: {agent.workspace}')
    print('输入任务开始对话（/help 查看命令，Ctrl-D 退出）')

    while True:
        try:
            line = input('\n你 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        if line in {'/exit', '/quit'}:
            return
        if line == '/help':
            print(HELP_TEXT)
            continue
        if line == '/reset':
            agent.reset()
            print('对话历史已清空')
            continue

        try:
            await agent.run_task(line)
        except KeyboardInterrupt:
            print('\n[已中断本次任务]', file=sys.stderr)
        except LLMError as exc:
            print(f'\n[LLM 错误] {exc}', file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='基于 infinity_agent 的极简编码 Agent')
    parser.add_argument('task', nargs='*', help='一次性执行的任务；省略则进入交互模式')
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG), help='配置文件路径（默认 config.infd）')
    return parser


def create_agent(config: AppConfig) -> SimpleAgent:
    client = create_client(
        DeepSeekConfig(model=config.model, api_key=config.api_key, base_url=config.base_url)
    )
    registry = ToolRegistry()
    WorkspaceTools(config.workspace).register(registry)
    return SimpleAgent(config, client, registry)


async def amain() -> int:
    args = build_parser().parse_args()
    config = load_app_config(Path(args.config).expanduser())

    agent = create_agent(config)
    async with agent:
        task = ' '.join(args.task).strip()
        if task:
            try:
                await agent.run_task(task)
            except LLMError as exc:
                print(f'[LLM 错误] {exc}', file=sys.stderr)
                return 1
            print(_dim(f'\n日志: {agent.log_path}'))
            return 0

        await repl(agent)
        return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == '__main__':
    main()
