from .apply_patch_tool import ApplyPatchTool
from .base_tool import BaseTool, ToolContext, ToolRegistry
from .claude_tool_runtime import (
    BashRuntimeTool,
    EditRuntimeTool,
    GlobRuntimeTool,
    GrepRuntimeTool,
    ReadRuntimeTool,
    WriteRuntimeTool,
    build_claude_tool_registry,
)
from .exec_command_tool import ExecCommandTool
from .grep_files_tool import GrepFilesTool
from .list_dir_tool import ListDirTool
from .placeholder_claude_tool import (
    CLAUDE_PLACEHOLDER_TOOL_CLASSES,
    PlaceholderClaudeTool,
    build_placeholder_claude_tool_registry,
)
from .read_file_tool import ReadFileTool
from .shell_command_tool import ShellCommandTool
from .unified_exec_manager import UnifiedExecManager
from .write_stdin_tool import WriteStdinTool


def build_default_tool_registry() -> ToolRegistry:
    return build_claude_tool_registry()


def build_local_execution_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    manager = UnifiedExecManager()
    registry.register(ReadFileTool())
    registry.register(ListDirTool())
    registry.register(GrepFilesTool())
    registry.register(ShellCommandTool())
    registry.register(ExecCommandTool(manager))
    registry.register(WriteStdinTool(manager))
    registry.register(ApplyPatchTool())
    return registry


__all__ = [
    "ApplyPatchTool",
    "BashRuntimeTool",
    "BaseTool",
    "CLAUDE_PLACEHOLDER_TOOL_CLASSES",
    "EditRuntimeTool",
    "ExecCommandTool",
    "GlobRuntimeTool",
    "GrepRuntimeTool",
    "GrepFilesTool",
    "ListDirTool",
    "PlaceholderClaudeTool",
    "ReadRuntimeTool",
    "ReadFileTool",
    "ShellCommandTool",
    "ToolContext",
    "ToolRegistry",
    "UnifiedExecManager",
    "WriteRuntimeTool",
    "WriteStdinTool",
    "build_claude_tool_registry",
    "build_default_tool_registry",
    "build_local_execution_tool_registry",
    "build_placeholder_claude_tool_registry",
]
