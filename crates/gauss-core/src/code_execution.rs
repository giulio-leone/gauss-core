//! Programmatic Tool Calling (PTC) — execute code in isolated subprocesses.
//!
//! Provides a `CodeRuntime` trait and implementations for Python, JavaScript (Node.js),
//! and Bash. Each runtime runs code in a subprocess with configurable timeout and
//! captures stdout, stderr, and exit code.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{GaussError, Result};

/// Result of executing code in a runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
    pub runtime: String,
}

impl ExecutionResult {
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }
}

/// Configuration for code execution.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum execution time.
    pub timeout: Duration,
    /// Working directory for the subprocess.
    pub working_dir: Option<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            working_dir: None,
            env: Vec::new(),
        }
    }
}

/// Trait for code execution runtimes.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait CodeRuntime: Send + Sync {
    /// Runtime name (e.g., "python", "javascript", "bash").
    fn name(&self) -> &str;

    /// Execute code and return the result.
    async fn execute(&self, code: &str, config: &RuntimeConfig) -> Result<ExecutionResult>;

    /// Check if this runtime is available on the system.
    async fn is_available(&self) -> bool;
}

// ─── Python Runtime ────────────────────────────────────────────────

/// Execute Python code via subprocess.
pub struct PythonRuntime {
    /// Path to the Python interpreter.
    interpreter: String,
}

impl PythonRuntime {
    pub fn new() -> Self {
        Self {
            interpreter: "python3".to_string(),
        }
    }

    pub fn with_interpreter(interpreter: impl Into<String>) -> Self {
        Self {
            interpreter: interpreter.into(),
        }
    }
}

impl Default for PythonRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl CodeRuntime for PythonRuntime {
    fn name(&self) -> &str {
        "python"
    }

    async fn execute(&self, code: &str, config: &RuntimeConfig) -> Result<ExecutionResult> {
        run_subprocess(&self.interpreter, &["-c", code], config, "python").await
    }

    async fn is_available(&self) -> bool {
        check_command(&self.interpreter, &["--version"]).await
    }
}

// ─── JavaScript Runtime ────────────────────────────────────────────

/// Execute JavaScript code via Node.js subprocess.
pub struct JavaScriptRuntime {
    interpreter: String,
}

impl JavaScriptRuntime {
    pub fn new() -> Self {
        Self {
            interpreter: "node".to_string(),
        }
    }

    pub fn with_interpreter(interpreter: impl Into<String>) -> Self {
        Self {
            interpreter: interpreter.into(),
        }
    }
}

impl Default for JavaScriptRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl CodeRuntime for JavaScriptRuntime {
    fn name(&self) -> &str {
        "javascript"
    }

    async fn execute(&self, code: &str, config: &RuntimeConfig) -> Result<ExecutionResult> {
        run_subprocess(&self.interpreter, &["-e", code], config, "javascript").await
    }

    async fn is_available(&self) -> bool {
        check_command(&self.interpreter, &["--version"]).await
    }
}

// ─── Bash Runtime ──────────────────────────────────────────────────

/// Execute Bash scripts via subprocess.
pub struct BashRuntime {
    shell: String,
}

impl BashRuntime {
    pub fn new() -> Self {
        Self {
            shell: "bash".to_string(),
        }
    }

    pub fn with_shell(shell: impl Into<String>) -> Self {
        Self {
            shell: shell.into(),
        }
    }
}

impl Default for BashRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl CodeRuntime for BashRuntime {
    fn name(&self) -> &str {
        "bash"
    }

    async fn execute(&self, code: &str, config: &RuntimeConfig) -> Result<ExecutionResult> {
        run_subprocess(&self.shell, &["-c", code], config, "bash").await
    }

    async fn is_available(&self) -> bool {
        check_command(&self.shell, &["--version"]).await
    }
}

// ─── Subprocess Execution ──────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
async fn run_subprocess(
    program: &str,
    args: &[&str],
    config: &RuntimeConfig,
    runtime_name: &str,
) -> Result<ExecutionResult> {
    use tokio::process::Command;

    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null());

    if let Some(ref dir) = config.working_dir {
        cmd.current_dir(dir);
    }

    for (key, val) in &config.env {
        cmd.env(key, val);
    }

    let child = cmd.spawn().map_err(|e| {
        GaussError::tool(runtime_name, format!("Failed to spawn process: {e}"))
    })?;

    match tokio::time::timeout(config.timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            timed_out: false,
            runtime: runtime_name.to_string(),
        }),
        Ok(Err(e)) => Err(GaussError::tool(
            runtime_name,
            format!("execution error: {e}"),
        )),
        Err(_) => Ok(ExecutionResult {
            stdout: String::new(),
            stderr: format!("Execution timed out after {:?}", config.timeout),
            exit_code: -1,
            timed_out: true,
            runtime: runtime_name.to_string(),
        }),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn check_command(program: &str, args: &[&str]) -> bool {
    tokio::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

// ─── Tool Factory ──────────────────────────────────────────────────

/// Create a Tool that executes code in the given runtime.
///
/// The tool expects a JSON argument `{ "code": "..." }` and returns
/// the ExecutionResult as JSON.
#[cfg(not(target_arch = "wasm32"))]
pub fn code_execution_tool(
    runtime: std::sync::Arc<dyn CodeRuntime>,
    config: RuntimeConfig,
) -> crate::tool::Tool {
    let rt_name = runtime.name().to_string();
    let description = format!(
        "Execute {} code. Pass {{\"code\": \"<your code>\"}}. Returns stdout, stderr, exit_code.",
        rt_name
    );

    crate::tool::Tool::builder(
        &format!("execute_{}", rt_name),
        &description,
    )
    .parameters_json(serde_json::json!({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": format!("{} code to execute", rt_name)
            }
        },
        "required": ["code"]
    }))
    .execute(move |args| {
        let runtime = runtime.clone();
        let config = config.clone();
        Box::pin(async move {
            let code = args
                .get("code")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    GaussError::tool("code_execution", "Missing 'code' argument")
                })?;
            let result = runtime.execute(code, &config).await?;
            serde_json::to_value(&result).map_err(|e| {
                GaussError::tool("code_execution", format!("Serialize error: {e}"))
            })
        })
    })
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_python_execution() {
        let rt = PythonRuntime::new();
        if !rt.is_available().await {
            return; // skip if python3 not available
        }
        let config = RuntimeConfig::default();
        let result = rt.execute("print('hello from python')", &config).await.unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello from python");
    }

    #[tokio::test]
    async fn test_javascript_execution() {
        let rt = JavaScriptRuntime::new();
        if !rt.is_available().await {
            return;
        }
        let config = RuntimeConfig::default();
        let result = rt.execute("console.log('hello from node')", &config).await.unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello from node");
    }

    #[tokio::test]
    async fn test_bash_execution() {
        let rt = BashRuntime::new();
        if !rt.is_available().await {
            return;
        }
        let config = RuntimeConfig::default();
        let result = rt.execute("echo 'hello from bash'", &config).await.unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello from bash");
    }

    #[tokio::test]
    async fn test_timeout() {
        let rt = BashRuntime::new();
        if !rt.is_available().await {
            return;
        }
        let config = RuntimeConfig {
            timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let result = rt.execute("sleep 10", &config).await.unwrap();
        assert!(result.timed_out);
        assert!(!result.success());
    }

    #[tokio::test]
    async fn test_exit_code() {
        let rt = BashRuntime::new();
        if !rt.is_available().await {
            return;
        }
        let config = RuntimeConfig::default();
        let result = rt.execute("exit 42", &config).await.unwrap();
        assert_eq!(result.exit_code, 42);
        assert!(!result.success());
    }

    #[tokio::test]
    async fn test_stderr_capture() {
        let rt = PythonRuntime::new();
        if !rt.is_available().await {
            return;
        }
        let config = RuntimeConfig::default();
        let result = rt
            .execute("import sys; sys.stderr.write('error msg')", &config)
            .await
            .unwrap();
        assert!(result.success());
        assert!(result.stderr.contains("error msg"));
    }
}
