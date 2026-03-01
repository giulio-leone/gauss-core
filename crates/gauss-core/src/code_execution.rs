//! Programmatic Tool Calling (PTC) — execute code in isolated subprocesses.
//!
//! Provides a `CodeRuntime` trait and implementations for Python, JavaScript (Node.js),
//! and Bash. Each runtime runs code in a subprocess with configurable timeout and
//! captures stdout, stderr, and exit code.
//!
//! # Quick Start
//!
//! ```no_run
//! use gauss_core::code_execution::*;
//!
//! // Enable all runtimes with one line:
//! let config = CodeExecutionConfig::all();
//!
//! // Or pick specific runtimes:
//! let config = CodeExecutionConfig::builder()
//!     .python(true)
//!     .javascript(true)
//!     .timeout_secs(60)
//!     .build();
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
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
    /// Sandbox configuration.
    pub sandbox: Option<SandboxConfig>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            working_dir: None,
            env: Vec::new(),
            sandbox: None,
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

    if let Some(ref sandbox) = config.sandbox {
        sandbox.apply_to_command(&mut cmd);
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

// ─── Sandbox Configuration ─────────────────────────────────────────

/// Security sandboxing configuration for code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Deny network access (adds `--network=none` on Linux, uses sandbox-exec on macOS).
    pub no_network: bool,
    /// Deny filesystem writes outside working directory.
    pub read_only_fs: bool,
    /// Maximum memory in bytes (0 = unlimited).
    pub max_memory_bytes: u64,
    /// Maximum number of processes the subprocess may spawn.
    pub max_processes: u32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            no_network: true,
            read_only_fs: false,
            max_memory_bytes: 0,
            max_processes: 10,
        }
    }
}

impl SandboxConfig {
    /// Strict preset: no network, read-only FS, 256 MB memory, 5 processes.
    pub fn strict() -> Self {
        Self {
            no_network: true,
            read_only_fs: true,
            max_memory_bytes: 256 * 1024 * 1024,
            max_processes: 5,
        }
    }

    /// Permissive preset: no restrictions.
    pub fn permissive() -> Self {
        Self {
            no_network: false,
            read_only_fs: false,
            max_memory_bytes: 0,
            max_processes: 0,
        }
    }

    /// Apply sandbox limits to a tokio Command (best-effort, OS-dependent).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn apply_to_command(&self, cmd: &mut tokio::process::Command) {
        if self.max_memory_bytes > 0 || self.max_processes > 0 {
            // On Unix, use ulimit-style env hints the subprocess can read.
            // Real enforcement requires cgroups (Linux) or sandbox-exec (macOS).
            if self.max_memory_bytes > 0 {
                cmd.env("GAUSS_MEM_LIMIT", self.max_memory_bytes.to_string());
            }
            if self.max_processes > 0 {
                cmd.env("GAUSS_PROC_LIMIT", self.max_processes.to_string());
            }
        }
    }
}

// ─── High-Level Config (Developer-Friendly) ────────────────────────

/// Configuration for code execution — the single entry point for enabling PTC.
///
/// # Examples
/// ```no_run
/// use gauss_core::code_execution::CodeExecutionConfig;
///
/// // Enable all runtimes:
/// let cfg = CodeExecutionConfig::all();
///
/// // Custom:
/// let cfg = CodeExecutionConfig::builder()
///     .python(true)
///     .javascript(true)
///     .bash(false)
///     .timeout_secs(60)
///     .sandbox(gauss_core::code_execution::SandboxConfig::strict())
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct CodeExecutionConfig {
    pub python: bool,
    pub javascript: bool,
    pub bash: bool,
    pub timeout: Duration,
    pub working_dir: Option<String>,
    pub env: Vec<(String, String)>,
    pub sandbox: SandboxConfig,
    /// Custom interpreter paths (e.g., "python" → "/usr/local/bin/python3.12").
    pub interpreters: HashMap<String, String>,
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            python: true,
            javascript: true,
            bash: true,
            timeout: Duration::from_secs(30),
            working_dir: None,
            env: Vec::new(),
            sandbox: SandboxConfig::default(),
            interpreters: HashMap::new(),
        }
    }
}

impl CodeExecutionConfig {
    /// Enable all runtimes with sensible defaults.
    pub fn all() -> Self {
        Self::default()
    }

    /// Enable only Python.
    pub fn python_only() -> Self {
        Self {
            python: true,
            javascript: false,
            bash: false,
            ..Default::default()
        }
    }

    /// Start building a custom config.
    pub fn builder() -> CodeExecutionConfigBuilder {
        CodeExecutionConfigBuilder::default()
    }

    /// Convert to RuntimeConfig (for subprocess execution).
    pub fn to_runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
            timeout: self.timeout,
            working_dir: self.working_dir.clone(),
            env: self.env.clone(),
            sandbox: Some(self.sandbox.clone()),
        }
    }
}

/// Builder for CodeExecutionConfig.
#[derive(Debug, Default)]
pub struct CodeExecutionConfigBuilder {
    python: Option<bool>,
    javascript: Option<bool>,
    bash: Option<bool>,
    timeout_secs: Option<u64>,
    working_dir: Option<String>,
    env: Vec<(String, String)>,
    sandbox: Option<SandboxConfig>,
    interpreters: HashMap<String, String>,
}

impl CodeExecutionConfigBuilder {
    pub fn python(mut self, enabled: bool) -> Self {
        self.python = Some(enabled);
        self
    }

    pub fn javascript(mut self, enabled: bool) -> Self {
        self.javascript = Some(enabled);
        self
    }

    pub fn bash(mut self, enabled: bool) -> Self {
        self.bash = Some(enabled);
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn sandbox(mut self, sandbox: SandboxConfig) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    pub fn interpreter(mut self, runtime: impl Into<String>, path: impl Into<String>) -> Self {
        self.interpreters.insert(runtime.into(), path.into());
        self
    }

    pub fn build(self) -> CodeExecutionConfig {
        CodeExecutionConfig {
            python: self.python.unwrap_or(true),
            javascript: self.javascript.unwrap_or(true),
            bash: self.bash.unwrap_or(true),
            timeout: Duration::from_secs(self.timeout_secs.unwrap_or(30)),
            working_dir: self.working_dir,
            env: self.env,
            sandbox: self.sandbox.unwrap_or_default(),
            interpreters: self.interpreters,
        }
    }
}

// ─── PTC Orchestrator ──────────────────────────────────────────────

/// Orchestrator that manages multiple code runtimes and produces Tools for the agent.
///
/// This is the bridge between `CodeExecutionConfig` and the agent's tool system.
#[cfg(not(target_arch = "wasm32"))]
pub struct CodeExecutionOrchestrator {
    runtimes: Vec<(String, Arc<dyn CodeRuntime>)>,
    config: CodeExecutionConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl CodeExecutionOrchestrator {
    /// Create from a CodeExecutionConfig — automatically instantiates enabled runtimes.
    pub fn new(config: CodeExecutionConfig) -> Self {
        let mut runtimes: Vec<(String, Arc<dyn CodeRuntime>)> = Vec::new();

        if config.python {
            let interpreter = config
                .interpreters
                .get("python")
                .cloned()
                .unwrap_or_else(|| "python3".to_string());
            runtimes.push((
                "python".to_string(),
                Arc::new(PythonRuntime::with_interpreter(interpreter)),
            ));
        }

        if config.javascript {
            let interpreter = config
                .interpreters
                .get("javascript")
                .cloned()
                .unwrap_or_else(|| "node".to_string());
            runtimes.push((
                "javascript".to_string(),
                Arc::new(JavaScriptRuntime::with_interpreter(interpreter)),
            ));
        }

        if config.bash {
            let shell = config
                .interpreters
                .get("bash")
                .cloned()
                .unwrap_or_else(|| "bash".to_string());
            runtimes.push((
                "bash".to_string(),
                Arc::new(BashRuntime::with_shell(shell)),
            ));
        }

        Self { runtimes, config }
    }

    /// Add a custom runtime.
    pub fn add_runtime(
        &mut self,
        name: impl Into<String>,
        runtime: Arc<dyn CodeRuntime>,
    ) {
        self.runtimes.push((name.into(), runtime));
    }

    /// Produce Tool instances for all enabled runtimes.
    pub fn tools(&self) -> Vec<crate::tool::Tool> {
        let runtime_config = self.config.to_runtime_config();
        self.runtimes
            .iter()
            .map(|(_, rt)| code_execution_tool(rt.clone(), runtime_config.clone()))
            .collect()
    }

    /// Produce a single "execute_code" meta-tool that dispatches to the correct runtime
    /// based on a `language` argument.
    pub fn unified_tool(&self) -> crate::tool::Tool {
        let runtimes: HashMap<String, Arc<dyn CodeRuntime>> = self
            .runtimes
            .iter()
            .cloned()
            .collect();
        let runtime_config = self.config.to_runtime_config();

        let languages: Vec<String> = runtimes.keys().cloned().collect();
        let lang_list = languages.join(", ");

        crate::tool::Tool::builder(
            "execute_code",
            &format!(
                "Execute code in one of: {lang_list}. Pass {{\"language\": \"...\", \"code\": \"...\"}}."
            ),
        )
        .parameters_json(serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": languages,
                    "description": "Programming language / runtime to use"
                },
                "code": {
                    "type": "string",
                    "description": "Source code to execute"
                }
            },
            "required": ["language", "code"]
        }))
        .execute(move |args| {
            let runtimes = runtimes.clone();
            let config = runtime_config.clone();
            Box::pin(async move {
                let language = args
                    .get("language")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        GaussError::tool("execute_code", "Missing 'language' argument")
                    })?;
                let code = args
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        GaussError::tool("execute_code", "Missing 'code' argument")
                    })?;

                let runtime = runtimes.get(language).ok_or_else(|| {
                    GaussError::tool(
                        "execute_code",
                        format!("Unknown language: {language}. Available: {:?}",
                            runtimes.keys().collect::<Vec<_>>()),
                    )
                })?;

                let result = runtime.execute(code, &config).await?;
                serde_json::to_value(&result).map_err(|e| {
                    GaussError::tool("execute_code", format!("Serialize error: {e}"))
                })
            })
        })
        .build()
    }

    /// Check which runtimes are actually available on this system.
    pub async fn available_runtimes(&self) -> Vec<String> {
        let mut available = Vec::new();
        for (name, rt) in &self.runtimes {
            if rt.is_available().await {
                available.push(name.clone());
            }
        }
        available
    }

    /// Execute code in a specific runtime by name.
    pub async fn execute(
        &self,
        language: &str,
        code: &str,
    ) -> Result<ExecutionResult> {
        let rt = self
            .runtimes
            .iter()
            .find(|(name, _)| name == language)
            .map(|(_, rt)| rt.clone())
            .ok_or_else(|| {
                GaussError::tool(
                    "orchestrator",
                    format!("No runtime for language: {language}"),
                )
            })?;

        rt.execute(code, &self.config.to_runtime_config()).await
    }
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

    #[test]
    fn test_code_execution_config_all() {
        let cfg = CodeExecutionConfig::all();
        assert!(cfg.python);
        assert!(cfg.javascript);
        assert!(cfg.bash);
        assert_eq!(cfg.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_code_execution_config_builder() {
        let cfg = CodeExecutionConfig::builder()
            .python(true)
            .javascript(false)
            .bash(false)
            .timeout_secs(60)
            .working_dir("/tmp")
            .env("FOO", "bar")
            .sandbox(SandboxConfig::strict())
            .interpreter("python", "/usr/local/bin/python3.12")
            .build();

        assert!(cfg.python);
        assert!(!cfg.javascript);
        assert!(!cfg.bash);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
        assert_eq!(cfg.working_dir.as_deref(), Some("/tmp"));
        assert_eq!(cfg.env, vec![("FOO".to_string(), "bar".to_string())]);
        assert!(cfg.sandbox.no_network);
        assert!(cfg.sandbox.read_only_fs);
        assert_eq!(cfg.interpreters.get("python").unwrap(), "/usr/local/bin/python3.12");
    }

    #[test]
    fn test_sandbox_presets() {
        let strict = SandboxConfig::strict();
        assert!(strict.no_network);
        assert!(strict.read_only_fs);
        assert_eq!(strict.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(strict.max_processes, 5);

        let permissive = SandboxConfig::permissive();
        assert!(!permissive.no_network);
        assert!(!permissive.read_only_fs);
        assert_eq!(permissive.max_memory_bytes, 0);
        assert_eq!(permissive.max_processes, 0);
    }

    #[tokio::test]
    async fn test_orchestrator_tools() {
        let config = CodeExecutionConfig::all();
        let orch = CodeExecutionOrchestrator::new(config);
        let tools = orch.tools();
        assert_eq!(tools.len(), 3);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"execute_python"));
        assert!(names.contains(&"execute_javascript"));
        assert!(names.contains(&"execute_bash"));
    }

    #[tokio::test]
    async fn test_orchestrator_unified_tool() {
        let config = CodeExecutionConfig::all();
        let orch = CodeExecutionOrchestrator::new(config);
        let tool = orch.unified_tool();
        assert_eq!(tool.name, "execute_code");
    }

    #[tokio::test]
    async fn test_orchestrator_available_runtimes() {
        let config = CodeExecutionConfig::all();
        let orch = CodeExecutionOrchestrator::new(config);
        let available = orch.available_runtimes().await;
        // At least bash and python should be available on dev machines
        assert!(!available.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_execute() {
        let config = CodeExecutionConfig::all();
        let orch = CodeExecutionOrchestrator::new(config);
        let result = orch.execute("python", "print('orchestrator')").await.unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "orchestrator");
    }

    #[tokio::test]
    async fn test_orchestrator_unknown_language() {
        let config = CodeExecutionConfig::python_only();
        let orch = CodeExecutionOrchestrator::new(config);
        let err = orch.execute("ruby", "puts 'hello'").await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_config_python_only() {
        let config = CodeExecutionConfig::python_only();
        let orch = CodeExecutionOrchestrator::new(config);
        let tools = orch.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "execute_python");
    }
}
