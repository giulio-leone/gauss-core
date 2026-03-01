//! Parser for AGENTS.MD specification files.
//!
//! AGENTS.MD is a markdown-based format for defining agent configurations.
//! Supports YAML frontmatter, heading-based sections, and key-value fields.

use serde::{Deserialize, Serialize};

use crate::error::{GaussError, Result};

/// Full specification of an agent parsed from an AGENTS.MD file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub instructions: Option<String>,
    pub tools: Vec<AgentToolSpec>,
    pub skills: Vec<String>,
    pub capabilities: Vec<String>,
    pub environment: Vec<(String, String)>,
    pub metadata: serde_json::Value,
}

/// A tool referenced in an AGENTS.MD file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Option<serde_json::Value>,
}

/// Current section being parsed.
#[derive(Debug, PartialEq)]
enum Section {
    None,
    Description,
    Model,
    Provider,
    Instructions,
    Tools,
    ToolEntry(String),
    Skills,
    Capabilities,
    Environment,
    Other,
}

/// Parse an AGENTS.MD file content into an AgentSpec.
pub fn parse_agents_md(content: &str) -> Result<AgentSpec> {
    let mut spec = AgentSpec::default();
    let (frontmatter, body) = extract_frontmatter(content);

    if let Some(fm) = frontmatter {
        spec.metadata = parse_yaml_frontmatter(&fm)?;
    }

    let mut section = Section::None;
    let mut section_lines: Vec<String> = Vec::new();
    let mut current_tool: Option<AgentToolSpec> = None;

    for line in body.lines() {
        // H1 heading — agent name
        if let Some(name) = line.strip_prefix("# ") {
            let name = name.trim();
            if !name.is_empty() && spec.name.is_empty() {
                spec.name = name.to_string();
                flush_section(&mut section, &mut section_lines, &mut spec, &mut current_tool)?;
                section = Section::Description;
                section_lines.clear();
                continue;
            }
        }

        // H2 heading — top-level sections
        if let Some(heading) = line.strip_prefix("## ") {
            flush_section(&mut section, &mut section_lines, &mut spec, &mut current_tool)?;
            section_lines.clear();

            let heading_lower = heading.trim().to_lowercase();
            section = match heading_lower.as_str() {
                "description" => Section::Description,
                "model" => Section::Model,
                "provider" => Section::Provider,
                "instructions" | "system prompt" => Section::Instructions,
                "tools" => Section::Tools,
                "skills" => Section::Skills,
                "capabilities" => Section::Capabilities,
                "environment" | "environment variables" => Section::Environment,
                _ => Section::Other,
            };
            continue;
        }

        // H3 heading inside Tools — tool entry
        if let Some(tool_name) = line.strip_prefix("### ") {
            if section == Section::Tools || matches!(section, Section::ToolEntry(_)) {
                flush_section(&mut section, &mut section_lines, &mut spec, &mut current_tool)?;
                section_lines.clear();
                let tool_name = tool_name.trim().to_string();
                section = Section::ToolEntry(tool_name);
                continue;
            }
        }

        // Key-value fields at top level (model: value, provider: value)
        if section == Section::Description || section == Section::None {
            if let Some(val) = try_kv(line, "model") {
                spec.model = Some(val);
                continue;
            }
            if let Some(val) = try_kv(line, "provider") {
                spec.provider = Some(val);
                continue;
            }
        }

        section_lines.push(line.to_string());
    }

    // Flush remaining section
    flush_section(&mut section, &mut section_lines, &mut spec, &mut current_tool)?;

    // Push final pending tool if any
    if let Some(tool) = current_tool.take() {
        spec.tools.push(tool);
    }

    if spec.name.is_empty() {
        return Err(GaussError::internal("AGENTS.MD must have a top-level # heading for the agent name"));
    }

    Ok(spec)
}

/// Discover AGENTS.MD files in a directory tree.
pub fn discover_agents(dir: &str) -> Result<Vec<AgentSpec>> {
    let mut agents = Vec::new();
    discover_recursive(std::path::Path::new(dir), &mut agents)?;
    Ok(agents)
}

fn discover_recursive(path: &std::path::Path, agents: &mut Vec<AgentSpec>) -> Result<()> {
    let entries = std::fs::read_dir(path)
        .map_err(|e| GaussError::internal(format!("Failed to read directory {}: {}", path.display(), e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| GaussError::internal(format!("Failed to read entry: {}", e)))?;
        let p = entry.path();
        if p.is_dir() {
            discover_recursive(&p, agents)?;
        } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.eq_ignore_ascii_case("agents.md") {
                let content = std::fs::read_to_string(&p)
                    .map_err(|e| GaussError::internal(format!("Failed to read {}: {}", p.display(), e)))?;
                agents.push(parse_agents_md(&content)?);
            }
        }
    }
    Ok(())
}

/// Flush accumulated section lines into the spec.
fn flush_section(
    section: &mut Section,
    lines: &mut Vec<String>,
    spec: &mut AgentSpec,
    current_tool: &mut Option<AgentToolSpec>,
) -> Result<()> {
    let text = lines.join("\n").trim().to_string();

    match section {
        Section::Description => {
            if !text.is_empty() {
                spec.description = text;
            }
        }
        Section::Model => {
            if !text.is_empty() {
                spec.model = Some(text);
            }
        }
        Section::Provider => {
            if !text.is_empty() {
                spec.provider = Some(text);
            }
        }
        Section::Instructions => {
            if !text.is_empty() {
                spec.instructions = Some(text);
            }
        }
        Section::ToolEntry(name) => {
            // Finish previous tool if any
            if let Some(tool) = current_tool.take() {
                spec.tools.push(tool);
            }
            let (desc, params) = parse_tool_body(&text)?;
            *current_tool = Some(AgentToolSpec {
                name: name.clone(),
                description: desc,
                parameters: params,
            });
        }
        Section::Tools => {
            // Flush any pending tool
            if let Some(tool) = current_tool.take() {
                spec.tools.push(tool);
            }
            // Tools section may contain list items as simple tool refs
            for line in text.lines() {
                if let Some(item) = strip_list_item(line) {
                    if !item.is_empty() {
                        spec.tools.push(AgentToolSpec {
                            name: item.to_string(),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        Section::Skills => {
            for line in text.lines() {
                if let Some(item) = strip_list_item(line) {
                    if !item.is_empty() {
                        spec.skills.push(item.to_string());
                    }
                }
            }
        }
        Section::Capabilities => {
            for line in text.lines() {
                if let Some(item) = strip_list_item(line) {
                    if !item.is_empty() {
                        spec.capabilities.push(item.to_string());
                    }
                }
            }
        }
        Section::Environment => {
            for line in text.lines() {
                let line = strip_list_item(line).unwrap_or(line.trim());
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim().to_string();
                    let v = v.trim().to_string();
                    if !k.is_empty() {
                        spec.environment.push((k, v));
                    }
                }
            }
        }
        Section::None | Section::Other => {}
    }

    *section = Section::None;
    lines.clear();
    Ok(())
}

/// Try to extract a key-value pair from a line like `key: value`.
fn try_kv(line: &str, key: &str) -> Option<String> {
    let trimmed = line.trim();
    let lower = trimmed.to_lowercase();
    if lower.starts_with(&format!("{}:", key)) {
        let val = trimmed[key.len() + 1..].trim().to_string();
        if !val.is_empty() {
            return Some(val);
        }
    }
    None
}

/// Extract YAML frontmatter delimited by `---`.
fn extract_frontmatter(content: &str) -> (Option<String>, &str) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (None, content);
    }
    // Skip the first `---` line
    let after_first = match trimmed.strip_prefix("---") {
        Some(rest) => rest.trim_start_matches(|c: char| c == '\r' || c == '\n'),
        None => return (None, content),
    };
    if let Some(end) = after_first.find("\n---") {
        let fm = &after_first[..end];
        let body_start = end + 4; // skip \n---
        let body = after_first[body_start..].trim_start_matches(|c: char| c == '\r' || c == '\n');
        (Some(fm.to_string()), body)
    } else {
        (None, content)
    }
}

/// Parse YAML frontmatter into a serde_json::Value.
/// Handles simple key: value pairs without a full YAML parser.
fn parse_yaml_frontmatter(yaml: &str) -> Result<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim().to_string();
            let value = value.trim();
            if !key.is_empty() {
                let json_val = if value == "true" {
                    serde_json::Value::Bool(true)
                } else if value == "false" {
                    serde_json::Value::Bool(false)
                } else if let Ok(n) = value.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(n) = value.parse::<f64>() {
                    serde_json::json!(n)
                } else {
                    serde_json::Value::String(value.to_string())
                };
                map.insert(key, json_val);
            }
        }
    }
    Ok(serde_json::Value::Object(map))
}

/// Strip markdown list prefix (`- ` or `* `).
fn strip_list_item(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .map(|s| s.trim())
}

/// Parse tool body text into description and optional JSON parameters.
fn parse_tool_body(text: &str) -> Result<(String, Option<serde_json::Value>)> {
    let mut desc_lines = Vec::new();
    let mut json_lines = Vec::new();
    let mut in_json = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```json") {
            in_json = true;
            continue;
        }
        if in_json && trimmed == "```" {
            in_json = false;
            continue;
        }
        if in_json {
            json_lines.push(line);
        } else {
            desc_lines.push(line);
        }
    }

    let description = desc_lines.join("\n").trim().to_string();
    let parameters = if json_lines.is_empty() {
        None
    } else {
        let json_str = json_lines.join("\n");
        let val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GaussError::internal(format!("Invalid JSON in tool parameters: {}", e)))?;
        Some(val)
    };

    Ok((description, parameters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_agent() {
        let md = "# MyAgent\nA simple agent.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.name, "MyAgent");
        assert_eq!(spec.description, "A simple agent.");
    }

    #[test]
    fn test_parse_model_heading() {
        let md = "# Bot\nDesc\n## Model\ngpt-4o";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn test_parse_model_kv() {
        let md = "# Bot\nmodel: gpt-4o-mini\nSome description.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn test_parse_provider_heading() {
        let md = "# Bot\nDesc\n## Provider\nopenai";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.provider.as_deref(), Some("openai"));
    }

    #[test]
    fn test_parse_provider_kv() {
        let md = "# Bot\nprovider: anthropic\nDesc.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.provider.as_deref(), Some("anthropic"));
    }

    #[test]
    fn test_parse_instructions() {
        let md = "# Bot\n## Instructions\nYou are a helpful assistant.\nBe concise.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(
            spec.instructions.as_deref(),
            Some("You are a helpful assistant.\nBe concise.")
        );
    }

    #[test]
    fn test_parse_system_prompt_alias() {
        let md = "# Bot\n## System Prompt\nDo X.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.instructions.as_deref(), Some("Do X."));
    }

    #[test]
    fn test_parse_tools_with_params() {
        let md = r#"# Bot
## Tools
### search
Search the web.
```json
{"type": "object", "properties": {"query": {"type": "string"}}}
```
### calculator
Do math."#;
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.tools.len(), 2);
        assert_eq!(spec.tools[0].name, "search");
        assert_eq!(spec.tools[0].description, "Search the web.");
        assert!(spec.tools[0].parameters.is_some());
        assert_eq!(spec.tools[1].name, "calculator");
        assert_eq!(spec.tools[1].description, "Do math.");
        assert!(spec.tools[1].parameters.is_none());
    }

    #[test]
    fn test_parse_skills() {
        let md = "# Bot\n## Skills\n- code-review.skill.md\n- summarize.skill.md";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.skills, vec!["code-review.skill.md", "summarize.skill.md"]);
    }

    #[test]
    fn test_parse_capabilities() {
        let md = "# Bot\n## Capabilities\n- Code generation\n- Bug fixing\n- Refactoring";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.capabilities.len(), 3);
        assert_eq!(spec.capabilities[0], "Code generation");
    }

    #[test]
    fn test_parse_environment() {
        let md = "# Bot\n## Environment\n- API_KEY=secret123\n- TIMEOUT=30";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.environment.len(), 2);
        assert_eq!(spec.environment[0], ("API_KEY".into(), "secret123".into()));
        assert_eq!(spec.environment[1], ("TIMEOUT".into(), "30".into()));
    }

    #[test]
    fn test_parse_frontmatter() {
        let md = "---\nversion: 1\nauthor: test\n---\n# Bot\nDesc.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.metadata["version"], 1);
        assert_eq!(spec.metadata["author"], "test");
    }

    #[test]
    fn test_missing_name_returns_error() {
        let md = "Just some text without a heading.";
        assert!(parse_agents_md(md).is_err());
    }

    #[test]
    fn test_parse_full_spec() {
        let md = r#"---
version: 2
---
# CodeAssistant
A coding assistant agent.
model: gpt-4o
provider: openai

## Instructions
You help developers write better code.

## Tools
### grep
Search files for patterns.

### edit
Edit files.

## Skills
- refactor.skill.md

## Capabilities
- Code review
- Test generation

## Environment
DEBUG=true
LOG_LEVEL=info
"#;
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.name, "CodeAssistant");
        assert_eq!(spec.description, "A coding assistant agent.");
        assert_eq!(spec.model.as_deref(), Some("gpt-4o"));
        assert_eq!(spec.provider.as_deref(), Some("openai"));
        assert!(spec.instructions.is_some());
        assert_eq!(spec.tools.len(), 2);
        assert_eq!(spec.skills, vec!["refactor.skill.md"]);
        assert_eq!(spec.capabilities.len(), 2);
        assert_eq!(spec.environment.len(), 2);
        assert_eq!(spec.metadata["version"], 2);
    }

    #[test]
    fn test_environment_without_list_prefix() {
        let md = "# Bot\n## Environment\nKEY=val\nFOO=bar";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.environment.len(), 2);
    }

    #[test]
    fn test_tools_as_list() {
        let md = "# Bot\n## Tools\n- search\n- calculator";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.tools.len(), 2);
        assert_eq!(spec.tools[0].name, "search");
    }

    #[test]
    fn test_frontmatter_bool_values() {
        let md = "---\nenabled: true\nverbose: false\n---\n# Bot\nDesc.";
        let spec = parse_agents_md(md).unwrap();
        assert_eq!(spec.metadata["enabled"], true);
        assert_eq!(spec.metadata["verbose"], false);
    }

    #[test]
    fn test_empty_sections_are_none() {
        let md = "# Bot\n## Model\n## Instructions";
        let spec = parse_agents_md(md).unwrap();
        assert!(spec.model.is_none());
        assert!(spec.instructions.is_none());
    }
}
