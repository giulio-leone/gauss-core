//! Parser for SKILL.MD specification files.
//!
//! SKILL.MD defines reusable skill specifications that agents can reference.
//! Each skill has a name, description, ordered steps, and typed inputs/outputs.

use serde::{Deserialize, Serialize};

use crate::error::{GaussError, Result};

/// A skill specification parsed from a SKILL.MD file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    pub steps: Vec<SkillStep>,
    pub inputs: Vec<SkillParam>,
    pub outputs: Vec<SkillParam>,
}

/// A single step in a skill.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillStep {
    pub description: String,
    pub action: Option<String>,
}

/// An input or output parameter for a skill.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillParam {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
}

/// Current section being parsed.
#[derive(Debug, PartialEq)]
enum Section {
    None,
    Description,
    Steps,
    Inputs,
    Outputs,
    Other,
}

/// Parse a SKILL.MD file content into a SkillSpec.
pub fn parse_skill_md(content: &str) -> Result<SkillSpec> {
    let mut spec = SkillSpec::default();
    let mut section = Section::None;
    let mut section_lines: Vec<String> = Vec::new();

    for line in content.lines() {
        // H1 heading — skill name
        if let Some(name) = line.strip_prefix("# ") {
            let name = name.trim();
            if !name.is_empty() && spec.name.is_empty() {
                spec.name = name.to_string();
                flush_section(&mut section, &mut section_lines, &mut spec)?;
                section = Section::Description;
                section_lines.clear();
                continue;
            }
        }

        // H2 heading — sections
        if let Some(heading) = line.strip_prefix("## ") {
            flush_section(&mut section, &mut section_lines, &mut spec)?;
            section_lines.clear();

            let heading_lower = heading.trim().to_lowercase();
            section = match heading_lower.as_str() {
                "steps" => Section::Steps,
                "inputs" | "input" => Section::Inputs,
                "outputs" | "output" => Section::Outputs,
                "description" => Section::Description,
                _ => Section::Other,
            };
            continue;
        }

        section_lines.push(line.to_string());
    }

    flush_section(&mut section, &mut section_lines, &mut spec)?;

    if spec.name.is_empty() {
        return Err(GaussError::internal(
            "SKILL.MD must have a top-level # heading for the skill name",
        ));
    }

    Ok(spec)
}

/// Flush accumulated section lines into the spec.
fn flush_section(
    section: &mut Section,
    lines: &mut Vec<String>,
    spec: &mut SkillSpec,
) -> Result<()> {
    let text = lines.join("\n").trim().to_string();

    match section {
        Section::Description => {
            if !text.is_empty() {
                spec.description = text;
            }
        }
        Section::Steps => {
            parse_steps(&text, &mut spec.steps)?;
        }
        Section::Inputs => {
            parse_params(&text, &mut spec.inputs)?;
        }
        Section::Outputs => {
            parse_params(&text, &mut spec.outputs)?;
        }
        Section::None | Section::Other => {}
    }

    *section = Section::None;
    lines.clear();
    Ok(())
}

/// Parse steps from list items.
///
/// Supports formats:
/// - `- Description` — step with description only
/// - `- Description | action: do_something` — step with description and action
/// - `1. Description` followed by `   Action: do_something` on the next line
fn parse_steps(text: &str, steps: &mut Vec<SkillStep>) -> Result<()> {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(item) = strip_list_item(line) {
            if item.is_empty() {
                i += 1;
                continue;
            }
            if let Some((desc, rest)) = item.split_once('|') {
                let desc = desc.trim().to_string();
                let rest = rest.trim();
                let action = rest
                    .strip_prefix("action:")
                    .map(|a| a.trim().to_string())
                    .or_else(|| {
                        if rest.is_empty() {
                            None
                        } else {
                            Some(rest.to_string())
                        }
                    });
                steps.push(SkillStep {
                    description: desc,
                    action,
                });
            } else {
                // Check if next line is an indented Action: line
                let mut action = None;
                if i + 1 < lines.len() {
                    let next = lines[i + 1].trim();
                    if let Some(a) = next
                        .strip_prefix("Action:")
                        .or_else(|| next.strip_prefix("action:"))
                    {
                        action = Some(a.trim().to_string());
                        i += 1; // skip the Action line
                    }
                }
                steps.push(SkillStep {
                    description: item.to_string(),
                    action,
                });
            }
        }
        i += 1;
    }
    Ok(())
}

/// Parse parameters from list items.
///
/// Supports formats:
/// - `- name (type): description` — required by default
/// - `- name (type, optional): description`
/// - `- name (type, required): description`
fn parse_params(text: &str, params: &mut Vec<SkillParam>) -> Result<()> {
    for line in text.lines() {
        if let Some(item) = strip_list_item(line) {
            if item.is_empty() {
                continue;
            }
            if let Some(param) = parse_single_param(item) {
                params.push(param);
            }
        }
    }
    Ok(())
}

/// Parse a single param line. Supports multiple formats:
/// - `name (type): description`
/// - `name (type, optional): description`
/// - `name: type (required) — description`
/// - `name: type — description`
fn parse_single_param(item: &str) -> Option<SkillParam> {
    // Format 1: name (type): description  or  name (type, required): description
    if let Some(paren_start) = item.find('(') {
        if let Some(paren_end) = item[paren_start..].find(')') {
            let paren_end = paren_start + paren_end;
            let name = item[..paren_start].trim().to_string();
            let inside = &item[paren_start + 1..paren_end];
            let after = item[paren_end + 1..].trim();

            // Check if name contains ":" (format: name: type (qualifier) — desc)
            if let Some((real_name, type_part)) = name.split_once(':') {
                let real_name = real_name.trim().to_string();
                let param_type = type_part.trim().to_string();
                let required = inside.trim().eq_ignore_ascii_case("required")
                    || !inside.split(',').any(|p| p.trim().eq_ignore_ascii_case("optional"));
                let description = after
                    .strip_prefix("—")
                    .or_else(|| after.strip_prefix('-'))
                    .unwrap_or(after)
                    .trim()
                    .to_string();
                if !real_name.is_empty() {
                    return Some(SkillParam {
                        name: real_name,
                        param_type: if param_type.is_empty() {
                            "string".to_string()
                        } else {
                            param_type
                        },
                        description,
                        required,
                    });
                }
            }

            // Standard format: name (type, optional): description
            let description = after
                .strip_prefix(':')
                .or_else(|| after.strip_prefix("—"))
                .or_else(|| after.strip_prefix('-'))
                .unwrap_or(after)
                .trim()
                .to_string();

            let parts: Vec<&str> = inside.split(',').map(|s| s.trim()).collect();
            let param_type = parts.first().unwrap_or(&"string").to_string();
            let required = !parts.iter().any(|p| p.eq_ignore_ascii_case("optional"));

            if !name.is_empty() {
                return Some(SkillParam {
                    name,
                    param_type,
                    description,
                    required,
                });
            }
        }
    }

    // Fallback: name: description
    if let Some((name, desc)) = item.split_once(':') {
        let name = name.trim().to_string();
        let desc = desc.trim().to_string();
        if !name.is_empty() {
            return Some(SkillParam {
                name,
                param_type: "string".to_string(),
                description: desc,
                required: true,
            });
        }
    }

    None
}

fn strip_list_item(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    // Bullet-style: - item  or  * item
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return Some(rest.trim());
    }
    // Numbered: 1. item, 2. item, etc.
    if let Some(dot_pos) = trimmed.find(". ") {
        let prefix = &trimmed[..dot_pos];
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return Some(trimmed[dot_pos + 2..].trim());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_skill() {
        let md = "# CodeReview\nReview code for quality.";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.name, "CodeReview");
        assert_eq!(spec.description, "Review code for quality.");
    }

    #[test]
    fn test_parse_steps_simple() {
        let md = "# Skill\nDesc\n## Steps\n- Read the code\n- Analyze patterns\n- Report findings";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps.len(), 3);
        assert_eq!(spec.steps[0].description, "Read the code");
        assert!(spec.steps[0].action.is_none());
    }

    #[test]
    fn test_parse_steps_with_action() {
        let md = "# Skill\nDesc\n## Steps\n- Read file | action: read_file\n- Analyze | action: analyze";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps.len(), 2);
        assert_eq!(spec.steps[0].action.as_deref(), Some("read_file"));
        assert_eq!(spec.steps[1].action.as_deref(), Some("analyze"));
    }

    #[test]
    fn test_parse_inputs() {
        let md = "# Skill\nDesc\n## Inputs\n- file_path (string): Path to the file\n- verbose (bool, optional): Enable verbose output";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.inputs[0].name, "file_path");
        assert_eq!(spec.inputs[0].param_type, "string");
        assert!(spec.inputs[0].required);
        assert_eq!(spec.inputs[1].name, "verbose");
        assert_eq!(spec.inputs[1].param_type, "bool");
        assert!(!spec.inputs[1].required);
    }

    #[test]
    fn test_parse_outputs() {
        let md = "# Skill\nDesc\n## Outputs\n- report (string): The review report";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.outputs[0].name, "report");
        assert_eq!(spec.outputs[0].param_type, "string");
    }

    #[test]
    fn test_missing_name_returns_error() {
        let md = "Just text without a heading.";
        assert!(parse_skill_md(md).is_err());
    }

    #[test]
    fn test_parse_full_skill() {
        let md = r#"# Summarize
Summarize documents concisely.

## Steps
- Read the document
- Extract key points | action: extract
- Generate summary | action: summarize

## Inputs
- content (string): The document content
- max_length (number, optional): Maximum summary length

## Outputs
- summary (string): The generated summary
- key_points (array): Extracted key points
"#;
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.name, "Summarize");
        assert_eq!(spec.description, "Summarize documents concisely.");
        assert_eq!(spec.steps.len(), 3);
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.outputs.len(), 2);
        assert!(spec.inputs[0].required);
        assert!(!spec.inputs[1].required);
    }

    #[test]
    fn test_description_section_heading() {
        let md = "# Skill\n## Description\nDetailed description here.";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.description, "Detailed description here.");
    }

    #[test]
    fn test_steps_with_pipe_no_action_prefix() {
        let md = "# Skill\nDesc\n## Steps\n- Read | grep";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps[0].description, "Read");
        assert_eq!(spec.steps[0].action.as_deref(), Some("grep"));
    }

    #[test]
    fn test_param_fallback_format() {
        let md = "# Skill\nDesc\n## Inputs\n- query: The search query";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.inputs[0].name, "query");
        assert_eq!(spec.inputs[0].param_type, "string");
        assert_eq!(spec.inputs[0].description, "The search query");
    }

    #[test]
    fn test_input_alias() {
        let md = "# Skill\nDesc\n## Input\n- x (int): A number";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.inputs[0].name, "x");
    }

    #[test]
    fn test_output_alias() {
        let md = "# Skill\nDesc\n## Output\n- result (string): The result";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.outputs.len(), 1);
    }

    #[test]
    fn test_empty_steps_section() {
        let md = "# Skill\nDesc\n## Steps\n## Inputs\n- x (int): val";
        let spec = parse_skill_md(md).unwrap();
        assert!(spec.steps.is_empty());
        assert_eq!(spec.inputs.len(), 1);
    }

    #[test]
    fn test_asterisk_list_items() {
        let md = "# Skill\nDesc\n## Steps\n* Step one\n* Step two";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps.len(), 2);
        assert_eq!(spec.steps[0].description, "Step one");
    }

    #[test]
    fn test_numbered_steps() {
        let md = "# Skill\nDesc\n## Steps\n1. First step\n2. Second step\n3. Third step";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps.len(), 3);
        assert_eq!(spec.steps[0].description, "First step");
        assert_eq!(spec.steps[2].description, "Third step");
    }

    #[test]
    fn test_numbered_steps_with_action_line() {
        let md = "# Skill\nDesc\n## Steps\n1. Read the code\n   Action: analyze_code\n2. Provide feedback";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.steps.len(), 2);
        assert_eq!(spec.steps[0].description, "Read the code");
        assert_eq!(spec.steps[0].action.as_deref(), Some("analyze_code"));
        assert!(spec.steps[1].action.is_none());
    }

    #[test]
    fn test_param_colon_type_required_dash() {
        let md = "# Skill\nDesc\n## Inputs\n- code: string (required) — The code to review\n- language: string — The programming language";
        let spec = parse_skill_md(md).unwrap();
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.inputs[0].name, "code");
        assert_eq!(spec.inputs[0].param_type, "string");
        assert!(spec.inputs[0].required);
        assert_eq!(spec.inputs[0].description, "The code to review");
        assert_eq!(spec.inputs[1].name, "language");
        assert!(spec.inputs[1].required); // no (optional) qualifier → defaults to required via fallback parser
    }
}
