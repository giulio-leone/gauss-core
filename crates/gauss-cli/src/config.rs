use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{Provider, ProviderConfig};
use serde::Deserialize;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
struct AgentConfig {
    name: String,
    provider: String,
    model: String,
    instructions: Option<String>,
}

pub async fn run_from_config(config_path: Option<&str>, prompt: &[String]) -> Result<(), String> {
    let path = resolve_config_path(config_path)?;
    let content = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;

    let config: AgentConfig = if path.ends_with(".toml") {
        toml::from_str(&content).map_err(|e| format!("TOML parse error: {e}"))?
    } else {
        serde_yaml::from_str(&content).map_err(|e| format!("YAML parse error: {e}"))?
    };

    let provider = build_provider(&config.provider, &config.model)?;

    let mut builder = Agent::builder(&config.name, provider);
    if let Some(ref instr) = config.instructions {
        builder = builder.instructions(instr);
    }
    let agent = builder.build();

    if !prompt.is_empty() {
        let text = prompt.join(" ");
        let messages = vec![Message::user(&text)];
        let response = agent
            .run(messages)
            .await
            .map_err(|e| format!("Agent error: {e}"))?;
        println!("{}", response.text);
    } else {
        println!("gauss run — {}/{}", config.provider, config.model);
        println!("Agent: {}", config.name);
        println!("Type 'exit' to quit.\n");

        let mut history: Vec<Message> = Vec::new();
        if let Some(ref instr) = config.instructions {
            history.push(Message::system(instr));
        }

        let stdin = io::stdin();
        loop {
            print!("you> ");
            io::stdout().flush().ok();

            let mut line = String::new();
            if stdin.lock().read_line(&mut line).is_err() || line.trim() == "exit" {
                break;
            }

            let text = line.trim().to_string();
            if text.is_empty() {
                continue;
            }

            history.push(Message::user(&text));

            match agent.run(history.clone()).await {
                Ok(result) => {
                    let reply = result.text.clone();
                    println!("\nassistant> {reply}\n");
                    history.push(Message::assistant(&reply));
                }
                Err(e) => {
                    eprintln!("Error: {e}\n");
                }
            }
        }
    }

    Ok(())
}

fn resolve_config_path(explicit: Option<&str>) -> Result<String, String> {
    if let Some(p) = explicit {
        if Path::new(p).exists() {
            return Ok(p.to_string());
        }
        return Err(format!("Config file not found: {p}"));
    }

    for candidate in ["gauss.yaml", "gauss.yml", "gauss.toml"] {
        if Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }

    Err("No config file found. Create gauss.yaml or specify --config".to_string())
}

fn build_provider(provider_type: &str, model: &str) -> Result<Arc<dyn Provider>, String> {
    let api_key_env = match provider_type {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "google" => "GOOGLE_API_KEY",
        "groq" => "GROQ_API_KEY",
        "ollama" => "OLLAMA_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        other => return Err(format!("Unknown provider: {other}")),
    };

    let api_key = if provider_type == "ollama" {
        std::env::var(api_key_env).unwrap_or_else(|_| "ollama".to_string())
    } else {
        std::env::var(api_key_env).map_err(|_| format!("Set {api_key_env} environment variable"))?
    };

    let config = ProviderConfig::new(&api_key);

    let provider: Arc<dyn Provider> = match provider_type {
        "openai" => Arc::new(OpenAiProvider::new(model, config)),
        "anthropic" => Arc::new(AnthropicProvider::new(model, config)),
        "google" => Arc::new(GoogleProvider::new(model, config)),
        "groq" => Arc::new(GroqProvider::create(model, config)),
        "ollama" => Arc::new(OllamaProvider::create(model, config)),
        "deepseek" => Arc::new(DeepSeekProvider::create(model, config)),
        _ => unreachable!(),
    };

    Ok(provider)
}
