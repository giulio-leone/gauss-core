use clap::Parser;
use gauss_core::message::Message;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "gauss", about = "Gauss — AI Agent Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Show version information
    Version,
    /// Run a prompt against a provider
    Chat {
        /// Provider: openai, anthropic, google
        #[arg(short, long, default_value = "openai")]
        provider: String,
        /// Model name
        #[arg(short, long)]
        model: String,
        /// System instructions
        #[arg(short, long)]
        system: Option<String>,
        /// Single prompt (if omitted, starts interactive mode)
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
    },
    /// List available providers
    Providers,
}

fn build_provider(provider_type: &str, model: &str) -> Result<Arc<dyn Provider>, String> {
    let api_key_env = match provider_type {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "google" => "GOOGLE_API_KEY",
        other => return Err(format!("Unknown provider: {other}")),
    };

    let api_key = std::env::var(api_key_env)
        .map_err(|_| format!("Set {api_key_env} environment variable"))?;

    let config = ProviderConfig::new(&api_key);

    let inner: Arc<dyn Provider> = match provider_type {
        "openai" => Arc::new(OpenAiProvider::new(model, config)),
        "anthropic" => Arc::new(AnthropicProvider::new(model, config)),
        "google" => Arc::new(GoogleProvider::new(model, config)),
        _ => unreachable!(),
    };

    Ok(Arc::new(RetryProvider::new(
        inner,
        RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        },
    )))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Version => {
            println!("gauss {}", env!("CARGO_PKG_VERSION"));
        }
        Commands::Providers => {
            println!("Available providers:");
            println!("  openai     — OpenAI (GPT-4o, GPT-5.2, etc.) [OPENAI_API_KEY]");
            println!("  anthropic  — Anthropic (Claude 4, etc.) [ANTHROPIC_API_KEY]");
            println!("  google     — Google (Gemini 2.5, etc.) [GOOGLE_API_KEY]");
        }
        Commands::Chat {
            provider,
            model,
            system,
            prompt,
        } => {
            let p = match build_provider(&provider, &model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };

            let opts = GenerateOptions::default();
            let mut history: Vec<Message> = Vec::new();

            if let Some(ref sys) = system {
                history.push(Message::system(sys));
            }

            if !prompt.is_empty() {
                // Single-shot mode
                let text = prompt.join(" ");
                history.push(Message::user(&text));
                match p.generate(&history, &[], &opts).await {
                    Ok(result) => {
                        println!("{}", result.text().unwrap_or(""));
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                // Interactive REPL
                println!("gauss chat — {provider}/{model}");
                println!("Type 'exit' to quit.\n");

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

                    match p.generate(&history, &[], &opts).await {
                        Ok(result) => {
                            let reply = result.text().unwrap_or("").to_string();
                            println!("\nassistant> {reply}\n");
                            history.push(Message::assistant(&reply));
                        }
                        Err(e) => {
                            eprintln!("Error: {e}\n");
                        }
                    }
                }
            }
        }
    }
}
