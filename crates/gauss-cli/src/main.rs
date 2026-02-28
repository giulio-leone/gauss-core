use clap::Parser;

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
    /// Run an agent from a configuration file
    Run {
        /// Path to agent configuration
        #[arg(short, long)]
        config: String,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Version => {
            println!("gauss {}", env!("CARGO_PKG_VERSION"));
        }
        Commands::Run { config } => {
            println!("Running agent from: {config}");
            // TODO: Implement agent runner
        }
    }
}
