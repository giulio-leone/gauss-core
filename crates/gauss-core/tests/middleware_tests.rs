use gauss_core::error;
use gauss_core::message::Message;
use gauss_core::middleware::*;
use std::sync::Arc;

struct TestMiddleware {
    name: String,
}

#[async_trait::async_trait]
impl Middleware for TestMiddleware {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> MiddlewarePriority {
        MiddlewarePriority::Normal
    }

    async fn before_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        params: &BeforeAgentParams,
    ) -> error::Result<Option<BeforeAgentResult>> {
        // Prepend middleware name to instructions
        let new_instructions = format!(
            "[{}] {}",
            self.name,
            params.instructions.as_deref().unwrap_or("")
        );
        Ok(Some(BeforeAgentResult {
            instructions: Some(new_instructions),
            ..Default::default()
        }))
    }
}

#[tokio::test]
async fn test_middleware_chain_ordering() {
    let mut chain = MiddlewareChain::new();

    chain.use_middleware(Arc::new(TestMiddleware {
        name: "first".into(),
    }));
    chain.use_middleware(Arc::new(TestMiddleware {
        name: "second".into(),
    }));

    let mut ctx = MiddlewareContext {
        session_id: "test-session".into(),
        agent_name: Some("test-agent".into()),
        timestamp: 0,
        metadata: Default::default(),
    };

    let params = BeforeAgentParams {
        messages: vec![Message::user("hello")],
        instructions: Some("original".into()),
        tools: vec![],
    };

    let (result, _, _) = chain.run_before_agent(&mut ctx, params).await.unwrap();
    // Both middlewares should have modified the instructions
    let instructions = result.instructions.unwrap();
    assert!(instructions.contains("[first]"));
    assert!(instructions.contains("[second]"));
}

#[tokio::test]
async fn test_middleware_context_metadata() {
    let mut ctx = MiddlewareContext {
        session_id: "s1".into(),
        agent_name: Some("agent-1".into()),
        timestamp: 0,
        metadata: Default::default(),
    };
    ctx.metadata
        .insert("key1".into(), serde_json::json!("value1"));
    assert_eq!(ctx.metadata.get("key1"), Some(&serde_json::json!("value1")));
    assert_eq!(ctx.agent_name.as_deref(), Some("agent-1"));
}

#[tokio::test]
async fn test_empty_chain() {
    let chain = MiddlewareChain::new();
    let mut ctx = MiddlewareContext {
        session_id: "s1".into(),
        agent_name: None,
        timestamp: 0,
        metadata: Default::default(),
    };
    let params = BeforeAgentParams {
        messages: vec![Message::user("test")],
        instructions: Some("test".into()),
        tools: vec![],
    };

    let (result, aborted, _) = chain.run_before_agent(&mut ctx, params).await.unwrap();
    assert!(!aborted);
    // Instructions unchanged when no middleware modifies them
    assert_eq!(result.instructions.as_deref(), Some("test"));
}
