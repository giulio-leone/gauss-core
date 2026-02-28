use gauss_core::plugin::*;
use std::sync::Arc;

#[tokio::test]
async fn event_bus_subscribe_publish() {
    let mut bus = EventBus::new();
    let received = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let received_clone = received.clone();

    bus.subscribe(
        "agent_start",
        Box::new(move |_event| {
            received_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        }),
    );

    bus.publish(&GaussEvent::AgentStart {
        agent_name: "test".into(),
        session_id: "s1".into(),
    });

    assert!(received.load(std::sync::atomic::Ordering::Relaxed));
}

#[tokio::test]
async fn event_bus_wildcard() {
    let mut bus = EventBus::new();
    let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let count_clone = count.clone();

    bus.subscribe(
        "*",
        Box::new(move |_event| {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }),
    );

    bus.publish(&GaussEvent::AgentStart {
        agent_name: "a".into(),
        session_id: "s".into(),
    });
    bus.publish(&GaussEvent::Error {
        source: "test".into(),
        message: "oops".into(),
    });

    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[tokio::test]
async fn event_bus_unsubscribe() {
    let mut bus = EventBus::new();
    let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let count_clone = count.clone();

    let id = bus.subscribe(
        "agent_start",
        Box::new(move |_event| {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }),
    );

    bus.publish(&GaussEvent::AgentStart {
        agent_name: "a".into(),
        session_id: "s".into(),
    });
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1);

    assert!(bus.unsubscribe(&id));

    bus.publish(&GaussEvent::AgentStart {
        agent_name: "b".into(),
        session_id: "s".into(),
    });
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1); // unchanged
}

#[tokio::test]
async fn plugin_registry_init_shutdown() {
    let mut registry = PluginRegistry::new();
    registry.register(Arc::new(TelemetryPlugin));

    assert_eq!(registry.list(), vec!["telemetry"]);

    registry.init_all().await.unwrap();
    assert!(registry.bus.subscription_count() > 0);

    registry.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn plugin_registry_dependency_order() {
    struct PluginA;
    struct PluginB;

    #[async_trait::async_trait]
    impl Plugin for PluginA {
        fn name(&self) -> &str {
            "plugin_a"
        }
        fn version(&self) -> &str {
            "1.0"
        }
        async fn init(
            &self,
            ctx: &mut PluginContext,
            _bus: &mut EventBus,
        ) -> gauss_core::error::Result<()> {
            let order = ctx
                .state
                .entry("init_order".to_string())
                .or_insert(serde_json::Value::Array(Vec::new()));
            order.as_array_mut().unwrap().push(serde_json::json!("a"));
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl Plugin for PluginB {
        fn name(&self) -> &str {
            "plugin_b"
        }
        fn version(&self) -> &str {
            "1.0"
        }
        fn dependencies(&self) -> Vec<&str> {
            vec!["plugin_a"]
        }
        async fn init(
            &self,
            ctx: &mut PluginContext,
            _bus: &mut EventBus,
        ) -> gauss_core::error::Result<()> {
            let order = ctx
                .state
                .entry("init_order".to_string())
                .or_insert(serde_json::Value::Array(Vec::new()));
            order.as_array_mut().unwrap().push(serde_json::json!("b"));
            Ok(())
        }
    }

    let mut registry = PluginRegistry::new();
    // Register B before A — should still init A first due to dependency
    registry.register(Arc::new(PluginB));
    registry.register(Arc::new(PluginA));

    registry.init_all().await.unwrap();

    let order = registry.ctx.state.get("init_order").unwrap();
    let arr = order.as_array().unwrap();
    assert_eq!(arr[0], "a");
    assert_eq!(arr[1], "b");
}

#[tokio::test]
async fn plugin_registry_circular_dependency() {
    struct CircA;
    struct CircB;

    #[async_trait::async_trait]
    impl Plugin for CircA {
        fn name(&self) -> &str {
            "circ_a"
        }
        fn version(&self) -> &str {
            "1.0"
        }
        fn dependencies(&self) -> Vec<&str> {
            vec!["circ_b"]
        }
        async fn init(
            &self,
            _ctx: &mut PluginContext,
            _bus: &mut EventBus,
        ) -> gauss_core::error::Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl Plugin for CircB {
        fn name(&self) -> &str {
            "circ_b"
        }
        fn version(&self) -> &str {
            "1.0"
        }
        fn dependencies(&self) -> Vec<&str> {
            vec!["circ_a"]
        }
        async fn init(
            &self,
            _ctx: &mut PluginContext,
            _bus: &mut EventBus,
        ) -> gauss_core::error::Result<()> {
            Ok(())
        }
    }

    let mut registry = PluginRegistry::new();
    registry.register(Arc::new(CircA));
    registry.register(Arc::new(CircB));

    let result = registry.init_all().await;
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Circular"));
}

#[tokio::test]
async fn memory_plugin_init() {
    let mut registry = PluginRegistry::new();
    registry.register(Arc::new(MemoryPlugin));

    registry.init_all().await.unwrap();

    // MemoryPlugin should have initialized the conversations state
    assert!(registry.ctx.state.contains_key("memory:conversations"));
}

#[tokio::test]
async fn custom_event() {
    let mut bus = EventBus::new();
    let received_data = Arc::new(std::sync::Mutex::new(String::new()));
    let data_clone = received_data.clone();

    bus.subscribe(
        "my_custom_event",
        Box::new(move |event| {
            if let GaussEvent::Custom { data, .. } = event {
                *data_clone.lock().unwrap() = data.to_string();
            }
        }),
    );

    bus.publish(&GaussEvent::Custom {
        name: "my_custom_event".into(),
        data: serde_json::json!({"key": "value"}),
    });

    assert!(received_data.lock().unwrap().contains("value"));
}
