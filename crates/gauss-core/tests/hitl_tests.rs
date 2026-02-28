use gauss_core::hitl::*;
use gauss_core::message::Message;

#[test]
fn test_approval_request_creation() {
    let req = ApprovalRequest {
        id: "req-1".into(),
        tool_name: "dangerous_tool".into(),
        args: serde_json::json!({"path": "/tmp/test"}),
        step_index: 0,
        session_id: "s1".into(),
        status: ApprovalStatus::Pending,
        created_at: 1000,
        modified_args: None,
        denial_reason: None,
    };

    assert_eq!(req.id, "req-1");
    assert_eq!(req.tool_name, "dangerous_tool");
}

#[test]
fn test_hitl_config_patterns() {
    let config = HitlConfig {
        require_approval_for: vec!["delete_*".into(), "execute_*".into()],
        timeout_ms: 30000,
        on_timeout: TimeoutAction::Deny,
    };

    assert_eq!(config.require_approval_for.len(), 2);
    assert_eq!(config.timeout_ms, 30000);
}

#[tokio::test]
async fn test_in_memory_checkpoint_store() {
    let store = InMemoryCheckpointStore::new();

    let checkpoint = Checkpoint::new("s1".into(), vec![Message::user("hello")]);

    let id = checkpoint.id.clone();
    store.save(&checkpoint).await.unwrap();

    let loaded = store.load(&id).await.unwrap();
    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert_eq!(loaded.session_id, "s1");
}

#[tokio::test]
async fn test_checkpoint_store_list() {
    let store = InMemoryCheckpointStore::new();

    for _ in 0..3 {
        let cp = Checkpoint::new("s1".into(), vec![Message::user("test")]);
        store.save(&cp).await.unwrap();
    }

    let checkpoints = store.list("s1").await.unwrap();
    assert_eq!(checkpoints.len(), 3);
}

#[tokio::test]
async fn test_checkpoint_store_delete() {
    let store = InMemoryCheckpointStore::new();

    let cp = Checkpoint::new("s1".into(), vec![]);
    let id = cp.id.clone();
    store.save(&cp).await.unwrap();

    store.delete(&id).await.unwrap();
    let loaded = store.load(&id).await.unwrap();
    assert!(loaded.is_none());
}

#[test]
fn test_approval_manager_submit_and_approve() {
    let manager = ApprovalManager::new();

    let req = manager
        .request_approval("delete_file".into(), serde_json::json!({}), 0, "s1".into())
        .unwrap();

    let req_id = req.id.clone();

    let pending = manager.list_pending().unwrap();
    assert_eq!(pending.len(), 1);

    manager.approve(&req_id, None).unwrap();

    let pending = manager.list_pending().unwrap();
    assert!(pending.is_empty());
}

#[test]
fn test_approval_manager_deny() {
    let manager = ApprovalManager::new();

    let req = manager
        .request_approval(
            "launch_missiles".into(),
            serde_json::json!({}),
            0,
            "s1".into(),
        )
        .unwrap();

    manager.deny(&req.id, Some("Too dangerous".into())).unwrap();

    let pending = manager.list_pending().unwrap();
    assert!(pending.is_empty());
}
