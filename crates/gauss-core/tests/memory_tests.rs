use gauss_core::memory::*;

#[tokio::test]
async fn test_in_memory_store_and_recall() {
    let mem = InMemoryMemory::new();

    let entry = MemoryEntry {
        id: "e1".into(),
        content: "The user asked about Rust".into(),
        entry_type: MemoryEntryType::Conversation,
        tier: Some(MemoryTier::Short),
        timestamp: "2024-01-01T00:00:00Z".into(),
        metadata: None,
        importance: Some(0.8),
        session_id: Some("s1".into()),
        embedding: None,
    };

    mem.store(entry.clone()).await.unwrap();

    let options = RecallOptions {
        query: Some("Rust".into()),
        limit: Some(5),
        ..Default::default()
    };

    let results = mem.recall(options).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "e1");
}

#[tokio::test]
async fn test_recall_filters_by_type() {
    let mem = InMemoryMemory::new();

    mem.store(MemoryEntry {
        id: "e1".into(),
        content: "hello".into(),
        entry_type: MemoryEntryType::Conversation,
        tier: None,
        timestamp: "2024-01-01T00:00:00Z".into(),
        metadata: None,
        importance: None,
        session_id: None,
        embedding: None,
    })
    .await
    .unwrap();

    mem.store(MemoryEntry {
        id: "e2".into(),
        content: "Rust is fast".into(),
        entry_type: MemoryEntryType::Fact,
        tier: None,
        timestamp: "2024-01-01T00:00:01Z".into(),
        metadata: None,
        importance: None,
        session_id: None,
        embedding: None,
    })
    .await
    .unwrap();

    let options = RecallOptions {
        entry_type: Some(MemoryEntryType::Fact),
        ..Default::default()
    };

    let results = mem.recall(options).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "e2");
}

#[tokio::test]
async fn test_clear_all() {
    let mem = InMemoryMemory::new();

    for i in 0..3 {
        mem.store(MemoryEntry {
            id: format!("e{i}"),
            content: format!("msg {i}"),
            entry_type: MemoryEntryType::Conversation,
            tier: None,
            timestamp: format!("2024-01-01T00:00:0{i}Z"),
            metadata: None,
            importance: None,
            session_id: None,
            embedding: None,
        })
        .await
        .unwrap();
    }

    let stats = mem.stats().await.unwrap();
    assert_eq!(stats.total_entries, 3);

    Memory::clear(&mem, None).await.unwrap();

    let stats = mem.stats().await.unwrap();
    assert_eq!(stats.total_entries, 0);
}

#[tokio::test]
async fn test_recall_with_min_importance() {
    let mem = InMemoryMemory::new();

    mem.store(MemoryEntry {
        id: "low".into(),
        content: "low importance".into(),
        entry_type: MemoryEntryType::Fact,
        tier: None,
        timestamp: "2024-01-01T00:00:00Z".into(),
        metadata: None,
        importance: Some(0.2),
        session_id: None,
        embedding: None,
    })
    .await
    .unwrap();

    mem.store(MemoryEntry {
        id: "high".into(),
        content: "high importance".into(),
        entry_type: MemoryEntryType::Fact,
        tier: None,
        timestamp: "2024-01-01T00:00:01Z".into(),
        metadata: None,
        importance: Some(0.9),
        session_id: None,
        embedding: None,
    })
    .await
    .unwrap();

    let results = mem
        .recall(RecallOptions {
            min_importance: Some(0.5),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "high");
}
