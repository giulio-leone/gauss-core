use gauss_core::rag::*;

#[test]
fn test_text_splitter_basic() {
    let splitter = TextSplitter::new(SplitterConfig {
        chunk_size: 50,
        chunk_overlap: 10,
        ..Default::default()
    });
    let doc = Document {
        id: "d1".into(),
        content: "Hello world. This is a test. We need enough text to split properly. More text here. Even more text to ensure splitting occurs correctly in the document.".into(),
        metadata: Default::default(),
    };
    let chunks = splitter.split(&doc);
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert_eq!(chunk.document_id, "d1");
    }
}

#[test]
fn test_text_splitter_short_text() {
    let splitter = TextSplitter::new(SplitterConfig {
        chunk_size: 1000,
        chunk_overlap: 100,
        ..Default::default()
    });
    let doc = Document {
        id: "d1".into(),
        content: "Short text".into(),
        metadata: Default::default(),
    };
    let chunks = splitter.split(&doc);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "Short text");
}

#[tokio::test]
async fn test_in_memory_vector_store() {
    let store = InMemoryVectorStore::new();

    let chunks = vec![
        Chunk {
            id: "c1".into(),
            document_id: "d1".into(),
            content: "Rust is a systems language".into(),
            index: 0,
            metadata: Default::default(),
            embedding: Some(vec![1.0, 0.0, 0.0]),
        },
        Chunk {
            id: "c2".into(),
            document_id: "d1".into(),
            content: "Python is great for ML".into(),
            index: 1,
            metadata: Default::default(),
            embedding: Some(vec![0.0, 1.0, 0.0]),
        },
    ];

    store.upsert(chunks).await.unwrap();

    let results = store.search(&[0.9, 0.1, 0.0], 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "c1");
    assert!(results[0].score > 0.9);
}

#[tokio::test]
async fn test_vector_store_delete() {
    let store = InMemoryVectorStore::new();

    store
        .upsert(vec![Chunk {
            id: "c1".into(),
            document_id: "d1".into(),
            content: "test".into(),
            index: 0,
            metadata: Default::default(),
            embedding: Some(vec![1.0]),
        }])
        .await
        .unwrap();

    store.delete(&["c1".to_string()]).await.unwrap();

    let results = store.search(&[1.0], 10).await.unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_document_creation() {
    let doc = Document {
        id: "d1".into(),
        content: "Hello world".into(),
        metadata: Default::default(),
    };
    assert_eq!(doc.id, "d1");
    assert_eq!(doc.content, "Hello world");
}
