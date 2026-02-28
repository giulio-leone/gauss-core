use gauss_core::context::*;
use gauss_core::message::Message;

#[test]
fn test_count_tokens_approx() {
    assert_eq!(count_tokens_approx("hello"), 2);
    assert_eq!(count_tokens_approx(""), 0);
    assert_eq!(count_tokens_approx("12345678"), 2);
    assert_eq!(count_tokens_approx("123456789"), 3);
}

#[test]
fn test_count_message_tokens() {
    let msg = Message::user("Hello world");
    let tokens = count_message_tokens(&msg);
    // "Hello world" = 11 chars → ceil(11/4)=3 + 4 overhead = 7
    assert_eq!(tokens, 7);
}

#[test]
fn test_context_tracker_basic() {
    let mut tracker = ContextTracker::new("gpt-4");
    let messages = vec![Message::system("system prompt"), Message::user("hello")];
    tracker.update(&messages);

    assert!(tracker.current_tokens > 0);
    assert!(tracker.available_tokens() > 0);
    assert!(!tracker.is_over_limit());
}

#[test]
fn test_prune_messages_oldest_first() {
    let mut messages = Vec::new();
    messages.push(Message::system("You are a helpful assistant."));
    for i in 0..100 {
        messages.push(Message::user(format!(
            "This is a long message number {i} with lots of text to fill the context window."
        )));
    }

    let config = PruningConfig {
        strategy: PruningStrategy::OldestFirst,
        threshold: 0.1, // very low to force pruning
        ..Default::default()
    };

    let pruned = prune_messages(&messages, "gpt-4", &config);
    assert!(pruned.len() < messages.len());
}

#[test]
fn test_prune_messages_sliding_window() {
    let mut messages = Vec::new();
    for i in 0..20 {
        messages.push(Message::user(format!("message {i}")));
    }

    let config = PruningConfig {
        strategy: PruningStrategy::SlidingWindow,
        window_size: Some(3),
        ..Default::default()
    };

    let pruned = prune_messages(&messages, "gpt-4", &config);
    assert_eq!(pruned.len(), 3);
}

#[test]
fn test_context_window_size() {
    assert!(context_window_size("gpt-4") > 0);
    // Unknown model should return a default
    assert!(context_window_size("unknown-model") > 0);
}

#[test]
fn test_context_tracker_usage() {
    let tracker = ContextTracker::new("gpt-4");
    assert_eq!(tracker.current_tokens, 0);
    assert!(tracker.available_tokens() > 0);
}
