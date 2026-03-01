use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

pub fn next_handle() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

pub fn py_err(msg: impl std::fmt::Display) -> pyo3::PyErr {
    PyRuntimeError::new_err(format!("{msg}"))
}

pub struct HandleRegistry<T> {
    inner: OnceLock<Mutex<HashMap<u32, T>>>,
}

impl<T> HandleRegistry<T> {
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    fn map(&self) -> &Mutex<HashMap<u32, T>> {
        self.inner.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub fn insert(&self, value: T) -> u32 {
        let id = next_handle();
        self.map().lock().unwrap().insert(id, value);
        id
    }

    pub fn get<F, R>(&self, handle: u32, f: F) -> PyResult<R>
    where
        F: FnOnce(&T) -> R,
    {
        let guard = self.map().lock().unwrap();
        guard
            .get(&handle)
            .map(f)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Handle {handle} not found")))
    }

    pub fn get_clone(&self, handle: u32) -> PyResult<T>
    where
        T: Clone,
    {
        self.get(handle, |v| v.clone())
    }

    pub fn remove(&self, handle: u32) -> PyResult<()> {
        self.map()
            .lock()
            .unwrap()
            .remove(&handle)
            .map(|_| ())
            .ok_or_else(|| PyRuntimeError::new_err(format!("Handle {handle} not found")))
    }

    pub fn with_mut<F, R>(&self, handle: u32, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut T) -> PyResult<R>,
    {
        let mut guard = self.map().lock().unwrap();
        let val = guard
            .get_mut(&handle)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Handle {handle} not found")))?;
        f(val)
    }

    pub fn raw(&self) -> &Mutex<HashMap<u32, T>> {
        self.map()
    }
}

unsafe impl<T: Send> Send for HandleRegistry<T> {}
unsafe impl<T: Send> Sync for HandleRegistry<T> {}
