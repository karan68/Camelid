pub mod api;
pub mod error;
pub mod gguf;
pub mod inference;
pub mod model;
pub mod tensor;
pub mod tokenizer;

pub use error::{BackendError, Result};

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
