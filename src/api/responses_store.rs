use std::{
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Value};

const SCHEMA_VERSION: i64 = 1;
pub(super) const MAX_CONTEXT_ITEMS: usize = 512;
pub(super) const MAX_CONTEXT_BYTES: usize = 8 * 1024 * 1024;
pub(super) const MAX_ITEM_BYTES: usize = 1024 * 1024;
pub(super) const MAX_CREATE_ITEMS: usize = 20;

#[derive(Clone)]
pub(super) struct ResponsesStore {
    path: Arc<PathBuf>,
}

#[derive(Debug)]
pub(super) enum StoreError {
    NotFound(&'static str),
    Conflict(&'static str),
    Limit(&'static str),
    Invalid(&'static str),
    Database(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(message)
            | Self::Conflict(message)
            | Self::Limit(message)
            | Self::Invalid(message) => formatter.write_str(message),
            Self::Database(message) => write!(formatter, "responses database error: {message}"),
        }
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error.to_string())
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Database(error.to_string())
    }
}

#[derive(Debug)]
pub(super) struct StoredResponse {
    pub response: Value,
    pub context: Vec<Value>,
    pub request_hash: String,
}

#[derive(Debug)]
pub(super) struct ConversationSnapshot {
    pub object: Value,
    pub items: Vec<Value>,
}

#[derive(Debug)]
pub(super) struct ResponseCommit<'a> {
    pub id: &'a str,
    pub created_at: u64,
    pub conversation_id: Option<&'a str>,
    pub previous_response_id: Option<&'a str>,
    pub request_hash: &'a str,
    pub idempotency_key: Option<&'a str>,
    pub input: &'a [Value],
    pub output: &'a [Value],
    pub context: &'a [Value],
    pub response: &'a Value,
    pub store_response: bool,
}

#[derive(Clone, Default)]
pub(super) struct ResponseLockPool {
    locks: Arc<Mutex<std::collections::HashMap<String, Weak<tokio::sync::Mutex<()>>>>>,
}

impl ResponseLockPool {
    pub fn for_key(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(key).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(key.to_string(), Arc::downgrade(&lock));
        lock
    }
}

impl Default for ResponsesStore {
    fn default() -> Self {
        Self::new(default_store_path())
    }
}

impl ResponsesStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path: Arc::new(path),
        }
    }

    pub fn create_conversation(
        &self,
        id: &str,
        created_at: u64,
        metadata: &Value,
        items: &[Value],
    ) -> Result<ConversationSnapshot, StoreError> {
        validate_metadata(metadata)?;
        validate_items(items, MAX_CREATE_ITEMS)?;
        validate_context(items)?;
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO conversations (id, created_at, updated_at, metadata_json)
             VALUES (?1, ?2, ?2, ?3)",
            params![id, to_i64(created_at), serde_json::to_string(metadata)?],
        )?;
        append_items(&transaction, id, created_at, items)?;
        transaction.commit()?;
        self.get_conversation(id)
    }

    pub fn get_conversation(&self, id: &str) -> Result<ConversationSnapshot, StoreError> {
        let connection = self.connect()?;
        let object = conversation_object(&connection, id)?;
        let items = conversation_items(&connection, id)?;
        Ok(ConversationSnapshot { object, items })
    }

    pub fn update_conversation(
        &self,
        id: &str,
        metadata: &Value,
        updated_at: u64,
    ) -> Result<Value, StoreError> {
        validate_metadata(metadata)?;
        let connection = self.connect()?;
        let changed = connection.execute(
            "UPDATE conversations
             SET metadata_json = ?2, updated_at = ?3
             WHERE id = ?1",
            params![id, serde_json::to_string(metadata)?, to_i64(updated_at)],
        )?;
        if changed == 0 {
            return Err(StoreError::NotFound("conversation not found"));
        }
        conversation_object(&connection, id)
    }

    pub fn delete_conversation(&self, id: &str) -> Result<(), StoreError> {
        let connection = self.connect()?;
        let changed = connection.execute("DELETE FROM conversations WHERE id = ?1", [id])?;
        if changed == 0 {
            return Err(StoreError::NotFound("conversation not found"));
        }
        Ok(())
    }

    pub fn add_conversation_items(
        &self,
        conversation_id: &str,
        created_at: u64,
        items: &[Value],
    ) -> Result<Vec<Value>, StoreError> {
        validate_items(items, MAX_CREATE_ITEMS)?;
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        require_conversation(&transaction, conversation_id)?;
        validate_append_capacity(&transaction, conversation_id, items)?;
        let inserted = append_items(&transaction, conversation_id, created_at, items)?;
        transaction.commit()?;
        Ok(inserted)
    }

    pub fn list_conversation_items(&self, conversation_id: &str) -> Result<Vec<Value>, StoreError> {
        let connection = self.connect()?;
        require_conversation(&connection, conversation_id)?;
        conversation_items(&connection, conversation_id)
    }

    pub fn get_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Value, StoreError> {
        let connection = self.connect()?;
        connection
            .query_row(
                "SELECT item_json
                 FROM conversation_items
                 WHERE conversation_id = ?1 AND id = ?2",
                params![conversation_id, item_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or(StoreError::NotFound("conversation item not found"))
            .and_then(|value| serde_json::from_str(&value).map_err(StoreError::from))
    }

    pub fn delete_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
        updated_at: u64,
    ) -> Result<(), StoreError> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        let changed = transaction.execute(
            "DELETE FROM conversation_items
             WHERE conversation_id = ?1 AND id = ?2",
            params![conversation_id, item_id],
        )?;
        if changed == 0 {
            return Err(StoreError::NotFound("conversation item not found"));
        }
        resequence_items(&transaction, conversation_id)?;
        transaction.execute(
            "UPDATE conversations SET updated_at = ?2 WHERE id = ?1",
            params![conversation_id, to_i64(updated_at)],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn get_response(&self, id: &str) -> Result<StoredResponse, StoreError> {
        let connection = self.connect()?;
        stored_response(
            &connection,
            "SELECT response_json, context_json, request_hash
             FROM responses WHERE id = ?1",
            id,
        )
    }

    pub fn get_response_by_idempotency_key(
        &self,
        key: &str,
    ) -> Result<Option<StoredResponse>, StoreError> {
        let connection = self.connect()?;
        let row = connection
            .query_row(
                "SELECT response_json, context_json, request_hash
                 FROM responses WHERE idempotency_key = ?1",
                [key],
                response_row,
            )
            .optional()?;
        row.map(parse_stored_response).transpose()
    }

    pub fn delete_response(&self, id: &str) -> Result<(), StoreError> {
        let connection = self.connect()?;
        let changed = connection.execute("DELETE FROM responses WHERE id = ?1", [id])?;
        if changed == 0 {
            return Err(StoreError::NotFound("response not found"));
        }
        Ok(())
    }

    pub fn commit_response(&self, commit: ResponseCommit<'_>) -> Result<(), StoreError> {
        validate_items(commit.input, MAX_CONTEXT_ITEMS)?;
        validate_items(commit.output, MAX_CONTEXT_ITEMS)?;
        validate_context(commit.context)?;
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        if let Some(conversation_id) = commit.conversation_id {
            require_conversation(&transaction, conversation_id)?;
            let mut appended = commit.input.to_vec();
            appended.extend_from_slice(commit.output);
            validate_append_capacity(&transaction, conversation_id, &appended)?;
            append_items(
                &transaction,
                conversation_id,
                super::unix_secs(),
                commit.input,
            )?;
            append_items(
                &transaction,
                conversation_id,
                super::unix_secs(),
                commit.output,
            )?;
        }
        if commit.store_response {
            let completed_at = commit
                .response
                .get("completed_at")
                .and_then(Value::as_u64)
                .map(to_i64);
            transaction
                .execute(
                    "INSERT INTO responses (
                        id, created_at, completed_at, conversation_id,
                        previous_response_id, request_hash, idempotency_key,
                        input_json, output_json, context_json, response_json
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    params![
                        commit.id,
                        to_i64(commit.created_at),
                        completed_at,
                        commit.conversation_id,
                        commit.previous_response_id,
                        commit.request_hash,
                        commit.idempotency_key,
                        serde_json::to_string(commit.input)?,
                        serde_json::to_string(commit.output)?,
                        serde_json::to_string(commit.context)?,
                        serde_json::to_string(commit.response)?,
                    ],
                )
                .map_err(|error| {
                    if matches!(
                        error,
                        rusqlite::Error::SqliteFailure(ref sqlite, _)
                            if sqlite.code == rusqlite::ErrorCode::ConstraintViolation
                    ) {
                        StoreError::Conflict("response id or idempotency key already exists")
                    } else {
                        StoreError::from(error)
                    }
                })?;
        }
        transaction.commit()?;
        Ok(())
    }

    fn connect(&self) -> Result<Connection, StoreError> {
        if let Some(parent) = self.path.parent() {
            let parent_existed = parent.exists();
            fs::create_dir_all(parent).map_err(|error| StoreError::Database(error.to_string()))?;
            if !parent_existed {
                set_private_directory_permissions(parent)?;
            }
        }
        let connection = Connection::open(self.path.as_path())?;
        set_private_file_permissions(self.path.as_path())?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        migrate(&connection)?;
        Ok(connection)
    }
}

fn migrate(connection: &Connection) -> Result<(), StoreError> {
    let version =
        connection.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))?;
    if version > SCHEMA_VERSION {
        return Err(StoreError::Database(format!(
            "database schema version {version} is newer than supported version {SCHEMA_VERSION}"
        )));
    }
    if version == 0 {
        connection.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS conversation_items (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                item_json TEXT NOT NULL,
                UNIQUE(conversation_id, position)
             );
             CREATE INDEX IF NOT EXISTS conversation_items_conversation_position
                ON conversation_items(conversation_id, position);
             CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                conversation_id TEXT
                    REFERENCES conversations(id) ON DELETE SET NULL,
                previous_response_id TEXT,
                request_hash TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                response_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS responses_previous
                ON responses(previous_response_id);
             CREATE INDEX IF NOT EXISTS responses_conversation
                ON responses(conversation_id);
             PRAGMA user_version = 1;
             COMMIT;",
        )?;
    }
    Ok(())
}

fn append_items(
    transaction: &Transaction<'_>,
    conversation_id: &str,
    created_at: u64,
    items: &[Value],
) -> Result<Vec<Value>, StoreError> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let first_position = transaction.query_row(
        "SELECT COALESCE(MAX(position) + 1, 0)
         FROM conversation_items WHERE conversation_id = ?1",
        [conversation_id],
        |row| row.get::<_, i64>(0),
    )?;
    let mut inserted = Vec::with_capacity(items.len());
    for (offset, item) in items.iter().enumerate() {
        let position = first_position + offset as i64;
        let item = item_with_id(item)?;
        let item_id = item["id"]
            .as_str()
            .ok_or(StoreError::Invalid("conversation item id must be a string"))?;
        transaction
            .execute(
                "INSERT INTO conversation_items
                    (id, conversation_id, position, created_at, item_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    item_id,
                    conversation_id,
                    position,
                    to_i64(created_at),
                    serde_json::to_string(&item)?,
                ],
            )
            .map_err(|error| {
                if matches!(
                    error,
                    rusqlite::Error::SqliteFailure(ref sqlite, _)
                        if sqlite.code == rusqlite::ErrorCode::ConstraintViolation
                ) {
                    StoreError::Conflict("conversation item id already exists")
                } else {
                    StoreError::from(error)
                }
            })?;
        inserted.push(item);
    }
    transaction.execute(
        "UPDATE conversations SET updated_at = ?2 WHERE id = ?1",
        params![conversation_id, to_i64(created_at)],
    )?;
    Ok(inserted)
}

fn validate_append_capacity(
    transaction: &Transaction<'_>,
    conversation_id: &str,
    items: &[Value],
) -> Result<(), StoreError> {
    let (existing_count, existing_bytes) = transaction.query_row(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(item_json)), 0)
         FROM conversation_items WHERE conversation_id = ?1",
        [conversation_id],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    let added_bytes = serde_json::to_vec(items)?.len() as i64;
    if existing_count.saturating_add(items.len() as i64) > MAX_CONTEXT_ITEMS as i64 {
        return Err(StoreError::Limit(
            "conversation exceeds the 512 item storage limit",
        ));
    }
    if existing_bytes.saturating_add(added_bytes) > MAX_CONTEXT_BYTES as i64 {
        return Err(StoreError::Limit(
            "conversation exceeds the 8 MiB storage limit",
        ));
    }
    Ok(())
}

fn item_with_id(item: &Value) -> Result<Value, StoreError> {
    let mut item = item.as_object().cloned().ok_or(StoreError::Invalid(
        "conversation items must be JSON objects",
    ))?;
    match item.get("id") {
        Some(Value::String(id)) if !id.is_empty() => {}
        Some(_) => return Err(StoreError::Invalid("conversation item id must be a string")),
        None => {
            item.insert(
                "id".to_string(),
                Value::String(format!("item_{}", uuid::Uuid::new_v4().simple())),
            );
        }
    }
    Ok(Value::Object(item))
}

fn resequence_items(
    transaction: &Transaction<'_>,
    conversation_id: &str,
) -> Result<(), StoreError> {
    let ids = {
        let mut statement = transaction.prepare(
            "SELECT id FROM conversation_items
             WHERE conversation_id = ?1 ORDER BY position ASC",
        )?;
        let ids = statement
            .query_map([conversation_id], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        ids
    };
    // Move positions out of the unique-key range before compacting them.
    transaction.execute(
        "UPDATE conversation_items
         SET position = position + 1000000000
         WHERE conversation_id = ?1",
        [conversation_id],
    )?;
    for (position, id) in ids.iter().enumerate() {
        transaction.execute(
            "UPDATE conversation_items SET position = ?2 WHERE id = ?1",
            params![id, position as i64],
        )?;
    }
    Ok(())
}

fn conversation_object(connection: &Connection, id: &str) -> Result<Value, StoreError> {
    let row = connection
        .query_row(
            "SELECT created_at, metadata_json FROM conversations WHERE id = ?1",
            [id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?
        .ok_or(StoreError::NotFound("conversation not found"))?;
    Ok(json!({
        "id": id,
        "object": "conversation",
        "created_at": row.0,
        "metadata": serde_json::from_str::<Value>(&row.1)?,
    }))
}

fn conversation_items(
    connection: &Connection,
    conversation_id: &str,
) -> Result<Vec<Value>, StoreError> {
    let mut statement = connection.prepare(
        "SELECT item_json FROM conversation_items
         WHERE conversation_id = ?1 ORDER BY position ASC",
    )?;
    let values = statement
        .query_map([conversation_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    values
        .iter()
        .map(|value| serde_json::from_str(value).map_err(StoreError::from))
        .collect()
}

fn require_conversation(connection: &Connection, conversation_id: &str) -> Result<(), StoreError> {
    let exists = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = ?1)",
        [conversation_id],
        |row| row.get::<_, bool>(0),
    )?;
    if exists {
        Ok(())
    } else {
        Err(StoreError::NotFound("conversation not found"))
    }
}

fn response_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, String, String)> {
    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
}

fn parse_stored_response(row: (String, String, String)) -> Result<StoredResponse, StoreError> {
    Ok(StoredResponse {
        response: serde_json::from_str(&row.0)?,
        context: serde_json::from_str(&row.1)?,
        request_hash: row.2,
    })
}

fn stored_response(
    connection: &Connection,
    sql: &str,
    parameter: &str,
) -> Result<StoredResponse, StoreError> {
    connection
        .query_row(sql, [parameter], response_row)
        .optional()?
        .ok_or(StoreError::NotFound("response not found"))
        .and_then(parse_stored_response)
}

fn validate_metadata(metadata: &Value) -> Result<(), StoreError> {
    if metadata.is_object() {
        Ok(())
    } else {
        Err(StoreError::Invalid("metadata must be a JSON object"))
    }
}

pub(super) fn validate_context(items: &[Value]) -> Result<(), StoreError> {
    validate_items(items, MAX_CONTEXT_ITEMS)?;
    let bytes = serde_json::to_vec(items)?.len();
    if bytes > MAX_CONTEXT_BYTES {
        return Err(StoreError::Limit(
            "conversation context exceeds the 8 MiB reconstruction limit",
        ));
    }
    Ok(())
}

fn validate_items(items: &[Value], maximum: usize) -> Result<(), StoreError> {
    if items.len() > maximum {
        return Err(StoreError::Limit("too many conversation items"));
    }
    for item in items {
        if serde_json::to_vec(item)?.len() > MAX_ITEM_BYTES {
            return Err(StoreError::Limit(
                "a conversation item exceeds the 1 MiB item limit",
            ));
        }
        if !item.is_object() {
            return Err(StoreError::Invalid(
                "conversation items must be JSON objects",
            ));
        }
    }
    Ok(())
}

fn to_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

pub(super) fn default_store_path() -> PathBuf {
    if let Some(path) = env::var_os("CAMELID_RESPONSES_DB").filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    #[cfg(windows)]
    if let Some(base) = env::var_os("LOCALAPPDATA") {
        return PathBuf::from(base)
            .join("Camelid")
            .join("responses.sqlite3");
    }
    if let Some(base) = env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(base)
            .join("camelid")
            .join("responses.sqlite3");
    }
    if let Some(home) = env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("camelid")
            .join("responses.sqlite3");
    }
    env::temp_dir().join("camelid").join("responses.sqlite3")
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|error| StoreError::Database(error.to_string()))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> Result<(), StoreError> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file_permissions(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| StoreError::Database(error.to_string()))
}

#[cfg(not(unix))]
fn set_private_file_permissions(_path: &Path) -> Result<(), StoreError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(temp: &tempfile::TempDir) -> ResponsesStore {
        ResponsesStore::new(temp.path().join("responses.sqlite3"))
    }

    #[test]
    fn conversation_items_survive_reopening_and_delete_cascades() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("responses.sqlite3");
        let first = ResponsesStore::new(path.clone());
        first
            .create_conversation(
                "conv_test",
                1,
                &json!({"owner": "test"}),
                &[json!({"type":"message","role":"user","content":"hello"})],
            )
            .unwrap();
        drop(first);

        let reopened = ResponsesStore::new(path);
        let snapshot = reopened.get_conversation("conv_test").unwrap();
        assert_eq!(snapshot.object["metadata"]["owner"], "test");
        assert_eq!(snapshot.items.len(), 1);
        assert!(snapshot.items[0]["id"]
            .as_str()
            .unwrap()
            .starts_with("item_"));

        reopened.delete_conversation("conv_test").unwrap();
        assert!(matches!(
            reopened.get_conversation("conv_test"),
            Err(StoreError::NotFound(_))
        ));
    }

    #[test]
    fn response_context_and_idempotency_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let store = store(&temp);
        let input = vec![json!({"type":"message","role":"user","content":"hello"})];
        let output = vec![json!({
            "id":"fc_test",
            "type":"function_call",
            "call_id":"call_test",
            "name":"weather",
            "arguments":"{\"city\":\"Paris\"}"
        })];
        let mut context = input.clone();
        context.extend(output.clone());
        let response = json!({
            "id":"resp_test",
            "object":"response",
            "output":output,
            "completed_at":2
        });
        store
            .commit_response(ResponseCommit {
                id: "resp_test",
                created_at: 1,
                conversation_id: None,
                previous_response_id: None,
                request_hash: "hash",
                idempotency_key: Some("key"),
                input: &input,
                output: response["output"].as_array().unwrap(),
                context: &context,
                response: &response,
                store_response: true,
            })
            .unwrap();

        let stored = store.get_response("resp_test").unwrap();
        assert_eq!(stored.context[1]["call_id"], "call_test");
        let replay = store
            .get_response_by_idempotency_key("key")
            .unwrap()
            .unwrap();
        assert_eq!(replay.request_hash, "hash");
    }

    #[test]
    fn item_and_context_limits_fail_closed() {
        let oversized = Value::String("x".repeat(MAX_ITEM_BYTES + 1));
        assert!(matches!(
            validate_context(&[json!({"type":"message","content":oversized})]),
            Err(StoreError::Limit(_))
        ));
        assert!(matches!(
            validate_context(&vec![json!({"type":"message"}); MAX_CONTEXT_ITEMS + 1]),
            Err(StoreError::Limit(_))
        ));
    }

    #[test]
    fn failed_response_commit_rolls_back_conversation_items_and_response_row() {
        let temp = tempfile::tempdir().unwrap();
        let store = store(&temp);
        store
            .create_conversation("conv_atomic", 1, &json!({}), &[])
            .unwrap();
        let input = vec![json!({
            "id":"item_duplicate",
            "type":"message",
            "role":"user",
            "content":"hello"
        })];
        let output = vec![json!({
            "id":"item_duplicate",
            "type":"message",
            "role":"assistant",
            "content":"world"
        })];
        let mut context = input.clone();
        context.extend(output.clone());
        let response = json!({
            "id":"resp_atomic",
            "object":"response",
            "output":output,
            "completed_at":2
        });
        let error = store
            .commit_response(ResponseCommit {
                id: "resp_atomic",
                created_at: 1,
                conversation_id: Some("conv_atomic"),
                previous_response_id: None,
                request_hash: "hash",
                idempotency_key: None,
                input: &input,
                output: response["output"].as_array().unwrap(),
                context: &context,
                response: &response,
                store_response: true,
            })
            .unwrap_err();
        assert!(matches!(error, StoreError::Conflict(_)));
        assert!(store
            .list_conversation_items("conv_atomic")
            .unwrap()
            .is_empty());
        assert!(matches!(
            store.get_response("resp_atomic"),
            Err(StoreError::NotFound(_))
        ));
    }
}
