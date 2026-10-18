CREATE TABLE IF NOT EXISTS training_events (
  revision INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id TEXT NOT NULL,
  submitted_at TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS training_events_by_review
  ON training_events (review_id, revision DESC);
CREATE TABLE IF NOT EXISTS model_versions (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  published_at TEXT NOT NULL,
  payload TEXT NOT NULL
);
