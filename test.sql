CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    content TEXT,
    plagiarism_percent REAL DEFAULT 0.0,
    ai_percent REAL DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'PENDING'
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);