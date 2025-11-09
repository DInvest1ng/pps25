-- Init DB schema (runs once at container init)
CREATE TABLE IF NOT EXISTS items (
  id BIGSERIAL PRIMARY KEY,
  description TEXT,
  image_key TEXT,
  created_at TIMESTAMP DEFAULT now(),
  is_deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS images (
  id BIGSERIAL PRIMARY KEY,
  object_key TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);