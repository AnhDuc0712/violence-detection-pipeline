-- ==============================
-- INIT DATABASE
-- ==============================

CREATE DATABASE violence_detection
    WITH ENCODING 'UTF8'
    TEMPLATE template0;

\c violence_detection;

-- ==============================
-- USERS
-- ==============================

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'viewer', 'ai')),
    created_at TIMESTAMP DEFAULT now()
);

-- ==============================
-- AUDIT LOGS
-- ==============================

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ==============================
-- CAMERAS
-- ==============================

CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    source_type TEXT NOT NULL CHECK (source_type IN ('file', 'rtsp', 'webcam')),
    source_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- ==============================
-- VIDEOS
-- ==============================

CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    camera_id INT REFERENCES cameras(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    duration FLOAT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT now()
);

-- ==============================
-- VIOLENCE EVENTS
-- ==============================

CREATE TABLE violence_events (
    id SERIAL PRIMARY KEY,
    video_id INT REFERENCES videos(id) ON DELETE CASCADE,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    confidence FLOAT CHECK (confidence BETWEEN 0 AND 1),
    violence_type TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ==============================
-- DETECTIONS
-- ==============================

CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    event_id INT REFERENCES violence_events(id) ON DELETE CASCADE,
    frame_index INT NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    class_name TEXT NOT NULL,
    score FLOAT CHECK (score BETWEEN 0 AND 1)
);

-- ==============================
-- INDEXES (PERFORMANCE)
-- ==============================

CREATE INDEX idx_videos_camera ON videos(camera_id);
CREATE INDEX idx_events_video ON violence_events(video_id);
CREATE INDEX idx_events_time ON violence_events(start_time, end_time);
CREATE INDEX idx_detections_event ON detections(event_id);
