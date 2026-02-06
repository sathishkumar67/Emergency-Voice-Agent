CREATE TABLE dbo.Incidents (
    incident_id INT IDENTITY(1,1) PRIMARY KEY,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),

    call_id NVARCHAR(64) NOT NULL,
    language NVARCHAR(8) NULL,

    call_type NVARCHAR(16) NULL,
    incident_type NVARCHAR(128) NULL,
    location NVARCHAR(256) NULL,
    caller_name NVARCHAR(128) NULL,

    confidence_overall FLOAT NULL,

    payload_json NVARCHAR(MAX) NULL,
    transcript_text NVARCHAR(MAX) NULL
);

CREATE TABLE dbo.CallMetrics (
    id INT IDENTITY(1,1) PRIMARY KEY,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),

    call_id NVARCHAR(64) NOT NULL,

    started_at_unix FLOAT NOT NULL,
    ended_at_unix FLOAT NULL,

    user_turns INT NOT NULL,
    agent_turns INT NOT NULL,

    stt_final_count INT NOT NULL,
    stt_interim_count INT NOT NULL,

    extraction_updates INT NOT NULL,

    incident_created BIT NOT NULL,
    incident_id INT NULL
);