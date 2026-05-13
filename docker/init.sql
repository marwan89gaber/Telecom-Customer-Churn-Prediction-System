-- docker/init.sql

CREATE TABLE IF NOT EXISTS customers (
    customer_id       VARCHAR(50) PRIMARY KEY,
    gender            VARCHAR(10),
    senior_citizen    INTEGER,
    partner           VARCHAR(5),
    dependents        VARCHAR(5),
    tenure            INTEGER,
    phone_service     VARCHAR(5),
    multiple_lines    VARCHAR(20),
    internet_service  VARCHAR(20),
    online_security   VARCHAR(20),
    online_backup     VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support      VARCHAR(20),
    streaming_tv      VARCHAR(20),
    streaming_movies  VARCHAR(20),
    contract          VARCHAR(20),
    paperless_billing VARCHAR(5),
    payment_method    VARCHAR(50),
    monthly_charges   NUMERIC(8,2),
    total_charges     NUMERIC(10,2),
    churn             VARCHAR(5)
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id   SERIAL PRIMARY KEY,
    customer_id     VARCHAR(50) REFERENCES customers(customer_id),
    churn_prob      NUMERIC(5,4),
    risk_tier       VARCHAR(10),
    predicted_at    TIMESTAMP DEFAULT NOW()
);
