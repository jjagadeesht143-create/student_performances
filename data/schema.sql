-- schema.sql
-- Run this once against MySQL to create the database and tables:
--   mysql -u root -p < data/schema.sql

CREATE DATABASE IF NOT EXISTS student_performance_db;
USE student_performance_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NULL,
    google_id VARCHAR(255) NULL UNIQUE,
    profile_picture VARCHAR(500) NULL,
    auth_provider VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    study_time DECIMAL(4,1) NOT NULL,
    attendance DECIMAL(5,1) NOT NULL,
    previous_score DECIMAL(5,1) NOT NULL,
    assignment_score DECIMAL(5,1) NOT NULL,
    parental_education VARCHAR(30) NOT NULL,
    extracurricular_activity VARCHAR(5) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NULL,
    predicted_score DECIMAL(5,1) NOT NULL,
    performance_category VARCHAR(30) NOT NULL,
    model_name VARCHAR(60) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL
);
