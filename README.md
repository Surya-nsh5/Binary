# 🧠 Intelligent Cache Replacement using Machine Learning

This project implements an **Intelligent Cache Replacement System** using **Machine Learning** to improve cache efficiency by predicting which cache item to replace.  
Two models are used: **Naive Bayes** and **Random Forest**.

---

## 📌 Overview

The project simulates cache behavior and uses machine learning to analyze patterns of cache access.  
It predicts which cache item should be replaced based on past access patterns, enhancing hit ratios and reducing latency.

---

## 🧩 Dataset

The dataset contains the following **7 features**:

| Feature Name       | Description |
|--------------------|-------------|
| last_access_time   | Timestamp of last access |
| access_count       | Number of times item accessed |
| recency_rank       | Rank based on recency of access |
| access_type        | Type of access (read/write) |
| cache_item         | Cache identifier |
| label              | Indicates whether item is replaced or retained |
| other_features     | Any additional computed metrics |

---

## 🧰 Requirements

Install the dependencies before running: sklearn, flask, pandas, joblib and many more according to the code...


