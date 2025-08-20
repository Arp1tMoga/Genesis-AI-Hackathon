# Cybersecurity Watchdog

## Overview

This project contains a collection of datasets for cybersecurity analysis. The goal is to provide data that can be used to develop and test security monitoring and threat detection systems.

## Datasets

The following datasets are included in this project:

*   **Network Traffic Logs (`network_traffic_logs.csv`):** Simulated network flows that include examples of normal and potentially malicious traffic. This dataset can be used to train and evaluate intrusion detection systems.

*   **System Event Logs (`system_event_logs.csv`):** Logs of system-level events, such as user logins, privilege escalations, and file access. This dataset is useful for identifying anomalous behavior and potential security breaches.

*   **Threat Intelligence (`threat_intelligence.csv`):** A collection of threat intelligence data, including known malicious IP addresses, file hashes, and other indicators of compromise (IOCs). This data can be used to enrich security event data and improve the accuracy of threat detection.

*   **Balanced Data (`balanced_data.csv`):** A balanced dataset for model training and evaluation.

## Getting Started

To get started with these datasets, you can use a variety of tools and programming languages, such as Python with libraries like Pandas and Scikit-learn.

### Example

```python
import pandas as pd

# Load the network traffic logs
df = pd.read_csv('network_traffic_logs.csv')

# Display the first 5 rows of the dataframe
print(df.head())
```

## How to Use the Data

For more detailed information on how to use these datasets, please refer to the `How to Use the Data.pdf` and `cybersecurity_datasets_description.md` files.
