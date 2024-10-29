
# Route Risk-Based Car Insurance Pricing Model

## Overview

Our project introduces a novel car insurance pricing framework that incorporates **route risk** into premium calculations. 
With the increased adoption of autonomous and semi-autonomous vehicles (AVs), detailed route data for drivers is now accessible. 
This data allows us to assess the inherent risk of specific routes and adjust insurance premiums accordingly.

## Features

- **Road Risk Model**: Predicts the likelihood and severity of accidents on road segments using Poisson regression for accident frequency and Gamma regression for severity.
- **Mapping and Visualization**: Visualizes road networks, overlays accident data, and creates density heatmaps for accident occurrences.
- **IoT Data Integration**: Utilizes datasets like POLIDriving and Driver Behavior Dataset to incorporate real-time driving event data and environmental conditions.
- **User Interfaces**: Provides client and server GUIs for seamless interaction and data management.
- **Profit Analysis**: Analyzes the impact of the new pricing model on the insurance company's profitability.

## Technical Description

### Road Network as a Graph

- **Nodes**: Intersections or points where roads connect.
- **Edges**: Road segments.

### Computing Risk Metrics for Roads

Each road segment $m_{ij}$ is assigned a risk score based on:

- **Frequency of Accidents**: Estimated using Poisson regression to calculate the expected time between accidents $\tau_{\text{accident}}$.
- **Severity of Accidents**: Estimated using Gamma regression to determine the severity $s$.

The risk score is calculated as:

$$
m_{ij} = \frac{1}{\tau_{\text{accident}}} \times s
$$

### Optimal Route Calculation

Using the risk scores $m_{ij}$, we apply Dijkstra's algorithm to find the least risky routes:

- **Penalizing Edges**: Edges with higher risk scores are penalized.
- **Minimizing Cumulative Risk**: Optimal paths minimize the total accumulated risk.

### Customer Route Deviation Metric

For each customer $k$, we track their actual routes over a time period $T$ and compute their average deviation from the optimal paths:

$$
\mu_k^T = \text{Average difference in cumulative risk over } T
$$

### Pricing Adjustment Function

We adjust the customer's premium based on their deviation metric $\mu_k^T$:

$$
\sigma_k(\mu_k^T) = \min\left( -\left( 1_{\mu_k^T \in I_l} \Delta_l + 1_{\mu_k^T \in I_m} \Delta_m + 1_{\mu_k^T \in I_h} \Delta_h \right) p_k,\ 0 \right)
$$

Where:

- $p_k$: Traditional premium for customer $k$.
- $I_l, I_m, I_h$: Intervals defining low, medium, and high-risk categories.
- $\Delta_l, \Delta_m, \Delta_h$: Discount factors (values between 0 and 1) for each risk category.
- $1_{\mu_k^T \in I_x}$: Indicator function, equal to 1 if $\mu_k^T \in I_x$, otherwise 0.

The final adjusted premium is:

$$
p_k^* = p_k + \sigma_k(\mu_k^T)
$$

### Traditional Premium Calculation

- **Estimating Expected Loss**:

  $$
  \mathbb{E}[L] = \text{Frequency} \times \text{Severity}
  $$

- **Applying Load Factor**:

  $$
  p_k = \mathbb{E}[L] \times \ell
  $$

  Where the load factor $\ell$ is:

  $$
  \ell = 1 + \text{profit\_percentage} + \text{margin\_of\_safety} + \text{admin\_fee}
  $$

## Data Sources

- **Synthetic Road and Crash Data**: Generated for specific geographic areas, including features like traffic density, road infrastructure quality, and visibility.
- **IoT Datasets**: Incorporates the POLIDriving dataset and Driver Behavior Dataset for real-time driving event data and environmental conditions.

## Visualization

- **Accident Density Heatmap**: Identifies hotspots based on accident frequency and severity.
- **Road Network Overlay**: Provides a clear visualization of risky road segments.

## User Interfaces

- **Client GUI**: Allows users to view their route risk assessments and premium adjustments.
- **Server GUI**: Enables efficient handling of large datasets and real-time updates.

## Profit Analysis

- **Impact Evaluation**: Assesses how the new pricing model affects the insurance company's profitability.
- **Strategic Insights**: Helps in adjusting parameters to balance competitiveness and financial goals.

## Project Structure

```
.
├── README.md
├── model.png
├── heatmap.png
├── client_gui.png
├── server_gui.png
├── profit.png
├── preprocessing_pipeline
│   ├── classic_pricing_model
│   │   └── behavioral_risk_modeler.py
│   ├── data
│   │   ├── synthetic_data.csv
│   │   └── behavioral_important_features.png
│   ├── iot_model
│   │   └── iot_model.ipynb
│   ├── road_risk_model
│   │   ├── generate_map_graph.py
│   │   └── model.py
│   └── runner.ipynb
├── pricing_algorithm
│   ├── dijkstra_map.py
│   ├── main.ipynb
│   └── road_risk_model.ipynb
└── search_algorithm
    ├── main.ipynb
    ├── accident_points.json
    └── chengdu_network.graphml
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Team

We are a dedicated team participating in a prestigious competitive hackathon, aiming to revolutionize car insurance pricing models with data-driven risk assessments.

## Additional Resources

- **Project Documentation**: Detailed explanations of models and algorithms.
- **Data Samples**: Available upon request for validation and testing purposes.

Thank you for reviewing our project. We believe this innovative approach will significantly impact how insurance premiums are calculated, promoting safer driving behaviors and optimized risk management.
