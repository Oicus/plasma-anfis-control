
# Plasma-ANFIS-Control

This repository contains the implementation of an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for intelligent plasma control. The project combines **fuzzy logic** and **machine learning** to develop a robust real-time control system.

---

## Features

- **Data Acquisition**: Collects sensor data such as plasma temperature and gamma flux.
- **Fuzzy Rule Extraction**: Automatically generates fuzzy logic rules using clustering techniques (e.g., K-Means).
- **ANFIS Model Training**: A PyTorch-based implementation of ANFIS for adaptive control.
- **Real-Time ROS2 Integration**: Supports real-time control using ROS2 nodes.
- **Docker Support**: Fully containerized for easy deployment.

---

## Project Structure

```
plasma-anfis-control/
├── data/                   # Raw sensor data
├── models/                 # Trained ANFIS models
├── src/                    # Source code
│   ├── data_acquisition.py # Sensor data collection
│   ├── fuzzy_rules.py      # Fuzzy logic rule extraction
│   ├── anfis_model.py      # ANFIS model implementation
│   └── ros2_node.py        # Real-time ROS2 control node
└── README.md               # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Oicus/plasma-anfis-control.git
   cd plasma-anfis-control
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install ROS2 for real-time control:
   Follow the [ROS2 installation guide](https://docs.ros.org/en/).

---

## Usage

### 1. Data Collection
Collect sensor data and save it for training:
```bash
python src/data_acquisition.py --duration 300 --sample-rate 200
```

### 2. Fuzzy Rule Extraction
Generate fuzzy logic rules from the collected data:
```bash
python src/fuzzy_rules.py --input data/raw_sensor_data.parquet --clusters 7
```

### 3. Train ANFIS Model
Train the ANFIS model using the extracted rules:
```bash
python src/anfis_model.py --epochs 5000 --batch-size 32
```

### 4. Real-Time Control
Run the ROS2 node for real-time control:
```bash
ros2 run plasma_control anfis_controller --model models/trained_model.pt
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Push your changes and create a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Scikit-Fuzzy**: For fuzzy logic tools.
- **PyTorch**: For ANFIS implementation.
- **ROS2**: For real-time control integration.

---

For more information, feel free to open an issue or contact the repository maintainer.
```
