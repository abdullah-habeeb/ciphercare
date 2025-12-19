export const GLOSSARY = {
    // Metrics
    "AUROC": "Area Under the ROC Curve. Measures the ability of a classifier to distinguish between classes. 1.0 is perfect, 0.5 is random guessing.",
    "Accuracy": "The percentage of correct predictions made by the model.",
    "Recall": "The ability of the model to find all the relevant cases (e.g., finding all sick patients).",
    "Precision": "The ability of the model to only identify relevant data points (e.g., minimizing false alarms).",

    // Federated Learning
    "FedProx": "Federated Proximal Optimization. An FL strategy that adds a regularization term to handle non-IID data (data that varies wildly between hospitals).",
    "Clients": "Individual hospitals or devices that train local models effectively without sharing raw data.",
    "Aggregator": "The central server that combines model updates from all clients to create a global model.",
    "Global Model": "The shared master model that effectively learns from all hospital data combined.",
    "Non-IID": "Non-Independent and Identically Distributed. Means data at one hospital looks very different from data at another (e.g., Hospital A has diverse ages, D only has elderly).",

    // Privacy
    "Differential Privacy": "A mathematical guarantee that the output of an algorithm stays roughly the same whether any single individual's data is included or not.",
    "DP": "Differential Privacy. Adds noise to hide individual contributions.",
    "Epsilon": "Privacy Budget (ε). Lower means more privacy (more noise). Higher means less privacy (less noise) but better accuracy.",
    "Delta": "Failure Probability (δ). The tiny chance that the privacy guarantee fails. Usually set to 1/N.",
    "Noise Scale": "The amount of random Gaussian noise added to gradients. Calculated based on sensitivity and epsilon.",
    "Gradient Clipping": "Limiting the maximum influence any single sample can have on the model update, preventing one outlier from leaking information.",
    "Sensitivity": "The maximum amount the output can change if one individual's data is added or removed.",

    // Blockchain
    "SHA-256": "Secure Hash Algorithm 256-bit. A cryptographic fingerprint of data. Changing a single character changes the entire hash completely.",
    "Merkle Root": "A single hash that represents the integrity of an entire block of transactions.",
    "Immutable": "Cannot be changed. Once a block is written to the blockchain, it cannot be altered without breaking the entire chain.",
    "Genesis Block": "The very first block in a blockchain.",

    // Models
    "S4 Model": "Structured State Space Sequence model. Specialized for handling long sequences like ECG signals efficiently.",
    "ResNet50": "Residual Network with 50 layers. A powerful standard model for image recognition (used for X-Rays).",
    "MLP": "Multi-Layer Perceptron. A classic neural network used for tabular data.",
    "Fusion": "Combining multiple data types (e.g., ECG + Vitals) into a single model judgment.",

    // Fairness
    "Fairness Weighting": "Our custom aggregation formula that rewards hospitals for Performance (AUROC) and Data Quantity, ensuring high-quality contributors get more influence.",
    "Domain Relevance": "A score measuring how useful a hospital's data is for the specific medical task."
};
