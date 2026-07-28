# From Isolated feature to Orbits: Discovering Music Concepts via Multi-SAE Alignment
Here is the outline of the paper:
* **Introduction**
* **Related Work**
  * Interpretability of Music Foundation Models
  * Evolution of SAE Architectures and Sparsity
  * SAE Alignment and Relational Interpretability
* **Methodology**
  * Transposition-Induced Orbits of Music Concepts
  * Multi-SAE Alignment with Shared Top-K Support
    * How shared Top-K support encourages alignment.
  * Recovering Rings and Sequences
* **Experiments**
  * Datasets and Configuration
  * Recovered Orbits
* **Downstream Task Evaluation**
  * Baselines
  * Metrics
  * Ablations
  * Results
    * Chord Rings
    * Subdominant Ring
    * SAE Evaluates Features Saliency Beyond Linear Recoverability
    * Global Recoverability vs. Local Structure
* **Limitations**
* **Conclusion and Future Work**
* **Datasets**
  * Training and Validation Data
    * Slakh2100.
    * Key-balanced split construction.
    * Segmentation and pitch shifting.
  * Evaluation Data
    * Chord-Recognition Datasets
      * POP909.
      * RWC-POP-100.
      * Slakh2100 test split.
    * Key-Detection Datasets
      * FMAKv2.
      * GTZAN-Key.
      * GiantSteps.
  * Evaluation Preprocessing and Exclusion Criteria
* **Baselines**
  * Chord Recognition
    * LVCR.
    * Linear probe.
  * Key Detection
    * S-KEY.
    * Linear probes.
* **Downstream-Task Inference**
  * Chord Recognition
    * Chord-boundary and silence features.
  * Key-Signature Detection from Scale-Degree Features
  * Key Detection from Chord Orbits
* **Evaluation Metrics**
  * Chord Recognition
  * Key Detection
* **Batched Post-hoc Evaluation Protocol**
* **Visualization of Recovered Orbits**
  * Sensitivity to the Recovery Threshold
  * Sparse Directed Orbit Graphs
  * Geometric Structure of the Subdominant Ring
  * Orbit Activation Patterns
    * Controlled synthetic diagnostic dataset.
    * Activation patterns in synthetic diagnostic dataset.
    * Activation patterns in real recordings.
* **Ablation Studies**
  * s-SAE and Layer-Wise Comparison
    * Chord Recognition.
    * Key Detection via the Major- and Minor-Chord Rings.
  * Effect of Top-K Sparsity
    * Chord Recognition.
    * Key Detection via the Major- and Minor-Chord Rings.
    * Key Detection via the Subdominant Ring.
* **Robustness Analysis**
  * Orbit-Aware Initialization
  * Robustness across Random Seeds
  * Genre Sensitivity
