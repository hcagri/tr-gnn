# BI-DIRECTIONAL TRACKLET EMBEDDING FOR MULTI-OBJECT TRACKING
Official repository of our ICIP 2024 paper.
## Abstract
The last decade has seen significant advancements in multi-object tracking, particularly with the emergence of deep learning based methods. However, many prior studies in online tracking have primarily 
focused on enhancing track management or extracting visual features, often leading to hybrid approaches with limited effectiveness, especially in scenarios with severe occlusions. 
Conversely, in offline tracking, there has been a lack of emphasis on robust motion cues. In response, this approach aims to present a novel solution for offline tracking by merging tracklets 
using some recent promising learning-based architectures. We leverage a jointly performing Transformer and Graph Neural Network (GNN) encoder to integrate both the individual motions of targets 
and their interactions in between. By enabling bi-directional information propagation between the Transformer and the GNN, proposed model allows motion modeling to depend on interactions, 
and conversely, interaction modeling to depend on the motion of each target. The proposed solution is an end-to-end trainable model that eliminates the requirement for any handcrafted short-term or 
long-term matching pro- cesses. This approach performs on par with state-of-the-art multi-object tracking algorithms, demonstrating its effectiveness and robustness.
## Acknowledgements 
We utilized the codebase of [SUSHI](https://github.com/dvl-tum/SUSHI) and built upon it for our modifications. We would like to extend our gratitude to the SUSHI team for their foundational work.
