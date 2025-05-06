🚗 Car Model Recognition ReportObjective: Build a high-performance deep learning system to classify 209 car models from images — ideal for fleet management, security, and visual search applications.🔍 1. Pipeline OverviewStageDescriptionData PreparationMerge Iran (13 classes) & Stanford (196 classes); resize to 400×400, normalize; extensive augmentation (rotation, crop, flip, brightness/contrast).Model DesignResNet-18 backbone (pretrained, fine-tuned) + 2× CBAM attention blocks (post-layer2/layer3); head: GAP → FC(209) → Softmax.TrainingWeighted Cross-Entropy + WeightedSampler; Adam (lr=1e-3, wd=1e-5); LR ↓×0.5 every 10 epochs; early stop (5 epochs); batch=32; epochs=40; multi-GPU (2× T4).EvaluationMetrics: Loss, Accuracy, Precision, Recall, F1-score on test set; visualize train/val curves.DeploymentExport model & inference script; provide REST API for image upload and prediction.📊 2. Dataset DetailsSourceClassesTrain ImagesTest ImagesTotal ImagesIran136,8091,4748,283Stanford1968,1448,04116,185Combined20914,9539,51524,468Diverse makes, angles & lighting scenarios ensure robust learning.🏗️ 3. Model ArchitectureInput (400×400×3)
│
├─ ResNet-18 Backbone (pretrained)
│   ├─ layer1
│   ├─ layer2 ──► CBAM Block
│   ├─ layer3 ──► CBAM Block
│   └─ layer4
│
├─ Global Average Pooling
└─ Fully Connected (209) + Softmax

Params: ~11.3M • Activation: ReLU
⚙️ 4. Training ConfigurationParameterValueOptimizerAdamLearning Rate (lr)1×10⁻³Weight Decay (wd)1×10⁻⁵Batch Size32Epochs40LR Scheduler×0.5 every 10 epochsEarly Stopping5 epochs patienceClass ImbalanceWeighted Cross-Entropy & SamplerHardware2× Tesla T4 GPUs (CUDA)FrameworkPyTorch📈 5. Training & Validation Curves🎯 6. Test Set PerformanceMetricTest ValueLoss0.7763Accuracy81.73%Precision82.66%Recall81.73%F1-score81.75%Achieved strong classification across 209 models with attention and imbalance handling.
