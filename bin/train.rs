// use burn::config::Config;
// use burn::module::Module;
// use burn::nn::{Linear, LinearConfig};
// use burn::optim::AdamConfig;
// use burn::tensor::backend::{AutodiffBackend, Backend};
// use burn::tensor::{Int, Tensor};
// use std::fs::File;
// use std::io::Read;

// // --- 1. Model Definition ---

// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     fc0: Linear<B>,
//     fc1: Linear<B>,
//     fc2: Linear<B>,
// }

// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     #[config(default = 199)]
//     features: usize,
//     #[config(default = 128)]
//     hl: usize,
// }

// impl ModelConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
//         Model {
//             fc0: LinearConfig::new(self.features, self.hl).init(device),
//             fc1: LinearConfig::new(self.hl * 2, 64).init(device),
//             fc2: LinearConfig::new(64, 1).init(device),
//         }
//     }
// }

// impl<B: Backend> Model<B> {
//     pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
//         let [batch_size, _] = input.dims();

//         // Split perspective: STM (0..199) and NSTM (199..398)
//         let stm = input.clone().slice([0..batch_size, 0..199]);
//         let nstm = input.slice([0..batch_size, 199..398]);

//         // Accumulator + SCReLU (clamp 0-1 and square)
//         let acc_stm = self.fc0.forward(stm).clamp(0.0, 1.0).powf_scalar(2.0);
//         let acc_nstm = self.fc0.forward(nstm).clamp(0.0, 1.0).powf_scalar(2.0);

//         // Concatenate and hidden layer
//         let combined = Tensor::cat(vec![acc_stm, acc_nstm], 1);
//         let l1 = self.fc1.forward(combined).clamp(0.0, 1.0).powf_scalar(2.0);

//         // Output Sigmoid
//         let out = self.fc2.forward(l1);
//         burn::tensor::activation::sigmoid(out)
//     }
// }

// // --- 2. Data Loading ---

// #[derive(Clone, Debug)]
// pub struct NNUEItem {
//     pub features: Vec<f32>,
//     pub label: f32,
// }

// pub struct NNUEDataset {
//     items: Vec<NNUEItem>,
// }

// impl NNUEDataset {
//     pub fn from_bin(path: &str) -> Self {
//         let mut file = File::open(path).expect("Could not open data file");
//         let mut buffer = Vec::new();
//         file.read_to_end(&mut buffer).unwrap();

//         let row_floats = 399; // 398 features + 1 label
//         let row_bytes = row_floats * 4;

//         // Basic alignment check (simplified version of your Python logic)
//         let mut start_offset = 0;
//         for skip in [0, 4, 8, 16] {
//             if (buffer.len() - skip) % row_bytes == 0 {
//                 start_offset = skip;
//                 break;
//             }
//         }

//         let data_f32: &[f32] = unsafe {
//             let (_, floats, _) = buffer[start_offset..].align_to::<f32>();
//             floats
//         };

//         let items = data_f32
//             .chunks_exact(row_floats)
//             .map(|chunk| NNUEItem {
//                 features: chunk[..398].to_vec(),
//                 label: chunk[398],
//             })
//             .collect();

//         println!("Loaded {} samples from {}", items.len(), path);
//         Self { items }
//     }
// }

// impl Dataset<NNUEItem> for NNUEDataset {
//     fn get(&self, index: usize) -> Option<NNUEItem> {
//         self.items.get(index).cloned()
//     }
//     fn len(&self) -> usize {
//         self.items.len()
//     }
// }

// // --- 3. Training Boilerplate ---

// #[derive(Clone, Debug)]
// pub struct NNUEBatcher<B: Backend> {
//     device: B::Device,
// }

// #[derive(Clone, Debug)]
// pub struct NNUEBatch<B: Backend> {
//     pub inputs: Tensor<B, 2>,
//     pub targets: Tensor<B, 2>,
// }

// impl<B: Backend> burn::data::dataloader::batcher::Batcher<NNUEItem, NNUEBatch<B>>
//     for NNUEBatcher<B>
// {
//     fn batch(&self, items: Vec<NNUEItem>) -> NNUEBatch<B> {
//         let inputs = items
//             .iter()
//             .map(|item| Tensor::<B, 1>::from_floats(item.features.as_slice(), &self.device))
//             .map(|tensor| tensor.reshape([1, 398]))
//             .collect();

//         let targets = items
//             .iter()
//             .map(|item| Tensor::<B, 1>::from_floats([item.label], &self.device))
//             .map(|tensor| tensor.reshape([1, 1]))
//             .collect();

//         NNUEBatch {
//             inputs: Tensor::cat(inputs, 0),
//             targets: Tensor::cat(targets, 0),
//         }
//     }
// }

// impl<B: AutodiffBackend> TrainStep<NNUEBatch<B>, RegressionOutput<B>> for Model<B> {
//     fn step(&self, batch: NNUEBatch<B>) -> TrainOutput<RegressionOutput<B>> {
//         let item = self.forward(batch.inputs);
//         let loss = burn::nn::loss::MseLoss::new().forward(
//             item.clone(),
//             batch.targets.clone(),
//             burn::nn::loss::Reduction::Mean,
//         );

//         TrainOutput::new(
//             self,
//             loss.backward(),
//             RegressionOutput::new(item, batch.targets, loss),
//         )
//     }
// }

// impl<B: Backend> ValidStep<NNUEBatch<B>, RegressionOutput<B>> for Model<B> {
//     fn step(&self, batch: NNUEBatch<B>) -> RegressionOutput<B> {
//         let item = self.forward(batch.inputs);
//         let loss = burn::nn::loss::MseLoss::new().forward(
//             item.clone(),
//             batch.targets.clone(),
//             burn::nn::loss::Reduction::Mean,
//         );

//         RegressionOutput::new(item, batch.targets, loss)
//     }
// }

// // --- 4. Main Execution ---

// fn main() {
//     // This backend automatically selects your AMD GPU
//     type MyBackend = burn::backend::Wgpu;
//     type MyAutodiffBackend = burn::backend::autodiff::AutodiffBackend<MyBackend>;

//     let device = burn::backend::wgpu::WgpuDevice::default();
//     let artifact_dir = "/tmp/nnue_training";

//     let dataset = NNUEDataset::from_bin("./databin/gen0_data.bin");
//     let model_config = ModelConfig::new();
//     let optimizer = AdamConfig::new();

//     let batcher_train = NNUEBatcher::<MyAutodiffBackend> {
//         device: device.clone(),
//     };
//     let batcher_valid = NNUEBatcher::<MyBackend> {
//         device: device.clone(),
//     };

//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(16384)
//         .shuffle(42)
//         .num_workers(4)
//         .build(dataset);

//     let learner = LearnerBuilder::new(artifact_dir)
//         .metric_train_numeric(burn::train::metric::LossMetric::new())
//         .metric_valid_numeric(burn::train::metric::LossMetric::new())
//         .with_file_checkpointer(NoRecorder::new())
//         .devices(vec![device.clone()])
//         .num_epochs(100)
//         .build(
//             model_config.init::<MyAutodiffBackend>(&device),
//             optimizer.init(),
//             1e-3,
//         );

//     let _model_trained = learner.fit(dataloader_train, Vec::new());

//     println!("Training complete. Optimized for AMD GPU.");
// }

fn main() {
    eprintln!("Rust-native training is not yet implemented. Use train.py instead.");
}
