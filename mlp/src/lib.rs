use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json;
use std::os::raw::c_char;
use std::ffi::{c_void, CStr};

// Définition de la structure du modèle MLP
#[derive(Serialize, Deserialize)]
pub struct MLP {
    d: Vec<usize>,
    W: Vec<Vec<Vec<f64>>>,
    L: usize,
    X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    loss: Vec<f64>,
}

impl MLP {
    fn new(npl: Vec<usize>) -> MLP {

        let d: Vec<usize> = npl.to_vec();
        let mut W: Vec<Vec<Vec<f64>>> = Vec::new();
        let L: usize = npl.len() - 1;

        let rng = rand::thread_rng();

        for l in 0..=L {
            W.push(Vec::new());
            if l == 0 {
                continue
            }

            for i in 0..=d[l - 1] {
                W[l].push(Vec::new());
                for j in 0..=d[l] {
                    if j == 0 {
                        W[l][i].push(0.0);
                    }else{
                        W[l][i].push(rand::thread_rng().gen_range(-1.0..1.0));
                    }
                }
            }
        }

        let mut X = Vec::new();
        for l in 0..=L {
            X.push(Vec::new());
            for j in 0..=npl[l] {
                X[l].push(if j == 0 {
                    1.0
                } else {
                    -1.0
                });
            }
        }

        let mut deltas = Vec::new();
        for l in 0..=L {
            deltas.push(Vec::new());
            for _j in 0..=npl[l] {
                deltas[l].push(0.0);
            }
        }

        let loss = Vec::new();

        MLP { d, W, L, X, deltas, loss }
    }

    fn propagate(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) {

        for j in 0..self.d[0] {
            self.X[0][j + 1] = sample_inputs[j];
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l]{
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][i][j] * self.X[l - 1][i];
                }

                if !is_classification{// Application de la fonction d'activation tanh à toutes les couches sauf la dernière
                    if l != self.L { // Vérifie si ce n'est pas la dernière couche
                        total = total.tanh();
                }}
                else {
                    total = total.tanh();
                }
                // Mise à jour des activations
                self.X[l][j] = total;
            }
        }
    }

    fn predict(&mut self, sample_inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(&sample_inputs, is_classification);
        let pred = self.X[self.L][1..].to_vec();
        pred
    }

    pub fn loss(outputs: &[f64], expected_outputs: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (output, expected) in outputs.iter().zip(expected_outputs.iter()) {
            loss += (output - expected).powi(2);
        }
        loss / (2.0 * outputs.len() as f64)
    }

    pub fn train(
        &mut self,
        all_samples_inputs: &[Vec<f64>],
        all_samples_expected_outputs: &[Vec<f64>],
        is_classification: bool,
        iteration_count: usize,
        alpha: f64,
        callback: Option<ProgressCallback>,
        user_data: *mut c_void,
    ) {
        for _it in 0..iteration_count {
            let k = rand::thread_rng().gen_range(0..all_samples_inputs.len());
            let inputs_k = &all_samples_inputs[k];
            let y_k = &all_samples_expected_outputs[k];

            self.propagate(inputs_k, is_classification);

            for j in 1..=self.d[self.L] {
                self.deltas[self.L][j] = self.X[self.L][j] - y_k[j - 1];
                if is_classification {
                    // dans le cas de la classification, multiplier les semi-gradients par la dérivée de la fonction d'activation tanh
                    self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                }
            }

            for l in (1..=self.L).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.W[l][i][j] * self.deltas[l][j];
                    }
                    self.deltas[l - 1][i] = (1.0 - self.X[l - 1][i].powi(2)) * total;
                }
            }

            for l in 1..=self.L {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                    }
                }
            }

            // Appeler le callback si fourni
            if let Some(cb) = callback {
                let progress = _it as f64 / iteration_count as f64 * 100.0;
                cb(progress, user_data);
            }

            if _it % 200 == 0 || _it == (iteration_count - 1) {
                let mut total_loss = 0.0;
                for (inputs, expected_outputs) in all_samples_inputs.iter().zip(all_samples_expected_outputs.iter()) {
                    self.propagate(inputs, is_classification);
                    let outputs = &self.X[self.L][1..];
                    let example_loss = MLP::loss(outputs, expected_outputs);
                    total_loss += example_loss;
                }
                let average_loss = total_loss / all_samples_inputs.len() as f64;
                self.loss.push(average_loss);
                println!("\n Iteration: {}, Average Loss: {}", _it, average_loss);
            }

        }

    }

    // Méthode pour enregistrer le modèle dans un fichier au format JSON
    pub fn save_model(&self, file_path: &str) {
        let model_json = serde_json::to_string(self).map_err(|err| {
            eprintln!("Failed to serialize model: {}", err);
            err
        });
        if let Ok(json) = model_json {
            if let Err(err) = std::fs::write(file_path, json) {
                eprintln!("Failed to write model to file: {}", err);
            }
        }

        println!("model save !!!");
    }

    // Méthode statique pour charger le modèle depuis un fichier JSON
    pub fn load_model(file_path: &str) -> Result<MLP, Box<dyn std::error::Error>> {
        println!("load mlp");
        let model_json = std::fs::read_to_string(file_path)?;
        let mlp = serde_json::from_str(&model_json)?;
        Ok(mlp)
    }

}

// Fonctions d'interface C pour communiquer avec le modèle depuis d'autres langages
#[no_mangle]
pub extern "C" fn create_mlp(npl: *const u32, npl_len: usize) -> Box<MLP> {
    let npl_slice = unsafe { std::slice::from_raw_parts(npl, npl_len) };
    let npl_vec: Vec<usize> = npl_slice.iter().map(|&val| val as usize).collect();
    let mlp = MLP::new(npl_vec);
    Box::new(mlp)
}

#[no_mangle]
pub extern "C" fn load_mlp(file_path: *const c_char) -> *mut MLP {
    let c_str = unsafe { CStr::from_ptr(file_path) };
    let str_slice = c_str.to_str().unwrap_or("");
    let mlp = MLP::load_model(str_slice).unwrap_or_else(|_| panic!("Failed to load model from {}", str_slice));
    Box::into_raw(Box::new(mlp))
}

#[no_mangle]
pub extern "C" fn save_mlp(mlp: &mut MLP, file_path: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(file_path) };
    let str_slice = c_str.to_str().unwrap_or("");

    mlp.save_model(str_slice);
}


#[no_mangle]
pub extern "C" fn mlp_predict(
    mlp: &mut MLP,
    inputs: *const f64,
    inputs_len: usize,
    is_classification: bool,
) -> *mut f64 {
    let slice = unsafe { std::slice::from_raw_parts(inputs, inputs_len) };
    let slice_vec: Vec<f64> = slice.iter().map(|&val| val as f64).collect();
    let outputs = mlp.predict(slice_vec, is_classification);
    Box::into_raw(outputs.into_boxed_slice()) as *mut f64
}

#[no_mangle]
pub extern "C" fn mlp_free(ptr: *mut f64) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

pub type ProgressCallback = extern "C" fn(progress: f64, user_data: *mut c_void);

#[no_mangle]
pub extern "C" fn train_mlp(
    mlp: &mut MLP,
    all_samples_inputs: *const f64,
    all_samples_inputs_row_len: usize,
    all_samples_outputs: *const f64,
    all_samples_outputs_len: usize,
    is_classification: bool,
    iteration_count: usize,
    alpha: f64,
    callback: Option<ProgressCallback>,
    user_data: *mut c_void,
) {
    let inputs_slice = unsafe { std::slice::from_raw_parts(all_samples_inputs, all_samples_inputs_row_len * mlp.d[0]) };
    let outputs_slice = unsafe { std::slice::from_raw_parts(all_samples_outputs, all_samples_outputs_len * mlp.d[mlp.d.len() - 1]) };
    let rows = all_samples_inputs_row_len;
    let cols = mlp.d[0];

    assert_eq!(inputs_slice.len(), rows * cols, "Input slice length mismatch!");

    let mut inputs: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            inputs[i][j] = inputs_slice[i * cols + j];
            //println!("{}, {},{}", inputs[i][j], i,j)
        }
    }

    let out_rows = all_samples_outputs_len;
    let out_cols = mlp.d[mlp.d.len() - 1];

    let mut outputs = vec![vec![0.0; mlp.d[mlp.d.len() - 1]]; all_samples_outputs_len];
    for i in 0..all_samples_outputs_len {
        for j in 0..mlp.d[mlp.d.len() - 1] {
            outputs[i][j] = outputs_slice[i * mlp.d[mlp.d.len() - 1] + j];
        }
    }

    mlp.train(&inputs, &outputs, is_classification, iteration_count, alpha, callback, user_data);
    //println!("save model");
    //mlp.save_model("./mlp_model.json");
}

#[cfg(test)]
mod tests {
    use crate::MLP;

    #[test]
    fn test_train() {

    }
}
