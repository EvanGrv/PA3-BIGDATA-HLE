use rand::Rng;

#[repr(C)]
struct MLP {
    d: Vec<usize>,
    W: Vec<Vec<Vec<f64>>>,
    L: usize,
    X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>
}

impl MLP {
    fn new(npl: Vec<usize>) -> MLP {

        println!("new");

        let d: Vec<usize> = npl.iter().cloned().collect();
        let mut W: Vec<Vec<Vec<f64>>> = vec![];
        let L: usize = npl.len() - 1;

        let mut rng = rand::thread_rng();

        for l in 0..=L {
            W.push(vec![]);

            if l == 0 {
                continue
            }

            for i in 0..=d[l - 1] {
                W[l].push(vec![]);

                for j in 0..=d[l] {
                    if j == 0 {
                        W[l][i].push(0.0);
                    }
                    else{
                        W[l][i].push(rng.gen::<f64>() * 2.0 - 1.0);
                    }
                }
            }
        }


        let mut X: Vec<Vec<f64>> = vec![];
        let mut deltas: Vec<Vec<f64>> = vec![];

        for l in 0..=L {
            X.push(vec![]);
            deltas.push(vec![]);

            for j in 0..=d[l] {
                deltas[l].push(0.0);
                if j == 0 {
                    X[l].push(1.0);
                }else {
                    X[l].push(0.0);
                }
            }
        }

        println!("self.X[0] len: {}", X[0].len());

        MLP { d, W, L, X, deltas }
    }

    fn _propagate(&mut self, sample_inputs: Vec<f64>, is_classification: bool) {

        for j in 0..sample_inputs.len() {
            self.X[0][j + 1] = sample_inputs[j];
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l]{
                let mut total = 0.0;
                
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][i][j] * self.X[l - 1][i];
                }

                if is_classification || l < self.L {
                    total = total.tanh()
                }

                self.X[l][j] = total;
            }
        }
    }

    fn predict(&mut self, sample_inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {

        println!("pred");

        self._propagate(sample_inputs, is_classification);
        let pred = self.X[self.L][1..].to_vec();
        pred
    }

    fn train(&mut self,
             all_samples_inputs: Vec<Vec<f64>>,
             all_samples_expected_outputs: Vec<Vec<f64>>,
             alpha: f64,
             nb_iter: i64,
             is_classification: bool)
    {
        println!("train");

        let mut rng = rand::thread_rng();

        for it in 0..=nb_iter {
            let k = rng.gen_range(0..all_samples_inputs.len());
            let samples_inputs: Vec<f64> = all_samples_inputs[k].clone();
            let sample_expected_outputs: Vec<f64> = all_samples_expected_outputs[k].clone();

            self._propagate(samples_inputs, is_classification);

            for j in 1..=self.d[self.L] {
                self.deltas[self.L][j] = self.X[self.L][j] - sample_expected_outputs[j - 1];
                if is_classification {
                    self.deltas[self.L][j] *= 1.0 - self.X[self.L][j] * self.X[self.L][j];
                }
            }

            for l in (2..=self.L).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.W[l][i][j] * self.X[l - 1][i];
                    }
                    total *= 1.0 - self.X[l - 1][i] * self.X[l - 1][i];
                }
            }

            for l in 1..=self.L {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
    }
}

#[no_mangle]
extern "C" fn create_mlp_model(arr_ptr: *mut i64, arr_len: usize) -> *mut MLP {

    println!("create");

    // Créer un slice mutable à partir du pointeur `arr` en utilisant la fonction `from_raw_parts_mut`.
    let slice: &mut [i64] = unsafe { std::slice::from_raw_parts_mut(arr_ptr, arr_len) };

    // Convertir le slice mutable en Vec<usize> en clonant chaque élément de l'ancien type vers le nouveau type.
    let arr: Vec<usize> = slice.iter().map(|&x| x as usize).collect();

    let mut model_box = Box::new(MLP::new(arr));

    Box::into_raw(model_box)
}


#[no_mangle]
extern "C" fn train_mlp_model(model: *mut MLP,
                              dataset_inputs: *const f64,
                              lines: i64,
                              columns: i64,
                              dataset_outputs: *const f64,
                              output_columns: i64,
                              alpha: f64,
                              nb_iter: i64,
                              is_classification: bool) {
    unsafe {
        // Vérifier si le pointeur du modèle n'est pas nul
        if model.is_null() {
            panic!("Le pointeur du modèle est nul !");
        }

        println!("train python");

        // Convertir les pointeurs des données d'entrée et de sortie en slices
        let inputs_slice = std::slice::from_raw_parts(dataset_inputs, (lines * columns) as usize);
        let outputs_slice = std::slice::from_raw_parts(dataset_outputs, (lines * output_columns) as usize);

        println!("{}", (lines * columns) as usize);
        println!("line {}", (lines) as usize);
        println!("col {}", (columns) as usize);

        // Convertir les slices en Vec<Vec<f64>> pour les entrées et les sorties
        let mut all_samples_inputs: Vec<Vec<f64>> = Vec::new();
        let mut all_samples_outputs: Vec<Vec<f64>> = Vec::new();

        for i in 0..lines {
            let mut input_row = Vec::new();
            for j in 0..columns {
                input_row.push(inputs_slice[(i * columns + j) as usize]);
            }
            all_samples_inputs.push(input_row);
        }

        for i in 0..lines {
            let mut output_row = Vec::new();
            for j in 0..output_columns {
                output_row.push(outputs_slice[(i * output_columns + j) as usize]);
            }
            all_samples_outputs.push(output_row);
        }

        // Appeler la méthode train du modèle MLP
        (*model).train(all_samples_inputs,
                       all_samples_outputs,
                       alpha,
                       nb_iter,
                       is_classification);
    }
}

#[no_mangle]
extern "C" fn predict_mlp_model(model: *mut MLP,
                                sample_inputs: *const f64,
                                lines: i64,
                                columns: i64,
                                is_classification: bool) -> *mut f64 {
    unsafe {
        // Vérifier si le pointeur du modèle n'est pas nul
        if model.is_null() {
            panic!("Le pointeur du modèle est nul !");
        }

        println!("pred python");

        // Convertir le pointeur des échantillons d'entrée en slice
        let inputs_slice = std::slice::from_raw_parts(sample_inputs, (lines * columns) as usize);

        // Convertir les slices en Vec<Vec<f64>> pour les entrées et les sorties
        let mut all_samples_inputs: Vec<Vec<f64>> = Vec::new();

        for i in 0..lines {
            let mut input_row = Vec::new();
            for j in 0..columns {
                input_row.push(inputs_slice[(i * columns + j) as usize]);
            }
            all_samples_inputs.push(input_row);
        }

        let mut pred: Vec<Vec<f64>> = vec![];

        for k in 0..all_samples_inputs.len() {
            println!("pred {}", k);
            let pred_value = (*model).predict(all_samples_inputs[k].clone(), is_classification);
            pred.push(pred_value);
            println!("{:?}", pred);
        }

        println!("Prediction : {:?}", pred);

        // Aplatir pred en un Vec<f64>
        let pred_flattened: Vec<f64> = pred.into_iter().flatten().collect();

        println!("Prediction_flat : {:?}", pred_flattened);

        // Convertir pred_flattened en un tableau boxé et obtenir un pointeur vers le premier élément
        let output_ptr = Box::into_raw(pred_flattened.into_boxed_slice()) as *mut f64;

        output_ptr
    }
}

#[no_mangle]
extern "C" fn delete_mlp_model(model: *mut MLP) {
    println!("del");

    unsafe {
        let _ = Box::from_raw(model);
    }
}


#[cfg(test)]
mod tests {
    use crate::MLP;

    #[test]
    fn test_train() {
        // Définition des listes X et Y
        /*let X: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ];

        let Y: Vec<Vec<f64>> = vec![
            vec![-1.0],
            vec![1.0],
            vec![1.0],
            vec![-1.0],
        ];*/

        let X: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0],
            vec![2.0, 3.0],
            vec![3.0, 3.0],
        ];

        let Y: Vec<Vec<f64>> = vec![
            vec![1.0],
            vec![-1.0],
            vec![-1.0],
        ];

        println!("new");
        let mut model = MLP::new(vec![2, 1]);

        println!("train");
        model.train(X.clone(), Y.clone(), 0.1, 100000, true);

        println!("pred");
        for k in 0..X.len() {
            println!("pred {}", k);
            let pred = model.predict(X[k].clone(), true);
            println!("{:?}", pred);
        }
    }
}