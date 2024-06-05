use rand::Rng;

#[repr(C)]
pub struct LinearModel {
    weights: *mut f64,
    bias: *mut f64,
    loss: *mut f64,
    loss_size: usize,
}

#[repr(C)]
pub struct LossMonitor {
    last_loss: f64,
    loss_change_threshold: f64,
    min_frequency: usize,
    max_frequency: usize,
    current_frequency: usize,
    last_loss_iteration: usize,
}


// Fonction pour entraîner le modèle linéaire
fn train(inputs: &[Vec<f64>], targets: &[Vec<f64>], k: usize, learning_rate: f64, n_features: usize, num_iterations: usize, isclassification: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut weights: Vec<f64> = vec![0.0; n_features * k];
    let mut bias: Vec<f64> = vec![0.0; k];
    let mut loss_values: Vec<f64> = vec![];

    let mut loss_monitor = LossMonitor {
        last_loss: f64::INFINITY,
        loss_change_threshold: 0.01,
        min_frequency: 100,
        max_frequency: 10000,
        current_frequency: 1000,
        last_loss_iteration: 0,
    };

    // Boucle pour les itérations d'entraînement
    for iteration in 0..=num_iterations {
        // Sélection aléatoire d'un échantillon d'entrée
        let i = rand::thread_rng().gen_range(0..inputs.len());
        let input = &inputs[i];
        let target = &targets[i];

        let mut output: Vec<f64> = vec![0.0; k];

        // Calculer la sortie pour chaque classe
        for j in 0..n_features {
            for c in 0..k {
                output[c] += input[j] * weights[j + n_features * c];
            }
        }

        for c in 0..k {
            output[c] += bias[c];
        }

        let mut errors: Vec<f64> = vec![0.0; k];

        // Calculer les erreurs pour chaque classe
        for c in 0..k {
            errors[c] = target[c] - output[c];
        }

        // Mettre à jour les poids et le biais du modèle
        for j in 0..n_features {
            for c in 0..k {
                weights[j + n_features * c] += learning_rate * input[j] * errors[c];
            }
        }

        for c in 0..k {
            bias[c] += learning_rate * errors[c];
        }

        // Utiliser LossMonitor pour décider quand calculer la perte
        if iteration % loss_monitor.current_frequency == 0 || iteration == num_iterations {
            let loss = calculate_loss(&weights, &bias, &inputs, &targets, isclassification);
            loss_values.push(loss);  // Ajouter la perte au vecteur loss_values
            println!("Itération: {}, Perte: {}", iteration, loss);
            
            // Mise à jour de LossMonitor
            update_loss_monitor(&mut loss_monitor, loss, iteration);
        }
    }

    (weights, bias, loss_values)
}

// Fonction pour calculer la perte (loss) du modèle linéaire
fn calculate_loss(weights: &[f64], bias: &[f64], inputs: &[Vec<f64>], targets: &[Vec<f64>], isclassification: bool) -> f64 {
    let k = bias.len();
    let n_samples = inputs.len();
    let mut loss = 0.0;

    for i in 0..n_samples {
        let input = &inputs[i];
        let target = &targets[i];
        let prediction = predict(weights, bias, input, isclassification);

        // Calculer la perte pour chaque classe prédite
        for c in 0..k {
            let error = target[c] - prediction[c];
            loss += error * error;
        }
    }

    loss / (n_samples * k) as f64
}

fn update_loss_monitor(monitor: &mut LossMonitor, current_loss: f64, iteration: usize) {
    let loss_change = (monitor.last_loss - current_loss).abs();

    if loss_change < monitor.loss_change_threshold {
        monitor.current_frequency = std::cmp::min(monitor.current_frequency * 2, monitor.max_frequency);
    } else {
        monitor.current_frequency = std::cmp::max(monitor.current_frequency / 2, monitor.min_frequency);
    }

    monitor.last_loss = current_loss;
    monitor.last_loss_iteration = iteration;
}

// Fonction pour prédire les classes à partir des poids, biais et entrées données
fn predict(weights: &[f64], bias: &[f64], input: &[f64], isclassification: bool) -> Vec<f64> {
    let k = bias.len();
    let n_features = input.len();
    let mut prediction: Vec<f64> = vec![0.0; k];

    // Calculer la prédiction pour chaque classe
    for j in 0..n_features {
        for c in 0..k {
            prediction[c] += input[j] * weights[j + n_features * c];
        }
    }

    // Ajouter le biais pour chaque classe
    for c in 0..k {
        prediction[c] += bias[c];
    }

    // Appliquer la fonction d'activation tanh si c'est un problème de classification
    if isclassification {
        for c in 0..k {
            prediction[c] = prediction[c].tanh();
        }
    }

    prediction
}

// Fonction de prédiction pour le modèle linéaire, accessible depuis l'extérieur du code Rust
#[no_mangle]
pub extern "C" fn predict_linear_model(features: *const f64, weights: *const f64, bias: *const f64, _num_samples: usize, num_features: usize, k: usize, isclassification: bool) -> *mut f64 {
    // Convertir les pointeurs en slices pour les données d'entrée
    unsafe {
        let features_slice = std::slice::from_raw_parts(features, 1 * num_features);
        let weights_slice = std::slice::from_raw_parts(weights, num_features * k);
        let bias_slice = std::slice::from_raw_parts(bias, k);
        let inputs: Vec<_> = features_slice.chunks(num_features).map(|chunk| chunk.to_vec()).collect();
        let weights = weights_slice.to_vec();
        let bias = bias_slice.to_vec();

        let target = predict(&weights, &bias, &inputs[0], isclassification);
        Box::into_raw(target.into_boxed_slice()) as *mut f64
    }
}

// Fonction d'entraînement pour le modèle linéaire, accessible depuis l'extérieur du code Rust
#[no_mangle]
pub extern "C" fn train_linear_model(
    features: *const f64,
    outputs: *const f64,
    num_samples: usize,
    num_features: usize,
    learning_rate: f64,
    num_iterations: usize,
    k: usize,
    isclassification: bool,
) -> LinearModel {
    // Convertir les pointeurs en slices pour les données d'entrée et de sortie
    let slicer_inputs = unsafe { std::slice::from_raw_parts(features, num_samples * num_features) };
    let slicer_outputs = unsafe { std::slice::from_raw_parts(outputs, num_samples * k) };
    let mut inputs: Vec<Vec<f64>> = vec![vec![0.0; num_features]; num_samples];
    let mut targets: Vec<Vec<f64>> = vec![vec![0.0; k]; num_samples];

    // Remplir les vecteurs d'entrée et de sortie avec les données
    for i in 0..num_samples {
        for j in 0..num_features {
            inputs[i][j] = slicer_inputs[i * num_features + j];
        }
    }

    for i in 0..num_samples {
        for j in 0..k {
            targets[i][j] = slicer_outputs[i * k + j];
        }
    }

    // Appeler la fonction d'entraînement et récupérer les résultats
    let (mut weights, mut bias, mut loss_values) = train(&inputs, &targets, k, learning_rate, num_features, num_iterations, isclassification);

    // Convertir les vecteurs en pointeurs pour les valeurs à retourner
    let weights_ptr = weights.as_mut_ptr();
    let bias_ptr = bias.as_mut_ptr();
    let loss_ptr = loss_values.as_mut_ptr();
    let loss_size = loss_values.len();

    // Empêcher la libération de la mémoire pour les vecteurs en les "oubliant"
    std::mem::forget(weights);
    std::mem::forget(bias);
    std::mem::forget(loss_values);

    // Créer et retourner la structure LinearModel avec les pointeurs
    LinearModel {
        weights: weights_ptr,
        bias: bias_ptr,
        loss: loss_ptr,
        loss_size: loss_size,
    }
}

fn main() {

    // Données d'entraînement
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];

    // Étiquettes pour les données d'entraînement
    let targets = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0]
    ];

    // Entraîner le modèle
    let (weights, bias, _loss_values) = train(&inputs, &targets, 2, 0.01, 2, 100000, true);

    // Données de test
    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];

    // Prédictions attendues pour les données de test
    let expected_predictions = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0]
    ];

    // Tester le modèle sur les données de test
    for i in 0..test_inputs.len() {
        let prediction = predict(&weights, &bias, &test_inputs[i], true);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification() {
        main()
    }
}
