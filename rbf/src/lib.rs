use std::ffi::CStr;
use std::slice;
use rand::prelude::*;
use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize)]
pub struct RBFNetwork {
    centers: Vec<Point>,
    gamma: f64,
    weights: Vec<Vec<f64>>,
}

impl RBFNetwork {
    pub fn new(num_centers: usize, gamma: f64, k: usize) -> Self {
        RBFNetwork {
            centers: Vec::new(),
            gamma,
            weights: vec![vec![0.0; num_centers]; k], // +1 for the bias weight
        }
    }

    // Méthode pour effectuer l'algorithme K-means et initialiser les centres des clusters
    fn kmeans(&mut self, data: &[Point], num_centers: usize, num_iterations: usize) {
        let mut rng = thread_rng();

        // Initialisation aléatoire des indices des centres de cluster
        let center_indices: Vec<usize> = (0..data.len()).choose_multiple(&mut rng, num_centers);
        // Initialisation des centres de cluster en utilisant les indices choisis aléatoirement
        self.centers = center_indices
            .iter()
            .map(|&index| data[index].clone())
            .collect();

        for _ in 0..num_iterations {
            let mut cluster_assignments: Vec<Vec<Point>> = vec![Vec::new(); num_centers];

            // Assigner les points de données aux centres de cluster les plus proches
            for point in data {
                let mut min_distance = f64::INFINITY;
                let mut closest_center_index = 0;

                for (i, center) in self.centers.iter().enumerate() {
                    let distance = self.euclidean_distance(point, center);

                    if distance < min_distance {
                        min_distance = distance;
                        closest_center_index = i;
                    }
                }
                cluster_assignments[closest_center_index].push(point.clone());
            }

            // Mettre à jour les centres de cluster en utilisant la moyenne des points attribués à chaque centre
            for (i, cluster) in cluster_assignments.iter().enumerate() {
                if !cluster.is_empty() {
                    let num_points = cluster.len() as f64;
                    let sum_x: f64 = cluster.iter().map(|point| point.x).sum();
                    let sum_y: f64 = cluster.iter().map(|point| point.y).sum();
                    self.centers[i].x = sum_x / num_points;
                    self.centers[i].y = sum_y / num_points;
                }
            }
        }

        // Calcul de la distance maximale entre les centres de cluster
        let num_centers = self.centers.len();
        let mut max_distance = 0.0;
        for i in 0..num_centers {
            for j in i + 1..num_centers {
                let distance = self.euclidean_distance(&self.centers[i], &self.centers[j]);
                if distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        // Calcul de la valeur de gamma en fonction de la distance maximale
        self.gamma = max_distance / (num_centers as f64).sqrt();
    }

    // Méthode pour calculer la distance au carré entre deux points
    fn squared_distance(&self, p1: &Point, p2: &Point) -> f64 {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        let d = dx * dx + dy * dy;
        d
    }

    // Méthode pour calculer la distance euclidienne entre deux points
    fn euclidean_distance(&self, p1: &Point, p2: &Point) -> f64 {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        let d = dx * dx + dy * dy;
        d.sqrt()
    }

    // Méthode pour calculer l'activation d'un point de données pour un centre donné
    fn compute_activation(&self, data_point: &Point, center: &Point) -> f64 {
        let distance = self.squared_distance(data_point, center);
        (-distance * distance / (self.gamma * self.gamma)).exp()
    }

    // Méthode pour ajuster les poids par régression linéaire
    fn fit_linear_regression(&mut self, data: &[Point], targets: &[Vec<f64>], _learning_rate: f64) {
        let num_data = data.len();
        let num_centers = self.centers.len();
        let num_classes = targets[0].len();

        for class_idx in 0..num_classes {
            let mut a_matrix: Vec<f64> = Vec::new();
            let mut y_matrix: Vec<f64> = Vec::new();

            // Construction de la matrice A et de la matrice Y pour la régression linéaire
            for i in 0..num_data {
                let activations: Vec<f64> = self
                    .centers
                    .iter()
                    .map(|center| self.compute_activation(&data[i], center))
                    .collect();

                a_matrix.extend_from_slice(&activations);
                y_matrix.push(targets[i][class_idx]);
            }

            // Utilisation de la formule spécifique à RBF pour ajuster les poids
            let a_matrix_t = Self::matrix_transpose(&a_matrix, num_data, num_centers);
            let a_matrix_t_a = Self::matrix_multiply(&a_matrix_t, &a_matrix, num_centers, num_data, num_centers);
            let a_matrix_t_y = Self::matrix_multiply_vector(&a_matrix_t, &y_matrix, num_centers, num_data);

            let class_weights = Self::matrix_solve(&a_matrix_t_a, &a_matrix_t_y);
            self.weights[class_idx] = class_weights;
        }
    }

    // Méthode pour effectuer l'entraînement du réseau RBF
    pub fn fit(&mut self, data: &[Point], targets: &[Vec<f64>], num_iterations: usize, num_centers: usize) {
        // self.kmeans(data, num_centers, 100);
        self.kmeans(data, num_centers, num_iterations);
        for _ in 0..num_iterations {
            self.fit_linear_regression(data, targets, 0.0001);
        }
    }

    // Méthode pour effectuer une prédiction pour un point de données donné
    pub fn predict(&self, data_point: &Point, is_classification: bool) -> Vec<f64> {
        let mut activations = vec![0.0; self.weights.len()];
        for (i, class_weights) in self.weights.iter().enumerate() {
            let mut activation_sum = 0.0;
            for (j, center) in self.centers.iter().enumerate() {
                let activation = self.compute_activation(data_point, center);
                activation_sum += activation * class_weights[j];
            }
            activations[i] = activation_sum;
        }

        if is_classification {
            activations.iter().map(|&x| f64::tanh(x)).collect()
        } else {
            activations
        }
    }

    // Méthode pour transposer une matrice
    fn matrix_transpose(matrix: &[f64], num_rows: usize, num_cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; num_rows * num_cols];
        for i in 0..num_rows {
            for j in 0..num_cols {
                result[j * num_rows + i] = matrix[i * num_cols + j];
            }
        }
        result
    }

    // Méthode pour multiplier deux matrices
    fn matrix_multiply(mat1: &[f64], mat2: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
        let mut result = vec![0.0; n * m];

        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += mat1[i * k + p] * mat2[p * m + j];
                }
                result[i * m + j] = sum;
            }
        }

        result
    }

    // Méthode pour multiplier une matrice par un vecteur
    fn matrix_multiply_vector(matrix: &[f64], vector: &[f64], size: usize, num_data: usize) -> Vec<f64> {
        let mut result = vec![0.0; size];
        for i in 0..size {
            let mut sum = 0.0;
            for j in 0..num_data {
                sum += matrix[i * num_data + j] * vector[j];
            }
            result[i] = sum;
        }
        result
    }

    // Méthode pour résoudre un système d'équations linéaires à l'aide de la décomposition LU
    fn matrix_solve(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
        let matrix = DMatrix::from_row_slice(vector.len(), vector.len(), matrix);
        let vector = DMatrix::from_vec(vector.len(), 1, vector.to_vec());
        let solution = matrix.lu().solve(&vector).unwrap();
        solution.iter().cloned().collect()
    }

    // Méthode pour sauvegarder le modèle dans un fichier JSON
    pub fn save_to_file(&self, filename: &str) {
        let serialized = serde_json::to_string(self).unwrap();
        std::fs::write(filename, serialized).unwrap();
        println!("model save");
    }

    // Méthode pour charger le modèle à partir d'un fichier JSON
    pub fn load_from_file(filename: &str) -> Self {
        let data = std::fs::read_to_string(filename).unwrap();
        serde_json::from_str(&data).unwrap()
    }
}

// Fonction pour créer une instance de RBFNetwork et renvoyer un pointeur brut
#[no_mangle]
pub extern "C" fn rbf_network_new(num_centers: usize, gamma: f64, k: usize) -> *mut RBFNetwork {
    let raw = Box::into_raw(Box::new(RBFNetwork::new(num_centers, gamma, k)));
    raw
}

// Fonction pour libérer la mémoire associée à l'instance de RBFNetwork
#[no_mangle]
pub unsafe extern "C" fn rbf_network_free(ptr: *mut RBFNetwork) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

// Fonction pour sauvegarder le model associée dans un fichier JSON
#[no_mangle]
pub unsafe extern "C" fn rbf_network_save(ptr: *mut RBFNetwork, name: *const i8) {
    if !ptr.is_null() {
        let c_str = CStr::from_ptr(name);
        let name_str = c_str.to_str().unwrap();
        (*ptr).save_to_file(name_str)
    }
}

// Fonction pour effectuer l'entraînement du réseau RBF à partir des données fournies
#[no_mangle]
pub unsafe extern "C" fn rbf_network_fit(
    ptr: *mut RBFNetwork,
    data: *const Point,
    num_data: usize,
    targets: *const f64,
    num_targets: usize,
    num_iterations: usize,
    num_centers: usize,
    k: usize,
) {
    if let Some(rbf_network) = ptr.as_mut() {
        let data_slice = slice::from_raw_parts(data, num_data);
        let data_vec = data_slice.to_vec();
        let slicer_inputs = slice::from_raw_parts(targets, num_targets * k);

        let mut targets: Vec<Vec<f64>> = vec![vec![0.0; k]; num_targets];

        for i in 0..num_targets {
            for j in 0..k {
                targets[i][j] = slicer_inputs[i * k + j];
            }
        }

        rbf_network.fit(&data_vec, &targets, num_iterations, num_centers);
    }
}

// Fonction pour effectuer une prédiction à partir d'un point de données donné
#[no_mangle]
pub unsafe extern "C" fn rbf_network_predict(
    ptr: *const RBFNetwork,
    data_point: *const Point,
    _num_classes: usize,
) -> *mut f64 {
    unsafe {
        if let Some(rbf_network) = ptr.as_ref() {
            let data_point_ref = data_point.as_ref().unwrap();
            let prediction = rbf_network.predict(data_point_ref, true);
            let prediction_boxed = prediction.into_boxed_slice();
            let prediction_ptr = prediction_boxed.as_ptr();
            std::mem::forget(prediction_boxed);
            prediction_ptr as *mut f64
        } else {
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // 1. Préparer les données
        let data = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 3.0 },
            Point { x: 3.0, y: 3.0 },
        ];

        let targets = vec![
            vec![1.0, 0.0],  // Classe 1
            vec![0.0, 1.0],  // Classe -1
            vec![0.0, 1.0],  // Classe -1
        ];

        // 2. Initialiser le réseau RBF
        let num_centers = 2; // Nombre de centres
        let gamma = 1.0; // Valeur initiale de gamma
        let num_classes = 2; // Nombre de classes

        let mut rbf_network = RBFNetwork::new(num_centers, gamma, num_classes);

        // 3. Entraîner le réseau
        let num_iterations = 10000;
        rbf_network.fit(&data, &targets, num_iterations, num_centers);

        // Sauvegarde du modèle dans un fichier JSON
        rbf_network.save_to_file("rbf_model.json");

        // 4. Faire des prédictions
        let test_point = Point { x: 2.0, y: 3.0 };

        let data_test = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 3.0 },
            Point { x: 3.0, y: 3.0 },
        ];

        for point in data_test {
            let predictions = rbf_network.predict(&point, true);
            println!("Predictions for point ({}, {}): {:?}", point.x, point.y, predictions);

            // Décider de la classe
            let predicted_class = predictions
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            let class_label = if predicted_class == 0 { 1 } else { -1 };
            println!("Predicted class: {}", class_label);
        }
    }
}



