use anyhow::{Context, Result};
use dialoguer::{Confirm, FuzzySelect};
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Col, Mat, MatRef, Par};
use mnist::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra_lapack::QrDecomposition;
use plotters::prelude::*;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

const EPSILON: f64 = 1e-8;
const N_TRAINING_SET: u32 = 1000;
const N_TESTING_SET: u32 = 10000;
const PCA_COMPONENTS: usize = 30;
const EPOCHS: usize = 1500;
const ALPHA: f64 = 0.5; // Learning Rate

fn main() -> Result<()> {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data_sets/mnist/")
        .label_format_digit()
        .training_set_length(N_TRAINING_SET)
        .test_set_length(N_TESTING_SET)
        .finalize();

    select_train_or_infer(&trn_img, &trn_lbl, &tst_img, &tst_lbl)?;

    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    n_train: u32,
    n_features: usize,
    bias: bool,
    pca: Option<Pca>,
    model: ModelType,
}

#[derive(Serialize, Deserialize, Debug)]
enum ModelType {
    LinearRegression {
        weights: DMatrix<f64>,
        epsilon: f64,
    },

    LogisticRegression {
        weights: DMatrix<f64>,
        learning_rate: f64,
        epochs: usize,
    },
}

impl Model {
    fn new(n_features: usize, bias: bool, pca: Option<Pca>, model: ModelType) -> Model {
        Model {
            n_train: N_TRAINING_SET,
            n_features,
            bias,
            pca,
            model,
        }
    }
}

// Convert logits to probabilities
fn softmax(logits: &DMatrix<f64>) -> DMatrix<f64> {
    let mut p = DMatrix::zeros(logits.nrows(), logits.ncols());

    for i in 0..logits.nrows() {
        let row = logits.row(i);

        let max_val = row.max();
        let stable_row = row.map(|val| (val - max_val).exp());

        let sum: f64 = stable_row.sum();

        // 3. Divide by the sum to get probabilities
        p.set_row(i, &(stable_row / sum));
    }

    p
}

fn cross_entropy_loss(p: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
    let mut loss = 0.0;

    for i in 0..p.nrows() {
        for j in 0..p.ncols() {
            if y[(i, j)] > 0.0 {
                loss -= y[(i, j)] * p[(i, j)].ln();
            }
        }
    }

    loss / p.nrows() as f64
}

// x is input matrix, y is the one hot matrix
// epoch is one round of learning
fn logistic_regression(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Model {
    let mut weights = DMatrix::zeros(x.ncols(), 10);
    for epoch in 1..=EPOCHS {
        let logits = x * &weights;
        let p = softmax(&logits);
        let loss = cross_entropy_loss(&p, y);
        println!("epoch: {epoch} | loss: {loss}");
        let error = &p - y;
        let gradient = x.transpose() * error / x.nrows() as f64;
        weights -= gradient * ALPHA;
    }

    let model_type = ModelType::LogisticRegression {
        weights,
        learning_rate: ALPHA,
        epochs: EPOCHS,
    };

    Model::new(x.ncols(), true, None, model_type)
}

// Saves weights to a new folder based on model
fn save_weights(model: Model) -> Result<()> {
    let path = std::path::Path::new("./weights");
    std::fs::create_dir_all(path)?;

    let pca = if model.pca.is_some() {
        String::from("true")
    } else {
        String::from("false")
    };

    let filename = match &model.model {
        ModelType::LinearRegression { weights, epsilon } => {
            let folder = String::from("linear_regression");
            std::fs::create_dir_all(format!("weights/{}", folder))?;
            format!(
                "weights/{}/linear_{}_{}_pca-{}.json",
                folder, model.n_train, epsilon, pca
            )
        }
        ModelType::LogisticRegression {
            weights: _,
            learning_rate,
            epochs,
        } => {
            let folder = String::from("logistic_regression");
            std::fs::create_dir_all(format!("weights/{}", folder))?;
            format!(
                "weights/{}/logistic_{}_{}_{}",
                folder, model.n_train, learning_rate, epochs
            )
        }
    };

    let file = File::create(filename).context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, &model)
        .context("Failed to serialize weights into JSON format")?;

    Ok(())
}

fn get_weights() -> Result<Model> {
    let folder = FileDialog::new()
        .set_title("Select the weights")
        .pick_file();

    match folder {
        Some(path) => {
            let file = File::open(path)?;
            let weights_json = BufReader::new(file);
            let weights: Model = serde_json::from_reader(weights_json)
                .context("Failed to deserialize weights JSON")?;
            Ok(weights)
        }
        None => Err(anyhow::anyhow!("No file selected")),
    }
}

fn inference(x: &DMatrix<f64>, w: &DMatrix<f64>) -> DVector<usize> {
    let scores = x * w;

    // Prepare an empty vector to store the predicted digits
    let mut predictions = DVector::zeros(scores.nrows());

    // Figure out which value has the highest probability in each row.
    // Each row has 10 probabilities representing each digit
    for (index, row) in scores.row_iter().enumerate() {
        let (best_digit_index, _best_digit) = row.transpose().argmax();
        predictions[index] = best_digit_index;
    }

    predictions
}

// Converts the digits into one-hot encoding
// the digit is represented as a row of binary numbers, where 1 and its index in the row indicates
// the digit, ie, 0 0 1 0 0 0 0 0 0 0 is the digit 2
fn one_hot_encode(trn_labels: &[u8]) -> DMatrix<f64> {
    let num_samples = trn_labels.len();
    let mut one_hot = DMatrix::zeros(num_samples, 10);

    for (lbl_idx, &label) in trn_labels.iter().enumerate() {
        one_hot[(lbl_idx, label as usize)] = 1.0;
    }

    one_hot
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Pca {
    mean: DVector<f64>,
    components: DMatrix<f64>,
}

fn save_pca(pca: Pca) -> Result<()> {
    let path = std::path::Path::new("./pca");
    std::fs::create_dir_all(path)?;

    let filename = "pca/pca.json";
    let file = File::create(filename).context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, &pca)
        .context("Failed to serialize PCA into JSON format")?;

    Ok(())
}

impl Pca {
    fn k_components(&mut self, k: usize) -> Pca {
        self.components = self.components.columns(0, k).into_owned();
        self.to_owned()
    }
}

fn fit_pca(trn_img: &[u8]) -> Pca {
    let matrix = DMatrix::from_row_slice(60000, 784, trn_img).map(|pixel| pixel as f64 / 255.0);
    let m = matrix.nrows();
    let n = matrix.ncols();

    // Calculate column means
    let mut means = vec![0.0; n];

    for j in 0..n {
        let mut sum = 0.0;

        for i in 0..m {
            sum += matrix[(i, j)];
        }

        means[j] = sum / m as f64;
    }

    // Center the data using the means
    let centered = DMatrix::from_fn(m, n, |i, j| matrix[(i, j)] - means[j]);

    // make sure not to divide by 0
    let denominater = if m > 1 { (m - 1) as f64 } else { 1.0 };
    // Build covariance matrix which represents how feature change together
    // The diagonal represents the variance of each specific feature
    let covariance_matrix = (&centered.transpose() * &centered) / denominater;

    // Get the eigendecomposition, eigenvectors represent the directions of maximum variance
    // The eigenvalues represent the amount of variance in an eigenvector
    let eigen = SymmetricEigen::new(covariance_matrix);

    let mut eigenpairs: Vec<(f64, DVector<f64>)> = (0..n)
        .map(|i| {
            (
                eigen.eigenvalues[i],
                eigen.eigenvectors.column(i).into_owned(),
            )
        })
        .collect();

    // Sort by eigenvalue, largest first to denote which direction has the most
    // variance/significance
    eigenpairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Sets the number of components
    let k = 784;

    let components = DMatrix::from_columns(
        &eigenpairs[..k]
            .iter()
            .map(|(_, v)| v.clone())
            .collect::<Vec<_>>(),
    );

    Pca {
        mean: DVector::from_vec(means),
        components,
    }
}

fn pca_transform(pca: &mut Pca, matrix: &DMatrix<f64>, k_components: usize) -> DMatrix<f64> {
    let matrix = matrix.clone_owned();
    let centered = DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        matrix[(i, j)] - pca.mean[j]
    });

    let pca = pca.k_components(k_components);

    centered * &pca.components
}

// Tolerance (Epsilon) is set internally by Faer
// fn svd_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
//     let svd = matrix.thin_svd().unwrap();
//
//     let pseudo_inverse = svd.pseudoinverse();
//     let solution = pseudo_inverse * vector;
//     let solution: Vec<f64> = solution.iter().copied().collect();
//
//     Weights::new(solution.as_slice(), digit, false)
// }

// fn qr_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
//     let qr = matrix.col_piv_qr();
//     let q = qr.compute_thin_Q();
//     let rt = qr.R().to_owned();
//
//     let rank = (0..rt.nrows().min(rt.ncols()))
//         .take_while(|&i| rt[(i, i)].abs() > EPSILON)
//         .count();
//
//     println!("Rank: {rank}");
//
//     // Q^T * b
//     let qtb = q.transpose() * vector;
//
//     let (qtb_truncated, _discard) = qtb.split_at_row(rank);
//     let mut qtb_truncated = qtb_truncated.to_owned();
//
//     // R (upper right triangle matrix)
//     let rt = rt.submatrix(0, 0, rank, rank);
//
//     solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::rayon(0));
//     // solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::Seq);
//
//     let mut permutated_y = qtb_truncated;
//     permutated_y.resize_with(matrix.ncols(), |_| 0.0);
//
//     let p = qr.P();
//
//     let (forward_idx, _inverse_idx) = p.arrays();
//
//     let mut x = Mat::<f64>::zeros(permutated_y.nrows(), 1);
//
//     for (i, &orig_col) in forward_idx.iter().enumerate() {
//         x[(orig_col, 0)] = permutated_y[i];
//     }
//
//     Weights::new(x.col_as_slice(0), digit, false)
// }

// Builds the pseudoinverse using SVD
fn svd_nalgebra_lapack(matrix: DMatrix<f64>) -> Result<DMatrix<f64>> {
    let matrix = matrix.insert_column(0, 1.0);

    let svd = nalgebra_lapack::SVD::new(matrix).unwrap();

    let rank = svd.rank(EPSILON);

    println!("Matrix Rank: {rank}");

    let ut = svd.u.transpose();

    // Trim U^T
    let mut sigma_inv_ut = ut.rows(0, rank).into_owned();

    // Multiply U^T by the inverted singular values to give E^-1 * U^T
    for i in 0..rank {
        let inv_sigma = 1.0 / svd.singular_values[i];

        for j in 0..sigma_inv_ut.ncols() {
            sigma_inv_ut[(i, j)] *= inv_sigma;
        }
    }

    // Trim V^T
    let vt = svd.vt.rows(0, rank).transpose();

    // Create pseudo inverse by multiplying V^T*E^-1*U^T
    let pseudoinverse = vt * sigma_inv_ut;

    Ok(pseudoinverse)
}

fn svd_nalgebra_lapack_pca(matrix: DMatrix<f64>) -> Result<DMatrix<f64>> {
    let mut pca = open_pca()?;
    let transformed_matrix = pca_transform(&mut pca, &matrix, PCA_COMPONENTS);
    let transformed_matrix = transformed_matrix.insert_column(0, 1.0);
    let pseudo_inverse = transformed_matrix.pseudo_inverse(EPSILON).unwrap();
    let rank = pseudo_inverse.rank(EPSILON);
    println!("Matrix Rank: {rank}");

    Ok(pseudo_inverse)
}

// Returns the factorized A matrix as QR
fn qr_nalgebra_lapack_pca(
    matrix: DMatrix<f64>,
) -> Result<nalgebra_lapack::QR<f64, nalgebra::Dyn, nalgebra::Dyn>> {
    let mut pca = open_pca()?;
    let transformed_matrix = pca_transform(&mut pca, &matrix, PCA_COMPONENTS);
    let transformed_matrix = transformed_matrix.insert_column(0, 1.0);
    let qr = nalgebra_lapack::QR::new(transformed_matrix)?;
    Ok(qr)
}

// QR nAlgebra without PCA
// Returns a tuple containing a trimmed version of Q and R matrices
fn qr_nalgebra_lapack(
    x: DMatrix<f64>,
) -> (
    DMatrix<f64>,
    DMatrix<f64>,
    nalgebra::PermutationSequence<nalgebra::Dyn>,
) {
    let qr = x.col_piv_qr();

    let (q, rt, p) = qr.unpack();
    let qt = q.transpose();

    let rank = (0..rt.nrows().min(rt.ncols()))
        .take_while(|&i| rt[(i, i)].abs() > EPSILON)
        .count();

    let qt_trimmed = qt.rows(0, rank).into_owned();
    let rt_trimmed = rt.view((0, 0), (rank, rank)).into_owned();

    (qt_trimmed, rt_trimmed, p)
}

#[derive(Debug)]
struct F1 {
    digit: u8,
    n_train: u32,
    epsilon: f64,
    tpos: f32,
    fpos: f32,
    fneg: f32,
}

impl F1 {
    fn new(digit: u8, n_train: u32, epsilon: f64) -> F1 {
        F1 {
            digit,
            n_train,
            epsilon,
            tpos: 0.0,
            fpos: 0.0,
            fneg: 0.0,
        }
    }

    fn precision(&self) -> f64 {
        (self.tpos / (self.tpos + self.fpos)) as f64
    }

    fn recall(&self) -> f64 {
        (self.tpos / (self.tpos + self.fneg)) as f64
    }

    fn f1(&self) -> f64 {
        2.0 * ((self.precision() * self.recall()) / (self.precision() + self.recall()))
    }
}

fn select_train_or_infer(
    trn_img: &[u8],
    trn_lbl: &[u8],
    tst_img: &[u8],
    tst_lbl: &[u8],
) -> Result<()> {
    loop {
        let items = vec!["Train Digits", "Inference", "Build PCA", "Exit"];
        let selection = FuzzySelect::new()
            .with_prompt("Select and option:")
            .items(&items)
            .interact()?;

        match selection {
            0 => {
                let library = select_training_library()?;
                let method = select_training_method()?;
                let use_pca = use_pca()?;
                train_all_digits(trn_img, trn_lbl, library, method, use_pca)?;
            }
            1 => {
                let weights = get_weights()?;

                // let file = File::open("logistic.json")?;
                // let weights: Vec<Vec<f64>> = serde_json::from_reader(file)?;
                let weights = match weights.model {
                    ModelType::LogisticRegression {
                        weights,
                        learning_rate: _,
                        epochs: _,
                    } => weights,
                    ModelType::LinearRegression { weights, epsilon } => weights,
                };

                digit_inference(tst_img, tst_lbl, weights)?;
            }
            2 => {
                let pca = fit_pca(trn_img);
                save_pca(pca)?;
            }
            _ => break,
        }
    }

    Ok(())
}

enum Method {
    SVD,
    QR,
    Logistic,
}

enum Library {
    NAlgebra,
    Faer,
}

fn svd_train_digits(pseudo_inverse: DMatrix<f64>, trn_lbl: &[u8], use_pca: bool) -> Result<()> {
    let mut all_weights = DMatrix::zeros(pseudo_inverse.nrows(), 10);
    for i in 0..=9 {
        let train_label =
            DVector::from_row_slice(trn_lbl).map(|digit| if digit == i { 1.0 } else { 0.0 });
        let weights = &pseudo_inverse * train_label;
        all_weights.set_column(i as usize, &weights);
    }

    let model_type = ModelType::LinearRegression {
        weights: all_weights,
        epsilon: EPSILON,
    };

    let model = Model::new(785, true, None, model_type);
    save_weights(model)?;

    Ok(())
}

fn train_all_digits(
    trn_img: &[u8],
    trn_lbl: &[u8],
    library: Library,
    method: Method,
    use_pca: bool,
) -> Result<()> {
    match library {
        Library::NAlgebra => {
            let train_data = prepare_trn_img_nalgebra(trn_img);
            match method {
                Method::SVD => {
                    let start = Instant::now();

                    let pseudo_inverse = if use_pca {
                        svd_nalgebra_lapack_pca(train_data)?
                    } else {
                        svd_nalgebra_lapack(train_data)?
                    };

                    svd_train_digits(pseudo_inverse, trn_lbl, use_pca)?;
                    println!("Time elapsed: {:?}", start.elapsed());
                }
                Method::QR => {
                    let start = Instant::now();

                    if use_pca {
                        let qr = qr_nalgebra_lapack_pca(train_data)?;
                        let mut all_weights = DMatrix::zeros(785, 10);
                        for digit in 0..=9 {
                            println!("Training {digit}");
                            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                            let weights = qr.solve(train_label)?;
                            all_weights.set_column(digit as usize, &weights);
                        }
                        let model_type = ModelType::LinearRegression {
                            weights: all_weights,
                            epsilon: EPSILON,
                        };

                        let model = Model::new(785, true, None, model_type);
                        save_weights(model)?;
                    } else {
                        let train_data = train_data.insert_column(0, 1.0);
                        let n_features = train_data.ncols();
                        let (q, rt, p) = qr_nalgebra_lapack(train_data);
                        let mut all_weights = DMatrix::zeros(n_features, 10);
                        for digit in 0..=9 {
                            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                            let qtb = &q * train_label;
                            let weights = rt.solve_upper_triangular(&qtb).unwrap();
                            let mut weights = weights.resize_vertically(785, 0.0);
                            p.inv_permute_rows(&mut weights);
                            all_weights.set_column(digit as usize, &weights);
                        }
                        let model_type = ModelType::LinearRegression {
                            weights: all_weights,
                            epsilon: EPSILON,
                        };
                        let model = Model::new(785, true, None, model_type);
                        save_weights(model)?;
                    }

                    println!("QR elapsed: {:?}", start.elapsed());
                }
                Method::Logistic => {
                    let y = one_hot_encode(trn_lbl);
                    let train_data = train_data.insert_column(0, 1.0);
                    let weights = logistic_regression(&train_data, &y);
                    save_weights(weights)?;
                }
            }
        }
        Library::Faer => match method {
            Method::SVD => {
                todo!();
                // let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
                // // let z = pca(train_data.clone());
                // svd_least_squares_faer(train_data, train_label, i)
            }
            Method::QR => {
                todo!();
                // let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
                // qr_least_squares_faer(train_data, train_label, i)
            }
            Method::Logistic => {
                todo!()
            }
        },
    };

    Ok(())
}

fn use_pca() -> dialoguer::Result<bool> {
    let pca: bool = Confirm::new()
        .with_prompt("Use Principle Component Analysis?")
        .interact()?;

    Ok(pca)
}

fn select_training_library() -> Result<Library> {
    let items = vec!["Faer", "nAlgebra"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Library::Faer),
        1 => Ok(Library::NAlgebra),
        _ => todo!(),
    }
}

fn select_training_method() -> Result<Method> {
    let items = vec!["SVD", "QR", "Logistic"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Method::SVD),
        1 => Ok(Method::QR),
        2 => Ok(Method::Logistic),
        _ => todo!(),
    }
}

fn prepare_trn_img_nalgebra(trn_img: &[u8]) -> DMatrix<f64> {
    DMatrix::from_row_slice(N_TRAINING_SET as usize, 784, trn_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0)
}

fn prepare_trn_lbl_nalgebra(trn_lbl: &[u8], digit_to_train: u8) -> DVector<f64> {
    DVector::from_row_slice(trn_lbl).map(|digit| if digit == digit_to_train { 1.0 } else { 0.0 })
}

fn prepare_train_data_faer(
    trn_img: &[u8],
    trn_lbl: &[u8],
    digit_to_train: u8,
) -> Result<(Mat<f64>, Col<f64>)> {
    let train_data = MatRef::from_row_major_slice(trn_img, N_TRAINING_SET as usize, 784)
        // .map(|pixel| if *pixel > 0 { 1.0 } else { 0.0 });
        .map(|pixel| *pixel as f64 / 255.0);

    // Add bias term in the form of a column of 1's
    let bias_col = Mat::from_fn(train_data.nrows(), 1, |_, _| 1.0);

    let train_data = faer::concat![[bias_col, train_data]];

    let train_label = Col::from_fn(N_TRAINING_SET as usize, |i| trn_lbl[i])
        .map(|digit| if *digit == digit_to_train { 1.0 } else { 0.0 });

    Ok((train_data, train_label))
}

fn open_pca() -> Result<Pca> {
    let filename = "pca/pca.json";
    let file = File::open(filename)?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], weights: DMatrix<f64>) -> Result<()> {
    let mut test_data = DMatrix::from_row_slice(N_TESTING_SET as usize, 784, tst_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0);

    // let n_train = weights.first().unwrap().n_train.unwrap();
    // let epsilon = weights.first().unwrap().epsilon.unwrap();
    // let is_pca = weights.first().unwrap().pca;

    // if is_pca {
    //     let mut pca = open_pca()?;
    //     test_data = pca_transform(&mut pca, &test_data, PCA_COMPONENTS);
    // }

    test_data = test_data.insert_column(0, 1.0);
    let mut metrics: Vec<F1> = (0..10)
        .map(|digit| F1::new(digit, N_TRAINING_SET, EPSILON))
        .collect();

    let predictions = inference(&test_data, &weights);
    let mut score = 0;
    for i in 0..predictions.nrows() as usize {
        for metric in &mut metrics {
            if predictions[i] == metric.digit as usize && tst_lbl[i] == metric.digit {
                metric.tpos += 1.0;
            } else if predictions[i] == metric.digit as usize && tst_lbl[i] != metric.digit {
                metric.fpos += 1.0;
            } else if predictions[i] != metric.digit as usize && tst_lbl[i] == metric.digit {
                metric.fneg += 1.0;
            }
        }
        if tst_lbl[i] == predictions[i] as u8 {
            score += 1
        };
    }

    for digit in &metrics {
        println!(
            "Digit: {}\nPrecision: {}\nRecall: {}\nF1: {}\n",
            digit.digit,
            digit.precision(),
            digit.recall(),
            digit.f1()
        );
    }

    let average_f1 = metrics.iter().fold(0.0, |acc, digit| acc + digit.f1());
    let average_f1 = average_f1 / metrics.len() as f64;

    f1_scatterplot(metrics)?;

    let total_scores = N_TESTING_SET;

    let percent_correct = (score as f32 / N_TESTING_SET as f32) * 100.0;

    println!(
        "Number of Tests: {}\n Number correct: {}\n Percent Correct: {:.2}%\n Average F1: {}",
        total_scores, score, percent_correct, average_f1
    );

    Ok(())
}

// Creates a scatterplot where
// - precision of digit is the x-axis
// - recall is x-axis
// - radius of point is F1 score

fn f1_scatterplot(metrics: Vec<F1>) -> Result<()> {
    let path = std::path::Path::new("./scatterplots");
    std::fs::create_dir_all(path)?;

    let n_train = metrics.first().unwrap().n_train;
    let epsilon = metrics.first().unwrap().epsilon;

    let scatterplot_folder = format!("scatterplot {}_{}", n_train, epsilon);
    let scatterplot_folder = path.join(scatterplot_folder);

    let filename = format!("{}.png", scatterplot_folder.display(),);
    let root = BitMapBackend::new(&filename, (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Digit Classification Performance", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1.0, 0f64..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Precision")
        .y_desc("Recall")
        .axis_desc_style(("sans-serif", 20).into_font())
        .label_style(("sans-serif", 12).into_font())
        .draw()?;

    chart.draw_series(metrics.iter().map(|m| {
        Circle::new(
            (m.precision(), m.recall()),
            (m.f1() * 30.0) as i32,
            BLACK.filled(),
        )
    }))?;

    chart.draw_series(metrics.iter().map(|m| {
        Text::new(
            format!("{}", m.digit),
            (m.precision() - 0.003, m.recall() + 0.007),
            ("sans-serif", 20)
                .into_font()
                .color(&RGBColor(255, 255, 255)),
        )
    }))?;

    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::{Col, MatRef};

    #[test]
    fn test_svd_least_squares() {
        let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
        let digit = 0;
        let epsilon = 1e-12;

        let pseudoinverse = svd_nalgebra_lapack(x).unwrap();
        let solution = pseudoinverse * y;
        let result = Weights::new(solution.as_slice(), digit, false);
        // let result = svd_least_squares(&x, &y, digit, epsilon);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = 0.001);
    }

    #[test]
    fn test_svd_least_squares_faer() {
        let x = [1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        let matrix = MatRef::from_row_major_slice(&x, 3, 2).to_owned();
        // let matrix = Mat::from_fn(3, 2, |i, j| x[i * 3 + j]);
        let y = [2.0, 3.0, 7.0];
        let vector = Col::from_fn(3, |i| y[i]);
        let digit = 0;

        let weights = svd_least_squares_faer(matrix, vector, digit).weights;
        assert_relative_eq!(weights[..], &vec![2.5, -1.0], epsilon = 0.001);
    }

    #[test]
    fn test_qr_least_squares_faer() {
        let x = [1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        let matrix = MatRef::from_row_major_slice(&x, 3, 2).to_owned();
        // let matrix = Mat::from_fn(3, 2, |i, j| x[i * 3 + j]);
        let y = [2.0, 3.0, 7.0];
        let vector = Col::from_fn(3, |i| y[i]);
        let digit = 0;

        let weights = qr_least_squares_faer(matrix, vector, digit).weights;
        assert_relative_eq!(weights[..], &vec![2.5, -1.0], epsilon = 0.001);
    }

    #[test]
    fn test_f1() {
        let f1_example = F1 {
            digit: 0,
            epsilon: 1.0,
            n_train: 0,
            tpos: 8.0,
            fpos: 7.0,
            fneg: 2.0,
        };

        let precision = f1_example.precision();
        let recall = f1_example.recall();
        let f1 = f1_example.f1();

        assert_relative_eq!(precision, 0.533, epsilon = 0.001);
        assert_relative_eq!(recall, 0.80, epsilon = 0.001);
        assert_relative_eq!(f1, 0.64, epsilon = 0.001);
    }
}
