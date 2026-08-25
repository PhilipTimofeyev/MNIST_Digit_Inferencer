use anyhow::{Context, Result};
use dialoguer::FuzzySelect;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Col, Mat, MatRef, Par};
use mnist::*;
use nalgebra::{DMatrix, DVector, SVD, SymmetricEigen};
use nalgebra_lapack::{QrDecomposition, colpiv_qr};
use plotters::prelude::*;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

const EPSILON: f64 = 1e-6;
const N_TRAINING_SET: u32 = 60000;
const N_TESTING_SET: u32 = 10000;
const PCA_COMPONENTS: usize = 70;

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

    let centered = DMatrix::from_fn(m, n, |i, j| matrix[(i, j)] - means[j]);

    // make sure not to divide by 0
    let denominater = if m > 1 { (m - 1) as f64 } else { 1.0 };
    let covariance_matrix = (&centered.transpose() * &centered) / denominater;

    let eigen = SymmetricEigen::new(covariance_matrix);

    let mut eigenpairs: Vec<(f64, DVector<f64>)> = (0..n)
        .map(|i| {
            (
                eigen.eigenvalues[i],
                eigen.eigenvectors.column(i).into_owned(),
            )
        })
        .collect();

    // Sort by eigenvalue, largest first
    eigenpairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

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

// Not in use
#[allow(dead_code)]
fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>, digit: u8, epsilon: f64) -> Weights {
    let svd = SVD::new(x.clone(), true, true);
    let weights = svd.solve(y, epsilon).unwrap();

    Weights::new(weights.as_slice(), digit)
}

// Tolerance (Epsilon) is set internally by Faer
fn svd_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
    let svd = matrix.thin_svd().unwrap();

    let pseudo_inverse = svd.pseudoinverse();
    let solution = pseudo_inverse * vector;
    let solution: Vec<f64> = solution.iter().copied().collect();

    Weights::new(solution.as_slice(), digit)
}

fn qr_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
    let qr = matrix.col_piv_qr();
    let q = qr.compute_thin_Q();
    let rt = qr.R().to_owned();

    let rank = (0..rt.nrows().min(rt.ncols()))
        .take_while(|&i| rt[(i, i)].abs() > EPSILON)
        .count();

    println!("Rank: {rank}");

    // Q^T * b
    let qtb = q.transpose() * vector;

    let (qtb_truncated, _discard) = qtb.split_at_row(rank);
    let mut qtb_truncated = qtb_truncated.to_owned();

    // R (upper right triangle matrix)
    let rt = rt.submatrix(0, 0, rank, rank);

    solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::rayon(0));
    // solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::Seq);

    let mut permutated_y = qtb_truncated;
    permutated_y.resize_with(matrix.ncols(), |_| 0.0);

    let p = qr.P();

    let (forward_idx, _inverse_idx) = p.arrays();

    let mut x = Mat::<f64>::zeros(permutated_y.nrows(), 1);

    for (i, &orig_col) in forward_idx.iter().enumerate() {
        x[(orig_col, 0)] = permutated_y[i];
    }

    Weights::new(x.col_as_slice(0), digit)
}

fn svd_least_squares_lapack_old(x: DMatrix<f64>, y: &DVector<f64>, digit: u8) -> Weights {
    let svd = nalgebra_lapack::SVD::new(x).unwrap();

    let epsilon = 1e-4;

    let max_singular_value = svd.singular_values[0];

    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s > epsilon * max_singular_value)
        .count();

    println!("Rank: {}", rank);

    // Equation to solve is w = V * sigma^-1 * U^T * y
    let ut_y = svd.u.transpose() * y;

    let mut trimmed_ut_y = ut_y.rows(0, rank).into_owned();

    for i in 0..rank {
        trimmed_ut_y[i] /= svd.singular_values[i];
    }

    let vt = svd.vt.rows(0, rank);

    // Finally multiply by V to get the least squares solution
    let solution = vt.transpose() * trimmed_ut_y;

    Weights::new(solution.as_slice(), digit)
}

fn svd_least_squares_lapack(pseudo_inverse: &DMatrix<f64>, y: &DVector<f64>, digit: u8) -> Weights {
    // let svd = nalgebra_lapack::SVD::new(x).unwrap();
    // let pseudo_inverse = svd.pseudo_inverse(EPSILON);

    let solution = pseudo_inverse * y;

    // let epsilon = 1e-4;
    //
    // let max_singular_value = svd.singular_values[0];
    //
    // let rank = svd
    //     .singular_values
    //     .iter()
    //     .filter(|&&s| s > epsilon * max_singular_value)
    //     .count();
    //
    // println!("Rank: {}", rank);
    //
    // // Equation to solve is w = V * sigma^-1 * U^T * y
    // let ut_y = svd.u.transpose() * y;
    //
    // let mut trimmed_ut_y = ut_y.rows(0, rank).into_owned();
    //
    // for i in 0..rank {
    //     trimmed_ut_y[i] /= svd.singular_values[i];
    // }
    //
    // let vt = svd.vt.rows(0, rank);
    //
    // // Finally multiply by V to get the least squares solution
    // let solution = vt.transpose() * trimmed_ut_y;

    Weights::new(solution.as_slice(), digit)
}

fn qr_least_squares_nalgebra(x: DMatrix<f64>, y: &DVector<f64>, digit: u8) -> Weights {
    let qr = x.col_piv_qr();

    let (q, rt, p) = qr.unpack();
    let qtb = q.transpose() * y;

    let rank = (0..rt.nrows().min(rt.ncols()))
        .take_while(|&i| rt[(i, i)].abs() > EPSILON)
        .count();

    let qtb_trimmed = qtb.rows(0, rank).into_owned();
    let rt_trimmed = rt.view((0, 0), (rank, rank));

    let solution_trimmed = rt_trimmed.solve_upper_triangular(&qtb_trimmed).unwrap();

    let mut solution = solution_trimmed.resize_vertically(785, 0.0);

    p.inv_permute_rows(&mut solution);
    println!("{rank}");

    Weights::new(solution.as_slice(), digit)
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

#[derive(Serialize, Deserialize, Debug)]
struct Weights {
    digit: u8,
    n_train: Option<u32>,
    epsilon: Option<f64>,
    weights: Vec<f64>,
}

impl Weights {
    fn new(vector: &[f64], digit: u8) -> Weights {
        let n_train = Some(N_TRAINING_SET);
        let epsilon = Some(EPSILON);
        let weights = vector.to_owned();

        Weights {
            digit,
            n_train,
            epsilon,
            weights,
        }
    }
}

fn select_train_or_infer(
    trn_img: &[u8],
    trn_lbl: &[u8],
    tst_img: &[u8],
    tst_lbl: &[u8],
) -> Result<()> {
    let mut digit_to_train;
    loop {
        let items = vec![
            "Train Single Digit",
            "Train All Digits",
            "Inference",
            "Build PCA",
            "Exit",
        ];
        let selection = FuzzySelect::new()
            .with_prompt("Select and option:")
            .items(&items)
            .interact()?;

        match selection {
            0 => {
                digit_to_train = select_digit_to_train()?;
                let method = select_training_method()?;
                train_single_digit(trn_img, trn_lbl, digit_to_train, method)?;
            }
            1 => {
                let library = select_training_library()?;
                let method = select_training_method()?;
                train_all_digits(trn_img, trn_lbl, library, method)?;
            }
            2 => {
                let weights = get_weights()?;
                digit_inference(tst_img, tst_lbl, weights)?
            }
            3 => {
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
}

enum Library {
    NAlgebra,
    Faer,
}

fn train_all_digits(
    trn_img: &[u8],
    trn_lbl: &[u8],
    library: Library,
    method: Method,
) -> Result<()> {
    match library {
        Library::NAlgebra => {
            println!("pca");
            let train_data = DMatrix::from_row_slice(N_TRAINING_SET as usize, 784, trn_img)
                // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
                .map(|pixel| pixel as f64 / 255.0);
            let mut pca = open_pca()?;
            let z = pca_transform(&mut pca, &train_data, PCA_COMPONENTS);
            let z = z.insert_column(0, 1.0);
            println!("end pca");
            match method {
                Method::SVD => {
                    let start = Instant::now();

                    let svd = nalgebra_lapack::SVD::new(z).unwrap();
                    let pseudo_inverse = svd.pseudo_inverse(EPSILON);
                    for i in 0..=9 {
                        let train_label = DVector::from_row_slice(trn_lbl)
                            .map(|digit| if digit == i { 1.0 } else { 0.0 });

                        let weights = svd_least_squares_lapack(&pseudo_inverse, &train_label, i);
                        save_json(weights)?;
                    }
                    println!("Time elapsed: {:?}", start.elapsed());
                }
                Method::QR => {
                    println!("Before QR: {:?}", Instant::now());

                    let start = Instant::now();
                    let qr = nalgebra_lapack::QR::new(z)?;

                    println!("QR elapsed: {:?}", start.elapsed());

                    println!("After QR");

                    for i in 0..=9 {
                        println!("Training {i}");
                        let train_label = DVector::from_row_slice(trn_lbl)
                            .map(|digit| if digit == i { 1.0 } else { 0.0 });
                        let solution = qr.solve(train_label)?;
                        let weights = Weights::new(solution.as_slice(), i);
                        save_json(weights)?;
                    }
                    println!("Time elapsed: {:?}", start.elapsed());
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
        },
    };

    Ok(())
}

// fn train_all_digits_old(trn_img: &[u8], trn_lbl: &[u8], method: Method) -> Result<()> {
//     let train_data = DMatrix::from_row_slice(N_TRAINING_SET as usize, 784, trn_img)
//         // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
//         .map(|pixel| pixel as f64 / 255.0);
//     let mut pca = open_pca()?;
//     let z = pca_transform(&mut pca, &train_data, PCA_COMPONENTS);
//     let z = z.insert_column(0, 1.0);
//     let svd = nalgebra_lapack::SVD::new(z).unwrap();
//     let pseudo_inverse = svd.pseudo_inverse(EPSILON);
//
//     for i in 0..=9 {
//         println!("Training digit: {}", i);
//
//         let start = Instant::now();
//         let weights = match method {
//             Method::Lapack => {
//                 // let (train_data, train_label) = prepare_train_data_nalgebra(trn_img, trn_lbl, i)?;
//                 let train_label = DVector::from_row_slice(trn_lbl)
//                     .map(|digit| if digit == i { 1.0 } else { 0.0 });
//
//                 svd_least_squares_lapack(&pseudo_inverse, &train_label, i)
//             }
//             Method::NAlgebraQR => {
//                 let (train_data, train_label) = prepare_train_data_nalgebra(trn_img, trn_lbl, i)?;
//                 qr_least_squares_nalgebra(train_data, &train_label, i)
//             }
//             Method::FaerSVD => {
//                 let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
//                 // let z = pca(train_data.clone());
//                 svd_least_squares_faer(train_data, train_label, i)
//             }
//             Method::FaerQR => {
//                 let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
//                 qr_least_squares_faer(train_data, train_label, i)
//             }
//         };
//         println!("Time elapsed: {:?}", start.elapsed());
//
//         save_json(weights)?;
//     }
//
//     Ok(())
// }

fn train_single_digit(trn_img: &[u8], trn_lbl: &[u8], digit: u8, method: Method) -> Result<()> {
    println!("Training digit: {}", digit);

    let start = Instant::now();
    todo!();

    // let weights = match method {
    //     Method::Lapack => {
    //         let (train_data, train_label) = prepare_train_data_nalgebra(trn_img, trn_lbl, digit)?;
    //         svd_least_squares_lapack(&train_data, &train_label, digit)
    //     }
    //     Method::NAlgebraQR => {
    //         let (train_data, train_label) = prepare_train_data_nalgebra(trn_img, trn_lbl, digit)?;
    //         qr_least_squares_nalgebra(train_data, &train_label, digit)
    //     }
    //     Method::FaerSVD => {
    //         let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, digit)?;
    //         svd_least_squares_faer(train_data, train_label, digit)
    //     }
    //     Method::FaerQR => {
    //         let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, digit)?;
    //         qr_least_squares_faer(train_data, train_label, digit)
    //     }
    // };
    // println!("Time elapsed: {:?}", start.elapsed());
    // save_json(weights)?;
    //
    // Ok(())
}

fn select_digit_to_train() -> Result<u8> {
    let digits: Vec<u8> = (0..=9).collect();

    let selection = FuzzySelect::new()
        .with_prompt("Select a digit to train:")
        .items(&digits)
        .interact()?;

    let digit: u8 = selection as u8;

    Ok(digit)
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
    let items = vec!["SVD", "QR"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Method::SVD),
        1 => Ok(Method::QR),
        _ => todo!(),
    }
}

fn prepare_train_data_nalgebra(
    trn_img: &[u8],
    trn_lbl: &[u8],
    digit_to_train: u8,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let train_data = DMatrix::from_row_slice(N_TRAINING_SET as usize, 784, trn_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0);

    // // Add bias term in the form of a column of 1's
    // let train_data = train_data.insert_column(0, 1.0);

    let train_label = DVector::from_row_slice(trn_lbl)
        .map(|digit| if digit == digit_to_train { 1.0 } else { 0.0 });

    Ok((train_data, train_label))
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

// Saves weights to a new folder with the name structure of "weights_N TRAINING SIZE_EPSILON"
fn save_json(weights: Weights) -> Result<()> {
    let path = std::path::Path::new("./weights");
    std::fs::create_dir_all(path)?;

    let weights_folder = format!("weights_{}_{}", N_TRAINING_SET, EPSILON);
    let weights_folder = path.join(weights_folder);
    std::fs::create_dir_all(&weights_folder)?;

    let filename = format!("{}/{} weights.json", weights_folder.display(), {
        weights.digit
    });
    let file = File::create(filename).context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, &weights)
        .context("Failed to serialize weights into JSON format")?;

    Ok(())
}

fn open_json(digit: u8, path: &std::path::Path) -> Result<Weights> {
    let filename = format!("{}/{} weights.json", path.display(), digit);
    let file = File::open(filename)?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

fn open_pca() -> Result<Pca> {
    let filename = "pca/pca.json";
    let file = File::open(filename)?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

fn get_weights() -> Result<Vec<Weights>> {
    let mut weights: Vec<Weights> = vec![];
    let folder = FileDialog::new()
        .set_title("Select a folder to open in terminal")
        .pick_folder();

    match folder {
        Some(path) => {
            for i in 0..=9 {
                let weight = open_json(i, path.as_path())?;
                weights.push(weight);
            }
        }
        None => {
            eprintln!("No folder selected.");
        }
    }

    Ok(weights)
}

fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], weights: Vec<Weights>) -> Result<()> {
    let test_data = DMatrix::from_row_slice(N_TESTING_SET as usize, 784, tst_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0);

    let n_train = weights.first().unwrap().n_train.unwrap();
    let epsilon = weights.first().unwrap().epsilon.unwrap();
    let mut pca = open_pca()?;

    let z = pca_transform(&mut pca, &test_data, PCA_COMPONENTS);
    let z = z.insert_column(0, 1.0);

    let mut results: Vec<(u8, u8)> = vec![];
    let mut metrics: Vec<F1> = (0..10)
        .map(|digit| F1::new(digit, n_train, epsilon))
        .collect();

    for (i, row) in z.row_iter().enumerate() {
        let (mut digit, mut max_score) = (0, f64::NEG_INFINITY);
        for digit_weights in &weights {
            let weights = DVector::from_row_slice(&digit_weights.weights);

            let score = row.transpose().dot(&weights);

            if score > max_score {
                max_score = score;
                digit = digit_weights.digit;
            }
        }

        for metric in &mut metrics {
            if digit == metric.digit && tst_lbl[i] == metric.digit {
                metric.tpos += 1.0;
            } else if digit == metric.digit && tst_lbl[i] != metric.digit {
                metric.fpos += 1.0;
            } else if digit != metric.digit && tst_lbl[i] == metric.digit {
                metric.fneg += 1.0;
            }
        }

        results.push((tst_lbl[i], digit));
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

    let num_correct = results.iter().fold(
        0,
        |acc: u32, digits| {
            if digits.0 == digits.1 { acc + 1 } else { acc }
        },
    );

    f1_scatterplot(metrics)?;

    let total_scores = N_TESTING_SET;

    let percent_correct = (num_correct as f32 / N_TESTING_SET as f32) * 100.0;

    println!(
        "Number of Tests: {}\n Number correct: {}\n Percent Correct: {:.2}%\n Average F1: {}",
        total_scores, num_correct, percent_correct, average_f1
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
        let result = svd_least_squares(&x, &y, digit, epsilon);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = epsilon);
    }

    #[test]
    fn test_svd_least_squares_lapack() {
        let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
        let digit = 0;
        let epsilon = 1e-12;
        let result = svd_least_squares_lapack(&x, &y, digit);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = epsilon);
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
