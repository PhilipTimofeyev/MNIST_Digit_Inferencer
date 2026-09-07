use crate::solvers::{qr, svd};
mod models;
use crate::models::logistic_regression;
mod preprocessing;
use crate::preprocessing::pca;
mod solvers;
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
const PCA_COMPONENTS: usize = 50;
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
    pca: Option<usize>,
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
    fn new(n_features: usize, pca: Option<usize>, model: ModelType) -> Model {
        Model {
            n_train: N_TRAINING_SET,
            n_features,
            pca,
            model,
        }
    }
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
        ModelType::LinearRegression {
            weights: _,
            epsilon,
        } => {
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
                let model = get_weights()?;
                digit_inference(tst_img, tst_lbl, model)?;
            }
            2 => {
                let pca = pca::fit_pca(trn_img);
                preprocessing::pca::save_pca(pca)?;
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

fn svd_train_digits(pseudo_inverse: DMatrix<f64>, trn_lbl: &[u8], pca: bool) -> Result<()> {
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

    let pca = if pca { Some(PCA_COMPONENTS) } else { None };

    let model = Model::new(pseudo_inverse.nrows(), pca, model_type);
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
                        svd::svd_nalgebra_lapack_pca(train_data)?
                    } else {
                        svd::svd_nalgebra_lapack(train_data)?
                    };

                    svd_train_digits(pseudo_inverse, trn_lbl, use_pca)?;
                    println!("Time elapsed: {:?}", start.elapsed());
                }
                Method::QR => {
                    let start = Instant::now();

                    if use_pca {
                        let qr = qr::qr_nalgebra_lapack_pca(train_data)?;
                        let mut all_weights = DMatrix::zeros(qr.ncols(), 10);
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

                        let model = Model::new(qr.nrows(), Some(PCA_COMPONENTS), model_type);
                        save_weights(model)?;
                    } else {
                        let train_data = train_data.insert_column(0, 1.0);
                        let n_features = train_data.ncols();
                        let (q, rt, p) = qr::qr_nalgebra_lapack(train_data);
                        let mut all_weights = DMatrix::zeros(n_features, 10);
                        for digit in 0..=9 {
                            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                            let qtb = &q * train_label;
                            let weights = rt.solve_upper_triangular(&qtb).unwrap();
                            let mut weights = weights.resize_vertically(n_features, 0.0);
                            p.inv_permute_rows(&mut weights);
                            all_weights.set_column(digit as usize, &weights);
                        }
                        let model_type = ModelType::LinearRegression {
                            weights: all_weights,
                            epsilon: EPSILON,
                        };

                        let model = Model::new(n_features, None, model_type);
                        save_weights(model)?;
                    }

                    println!("QR elapsed: {:?}", start.elapsed());
                }
                Method::Logistic => {
                    let y = logistic_regression::one_hot_encode(trn_lbl);
                    let train_data = train_data.insert_column(0, 1.0);
                    let weights = logistic_regression::logistic_regression(&train_data, &y);
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

fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], model: Model) -> Result<()> {
    let weights = match model.model {
        ModelType::LogisticRegression {
            weights,
            learning_rate: _,
            epochs: _,
        } => weights,
        ModelType::LinearRegression {
            weights,
            epsilon: _,
        } => weights,
    };

    let mut test_data = DMatrix::from_row_slice(N_TESTING_SET as usize, 784, tst_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0);

    if let Some(n_components) = model.pca {
        let mut pca = pca::open_pca()?;
        test_data = pca::pca_transform(&mut pca, &test_data, n_components);
    }

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
