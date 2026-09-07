mod cli;
mod inference;
mod models;
mod preprocessing;
mod solvers;
use anyhow::{Context, Result};
use faer::{Col, Mat, MatRef};
use mnist::*;
use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const N_TRAINING_SET: u32 = 1000;
const N_TESTING_SET: u32 = 10000;
const PCA_COMPONENTS: usize = 50;

// Linear Regression
const EPSILON: f64 = 1e-8;

// Logistic Regression
const EPOCHS: usize = 50;
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

    cli::select_train_or_infer(&trn_img, &trn_lbl, &tst_img, &tst_lbl)?;

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

enum Solver {
    SVD,
    QR,
}

enum Library {
    NAlgebra,
    Faer,
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

        let pseudoinverse = solvers::svd::svd_nalgebra_lapack(x).unwrap();
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
