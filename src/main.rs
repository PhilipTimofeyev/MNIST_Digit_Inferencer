use anyhow::{Context, Result};
use dialoguer::FuzzySelect;
use mnist::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_lapack::SVD;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const EPSILON: f64 = 1e-12;

fn main() -> Result<()> {
    let n_training_set = 10_000;
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data_sets/mnist/")
        .label_format_digit()
        .training_set_length(n_training_set)
        .validation_set_length(10)
        .test_set_length(100)
        .finalize();

    // For linear regression, only need a binary pixel value of on or off, which is why the pixel value is
    // converted to 0 (off) or 1 (on)
    let train_data = DMatrix::from_row_slice(n_training_set as usize, 784, &trn_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });

    let train_label =
        DVector::from_row_slice(&trn_lbl).map(|digit| if digit == 5 { 1.0 } else { 0.0 });

    select_train_or_infer(&train_data, &train_label, &tst_img, &tst_lbl)?;

    Ok(())
}

fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) -> Weights {
    let svd = x.clone().svd(true, true);
    let weights = svd.solve(y, EPSILON).unwrap();

    Weights::new(&weights)
}

fn svd_least_squares_lapack(x: &DMatrix<f64>, y: &DVector<f64>) -> Weights {
    let svd = SVD::new(x.clone()).unwrap();

    // Equation to solve is w = V * sigma^-1 * U^T * y

    let ut_y = svd.u.transpose() * y;

    // Since the sigma matrix is comprised of singular values of A^T*A, it will have a multiplicity
    // of the number of rows (784 in the case of the MNIST data set). Beyond those rows the values
    // will be zero, so to save time on computation, the ut_y matrix can be trimmed down to 784 x 1,
    // since those values would be multiplied by zero anyway.
    let mut trimmed_ut_y = ut_y.rows(0, svd.singular_values.len()).into_owned();

    // This filters out values that would cause a division by zero, and divides (U^T*y) by the
    // singular values
    for (trimmed_i, singular_i) in trimmed_ut_y.iter_mut().zip(svd.singular_values.iter()) {
        if *singular_i > EPSILON {
            *trimmed_i /= *singular_i;
        } else {
            *trimmed_i = 0.0;
        }
    }

    // Finally multiply by V to get the least squares solution
    let solution = svd.vt.transpose() * trimmed_ut_y;

    Weights::new(&solution)
}

#[derive(Serialize, Deserialize, Debug)]
struct Weights {
    rows: usize,
    cols: usize,
    weights: Vec<f64>,
}

impl Weights {
    fn new(vector: &DVector<f64>) -> Weights {
        let rows = vector.nrows();
        let cols = vector.ncols();
        let weights = vector.data.as_vec().to_owned();

        Weights {
            rows,
            cols,
            weights,
        }
    }
}

fn select_train_or_infer(
    train_data: &DMatrix<f64>,
    train_label: &DVector<f64>,
    tst_img: &[u8],
    tst_lbl: &[u8],
) -> Result<()> {
    let items = vec!["Train", "Inference"];

    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()
        .unwrap();

    match selection {
        0 => {
            let weights = svd_least_squares_lapack(train_data, train_label);
            save_json(weights)?
        }
        1 => {
            let weights = open_json()?;

            digit_inference(tst_img, tst_lbl, weights);
        }
        _ => println!("Quit"),
    }

    Ok(())
}

fn save_json(weights: Weights) -> Result<()> {
    let file = File::create("weights.json").context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &weights)
        .context("Failed to serialize weights into JSON format")?;
    Ok(())
}

fn open_json() -> Result<Weights> {
    let file = File::open("weights.json")?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], weights: Weights) {
    let test_data = DMatrix::from_row_slice(100, 784, &tst_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });

    let weights = DVector::from_row_slice(&weights.weights);

    let scores = &test_data * &weights;

    for i in 0..100 {
        println!("digit={} score={}", tst_lbl[i], scores[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_svd_least_squares() {
        let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
        let result = svd_least_squares(&x, &y);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = 0.0001);
    }

    #[test]
    fn test_svd_least_squares_lapack() {
        let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
        let result = svd_least_squares_lapack(&x, &y);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = 0.0001);
    }
}
