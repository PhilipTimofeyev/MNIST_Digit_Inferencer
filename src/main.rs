use anyhow::{Context, Result};
use mnist::*;
use nalgebra::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() -> Result<()> {
    let n_training_set = 5_000;
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
        .test_set_length(50)
        .finalize();

    // For linear regression, only need a binary pixel value of on or off, which is why the pixel value is
    // converted to 0 (off) or 1 (on)
    let train_data = DMatrix::from_row_slice(n_training_set as usize, 784, &trn_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });

    let train_label =
        DVector::from_row_slice(&trn_lbl).map(|digit| if digit == 3 { 1.0 } else { 0.0 });

    // let weights = svd_least_squares(&train_data, &train_label);
    // save_json(weights)?;

    let weights = open_json()?;

    digit_inference(tst_img, tst_lbl, weights);

    Ok(())
}

fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) -> Weights {
    let svd = x.clone().svd(true, true);
    let weights = svd.solve(y, 1e-12).unwrap();

    Weights::new(&weights)
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

fn digit_inference(tst_img: Vec<u8>, tst_lbl: Vec<u8>, weights: Weights) {
    let test_data = DMatrix::from_row_slice(50, 784, &tst_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });

    let weights = DVector::from_row_slice(&weights.weights);

    let scores = &test_data * &weights;

    for i in 0..50 {
        println!("digit={} score={}", tst_lbl[i], scores[i]);
    }
}
