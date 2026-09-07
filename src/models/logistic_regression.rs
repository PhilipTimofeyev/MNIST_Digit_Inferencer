use super::super::{prepare_trn_img_nalgebra, save_weights};
use crate::{ALPHA, EPOCHS};
use crate::{Library, Model, ModelType};
use anyhow::Result;
use nalgebra::DMatrix;

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
pub fn logistic_regression(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Model {
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

    Model::new(x.ncols(), None, model_type)
}

// Converts the digits into one-hot encoding
// the digit is represented as a row of binary numbers, where 1 and its index in the row indicates
// the digit, ie, 0 0 1 0 0 0 0 0 0 0 is the digit 2
pub fn one_hot_encode(trn_labels: &[u8]) -> DMatrix<f64> {
    let num_samples = trn_labels.len();
    let mut one_hot = DMatrix::zeros(num_samples, 10);

    for (lbl_idx, &label) in trn_labels.iter().enumerate() {
        one_hot[(lbl_idx, label as usize)] = 1.0;
    }

    one_hot
}

pub fn train(trn_img: &[u8], trn_lbl: &[u8], library: Library, use_pca: bool) -> Result<()> {
    match library {
        Library::NAlgebra => {
            let train_data = prepare_trn_img_nalgebra(trn_img);

            let y = one_hot_encode(trn_lbl);
            let train_data = train_data.insert_column(0, 1.0);
            let weights = logistic_regression(&train_data, &y);
            save_weights(weights)?;
        }
        Library::Faer => {}
    };

    Ok(())
}
