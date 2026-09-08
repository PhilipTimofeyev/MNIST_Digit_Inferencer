use super::super::{prepare_trn_img_nalgebra, save_weights};
use crate::{ALPHA, EPOCHS, dmatrix_to_vec2d, mat_to_vec2d, prepare_trn_img_faer};
use crate::{Library, Model, ModelType};
use anyhow::Result;
use faer::Mat;
use nalgebra::DMatrix;

pub fn train(trn_img: &[u8], trn_lbl: &[u8], library: Library, use_pca: bool) -> Result<()> {
    match library {
        Library::NAlgebra => {
            let train_data = prepare_trn_img_nalgebra(trn_img);

            let y = n_algebra::one_hot_encode(trn_lbl);
            let train_data = train_data.insert_column(0, 1.0);
            let weights = n_algebra::logistic_regression(&train_data, &y);
            let weights = dmatrix_to_vec2d(&weights);

            let model_type = ModelType::LogisticRegression {
                weights,
                learning_rate: ALPHA,
                epochs: EPOCHS,
            };

            let model = Model::new(785, None, model_type);
            save_weights(model)?;
        }
        Library::Faer => {
            let train_data = prepare_trn_img_faer(trn_img);
            let y = faer_lib::one_hot_encode(trn_lbl);
            let bias_col = Mat::from_fn(train_data.nrows(), 1, |_, _| 1.0);
            let train_data = faer::concat![[bias_col, train_data]];
            let weights = faer_lib::logistic_regression(&train_data, &y);
            let model_type = ModelType::LogisticRegression {
                weights,
                learning_rate: ALPHA,
                epochs: EPOCHS,
            };

            let model = Model::new(785, None, model_type);
            save_weights(model)?;
        }
    };

    Ok(())
}

pub mod n_algebra {
    use super::*;
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
    pub fn logistic_regression(x: &DMatrix<f64>, y: &DMatrix<f64>) -> DMatrix<f64> {
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

        weights
    }
}

pub mod faer_lib {
    use super::*;
    pub fn logistic_regression(x: &Mat<f64>, y: &Mat<f64>) -> Vec<Vec<f64>> {
        let mut weights = Mat::<f64>::zeros(x.ncols(), 10);

        for epoch in 1..=EPOCHS {
            let logits = x * &weights;
            let p = softmax(&logits);
            let loss = cross_entropy_loss(&p, y);
            println!("epoch: {epoch} | loss: {loss}");
            let error = &p - y;
            let gradient = x.transpose() * error / x.nrows() as f64;
            weights -= gradient * ALPHA;
        }

        mat_to_vec2d(&weights)
    }
    pub fn softmax(logits: &Mat<f64>) -> Mat<f64> {
        let mut p = Mat::<f64>::zeros(logits.nrows(), logits.ncols());

        for i in 0..logits.nrows() {
            let row = logits.row(i);

            let max_val = row.max().unwrap();
            let stable_row = row.map(|val| (val - max_val).exp());

            let sum: f64 = stable_row.sum();

            // 3. Divide by the sum to get probabilities

            p.row_mut(i).copy_from(&(&stable_row / sum));
        }

        p
    }

    pub fn one_hot_encode(trn_labels: &[u8]) -> Mat<f64> {
        let num_samples = trn_labels.len();
        let mut one_hot = Mat::<f64>::zeros(num_samples, 10);

        for (lbl_idx, &label) in trn_labels.iter().enumerate() {
            one_hot[(lbl_idx, label as usize)] = 1.0;
        }

        one_hot
    }

    fn cross_entropy_loss(p: &Mat<f64>, y: &Mat<f64>) -> f64 {
        let mut loss = 0.0;

        for i in 0..p.nrows() {
            for j in 0..p.ncols() {
                let y_val = y[(i, j)];
                if y_val > 0.0 {
                    loss -= y_val * p[(i, j)].ln();
                }
            }
        }

        loss / p.nrows() as f64
    }
}
