use mnist::*;
use nalgebra::*;

fn main() {
    let n_training_set = 5_000;
    let Mnist {
        trn_img,
        trn_lbl,
        // tst_img,
        // tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data_sets/mnist/")
        .label_format_digit()
        .training_set_length(n_training_set)
        .validation_set_length(10)
        .test_set_length(10_000)
        .finalize();

    // For linear regression, only need a binary pixel value of on or off, which is why the pixel value is
    // converted to 0 (off) or 1 (on)
    let train_data = DMatrix::from_row_slice(n_training_set as usize, 784, &trn_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });

    let train_label =
        DVector::from_row_slice(&trn_lbl).map(|digit| if digit == 3 { 1.0 } else { 0.0 });

    svd_least_squares(&train_data, &train_label);
}

fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) {
    let svd = x.clone().svd(true, true);
    let weights = svd.solve(y, 1e-12).unwrap();

    println!("{:?}", weights);
}
