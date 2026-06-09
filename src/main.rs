use mnist::*;
use nalgebra::*;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        // tst_img,
        // tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data_sets/mnist/")
        .label_format_digit()
        .training_set_length(5_000)
        .validation_set_length(10)
        .test_set_length(10_000)
        .finalize();

    // For linear regression, only need a binary pixel value of on or off, which is why the pixel value is
    // converted to 0 (off) or 1 (on)
    let train_data: Vec<f64> = trn_img
        .iter()
        .map(|pixel| if *pixel as f64 > 0.0 { 1.0 } else { 0.0 })
        .collect();

    // Convert vector to DMatrix with n amount of rows and 794 columns
    let train_data = DMatrix::from_row_slice(5_000, 784, &train_data);

    // Train for a specific digit using a 1 if digit matches, otherwise 0
    let train_label: Vec<f64> = trn_lbl
        .iter()
        .map(|digit| if *digit == 0 { 1.0 } else { 0.0 })
        .collect();

    // Convert vector to DVector
    let train_label = DVector::from_row_slice(&train_label);

    svd_least_squares(&train_data, &train_label);
}

fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) {
    let svd = x.clone().svd(true, true);
    let weights = svd.solve(y, 1e-12).unwrap();

    println!("{:?}", weights);
}
