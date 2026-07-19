use anyhow::{Context, Result};
use dialoguer::FuzzySelect;
use mnist::*;
use nalgebra::{DMatrix, DVector, SVD};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const EPSILON: f64 = 5.0;
const N_TRAINING_SET: u32 = 10000;
const N_TESTING_SET: u32 = 10000;

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
        .validation_set_length(10)
        .test_set_length(N_TESTING_SET)
        .finalize();

    select_train_or_infer(&trn_img, &trn_lbl, &tst_img, &tst_lbl)?;

    Ok(())
}

fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>, digit: u8, epsilon: f64) -> Weights {
    let svd = SVD::new(x.clone(), true, true);
    let weights = svd.solve(y, epsilon).unwrap();

    Weights::new(&weights, digit)
}

fn svd_least_squares_lapack(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    digit: u8,
    epsilon: f64,
) -> Weights {
    let svd = nalgebra_lapack::SVD::new(x.clone()).unwrap();

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
        if *singular_i > epsilon {
            *trimmed_i /= *singular_i;
        } else {
            *trimmed_i = 0.0;
        }
    }

    // Finally multiply by V to get the least squares solution
    let solution = svd.vt.transpose() * trimmed_ut_y;

    Weights::new(&solution, digit)
}

#[derive(Debug)]
struct F1 {
    digit: u8,
    tpos: f32,
    fpos: f32,
    fneg: f32,
}

impl F1 {
    fn new(digit: u8) -> F1 {
        F1 {
            digit,
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
    rows: usize,
    cols: usize,
    weights: Vec<f64>,
}

impl Weights {
    fn new(vector: &DVector<f64>, digit: u8) -> Weights {
        let rows = vector.nrows();
        let cols = vector.ncols();
        let weights = vector.data.as_vec().to_owned();

        Weights {
            digit,
            rows,
            cols,
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
            "Exit",
        ];
        let selection = FuzzySelect::new()
            .with_prompt("Select and option:")
            .items(&items)
            .interact()?;

        match selection {
            0 => {
                digit_to_train = select_digit_to_train()?;
                let (train_data, train_label) =
                    prepare_train_data(trn_img, trn_lbl, digit_to_train)?;
                println!("Training digit: {}", digit_to_train);
                let weights =
                    svd_least_squares_lapack(&train_data, &train_label, digit_to_train, EPSILON);
                save_json(weights)?
            }
            1 => train_all_digits(trn_img, trn_lbl)?,

            2 => {
                let weights = get_weights()?;
                digit_inference(tst_img, tst_lbl, weights)?
            }
            _ => break,
        }
    }

    Ok(())
}

fn train_all_digits(trn_img: &[u8], trn_lbl: &[u8]) -> Result<()> {
    for i in 0..=9 {
        let (train_data, train_label) = prepare_train_data(trn_img, trn_lbl, i)?;
        let weights = svd_least_squares_lapack(&train_data, &train_label, i, EPSILON);
        save_json(weights)?
    }

    Ok(())
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

fn prepare_train_data(
    trn_img: &[u8],
    trn_lbl: &[u8],
    digit_to_train: u8,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let train_data = DMatrix::from_row_slice(N_TRAINING_SET as usize, 784, trn_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
    // .map(|pixel| pixel as f64 / 255.0);

    // Add bias term in the form of a column of 1's
    let train_data = train_data.insert_column(0, 1.0);

    let train_label = DVector::from_row_slice(trn_lbl)
        .map(|digit| if digit == digit_to_train { 1.0 } else { 0.0 });

    Ok((train_data, train_label))
}

fn save_json(weights: Weights) -> Result<()> {
    let filename = format!("weights/{} weights.json", { weights.digit });
    let file = File::create(filename).context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &weights)
        .context("Failed to serialize weights into JSON format")?;
    Ok(())
}

fn open_json(digit: u8) -> Result<Weights> {
    let filename = format!("weights/{} weights.json", digit);
    let file = File::open(filename)?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

fn get_weights() -> Result<Vec<Weights>> {
    let mut weights: Vec<Weights> = vec![];
    for i in 0..=9 {
        let weight = open_json(i)?;
        weights.push(weight);
    }

    Ok(weights)
}

fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], weights: Vec<Weights>) -> Result<()> {
    let test_data = DMatrix::from_row_slice(N_TESTING_SET as usize, 784, tst_img)
        .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
    // .map(|pixel| pixel as f64 / 255.0);

    let test_data = test_data.insert_column(0, 1.0);

    let mut results: Vec<(u8, u8)> = vec![];
    let mut metrics: Vec<F1> = (0..10).map(F1::new).collect();

    for (i, row) in test_data.row_iter().enumerate() {
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
    let root = BitMapBackend::new("F1_scatterplot.png", (1920, 1080)).into_drawing_area();
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
        let result = svd_least_squares_lapack(&x, &y, digit, epsilon);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = epsilon);
    }

    #[test]
    fn test_f1() {
        let f1_example = F1 {
            digit: 0,
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
