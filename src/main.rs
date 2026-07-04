use anyhow::{Context, Result};
use dialoguer::FuzzySelect;
use mnist::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_lapack::SVD;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const EPSILON: f64 = 1.0;
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

// fn svd_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) -> Weights {
//     let svd = x.clone().svd(true, true);
//     let weights = svd.solve(y, EPSILON).unwrap();
//
//     Weights::new(&weights)
// }

fn svd_least_squares_lapack(x: &DMatrix<f64>, y: &DVector<f64>, digit: u8) -> Weights {
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

    Weights::new(&solution, digit)
}

#[derive(Debug)]
struct F1 {
    digit: u8,
    tpos: f32,
    tneg: f32,
    fpos: f32,
    fneg: f32,
}

impl F1 {
    fn new(digit: u8) -> F1 {
        F1 {
            digit,
            tpos: 0.0,
            tneg: 0.0,
            fpos: 0.0,
            fneg: 0.0,
        }
    }

    fn precision(&self) -> f32 {
        self.tpos / (self.tpos + self.fpos)
    }

    fn recall(&self) -> f32 {
        self.tpos / (self.tpos + self.fneg)
    }

    fn f1(&self) -> f32 {
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
    let mut digit_to_train = 0;
    loop {
        let items = vec!["Train", "Train All", "Inference", "Inference All", "Exit"];
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
                let weights = svd_least_squares_lapack(&train_data, &train_label, digit_to_train);
                save_json(weights)?
            }
            1 => train_all_digits(trn_img, trn_lbl)?,
            2 => {
                println!("{}", digit_to_train);
                //FIX THIS WITH ARGUMENT NONE
                let weights = open_json(0)?;

                digit_inference(tst_img, tst_lbl, weights, digit_to_train)?;
            }
            3 => {
                let weights = get_weights()?;
                digit_inference_all(tst_img, tst_lbl, weights)?
            }
            _ => break,
        }
    }

    Ok(())
}

fn train_all_digits(trn_img: &[u8], trn_lbl: &[u8]) -> Result<()> {
    for i in 0..=9 {
        let (train_data, train_label) = prepare_train_data(trn_img, trn_lbl, i)?;
        let weights = svd_least_squares_lapack(&train_data, &train_label, i);
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

fn digit_inference(
    tst_img: &[u8],
    tst_lbl: &[u8],
    weights: Weights,
    trained_digit: u8,
) -> Result<()> {
    let test_data = DMatrix::from_row_slice(N_TESTING_SET as usize, 784, tst_img)
        // .map(|pixel| if pixel as f64 > 0.0 { 1.0 } else { 0.0 });
        .map(|pixel| pixel as f64 / 255.0);

    // let test_data = test_data.insert_column(0, 1.0);

    let weights = DVector::from_row_slice(&weights.weights);

    let scores = &test_data * &weights;
    let x: Vec<f64> = scores.iter().cloned().collect();

    let y: Vec<f64> = tst_lbl
        .iter()
        .map(|digit| if *digit == trained_digit { 1.0 } else { 0.0 })
        .collect();

    let total_scores = N_TESTING_SET;

    let mut num_correct = 0;
    for i in 0..N_TESTING_SET as usize {
        if scores[i] >= 0.5 && y[i] == 1.0 || scores[i] < 0.5 && y[i] == 0.0 {
            num_correct += 1
        } else {
            // println!("digit: {}, \nscore: {}", y[i], scores[i]);
        }
    }

    let percent_correct = (num_correct as f32 / N_TESTING_SET as f32) * 100.0;

    println!(
        "Total scores: {}\n Number correct: {}\n Percent Correct: {:.2}%",
        total_scores, num_correct, percent_correct
    );
    //
    score_scatterplot(x, y)?;
    // for i in 0..50 {
    //     println!("digit={} score={}", tst_lbl[i], scores[i]);
    // }

    Ok(())
}

fn digit_inference_all(tst_img: &[u8], tst_lbl: &[u8], weights: Vec<Weights>) -> Result<()> {
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
            // println!(
            //     "score: {:?} digit: {} actual: {}",
            //     score, digit_weights.digit, tst_lbl[i],
            // );
            // if digit_1_f1.digit == digit_weights.digit {
            //     if score >= 0.5 && tst_lbl[i] == digit_1_f1.digit {
            //         digit_1_f1.tpos += 1.0;
            //         // println!("tpos");
            //     } else if score >= 0.5 && tst_lbl[i] != digit_1_f1.digit {
            //         digit_1_f1.fpos += 1.0;
            //         // println!("fpos");
            //     } else if score < 0.5 && tst_lbl[i] == digit_1_f1.digit {
            //         // println!("fneg");
            //         digit_1_f1.fneg += 1.0;
            //     } else if score < 0.5 && tst_lbl[i] != digit_1_f1.digit {
            //         digit_1_f1.tneg += 1.0;
            //         // println!("tneg");
            //     };
            // }
        }

        for metric in &mut metrics {
            if digit == metric.digit && tst_lbl[i] == metric.digit {
                metric.tpos += 1.0;
            } else if digit == metric.digit && tst_lbl[i] != metric.digit {
                metric.fpos += 1.0;
            } else if digit != metric.digit && tst_lbl[i] == metric.digit {
                metric.fneg += 1.0;
            } else {
                metric.tneg += 1.0;
            }
        }

        // if digit == digit_1_f1.digit && tst_lbl[i] == digit_1_f1.digit {
        //     digit_1_f1.tpos += 1.0;
        //     // println!("tpos");
        // } else if digit == digit_1_f1.digit && tst_lbl[i] != digit_1_f1.digit {
        //     digit_1_f1.fpos += 1.0;
        //     // println!("fpos");
        // } else if digit != digit_1_f1.digit && tst_lbl[i] == digit_1_f1.digit {
        //     // println!("fneg");
        //     digit_1_f1.fneg += 1.0;
        // } else {
        //     digit_1_f1.tneg += 1.0;
        //     // println!("tneg");
        // };
        results.push((tst_lbl[i], digit));

        // println!("{:?}", results);
        // println!("Digit {}, max score: {:?}", digit, max_score);
        // println!("Actual Digit: {}", tst_lbl[i]);
    }

    for digit in metrics {
        println!(
            "Digit: {}\nPrecision: {}\nRecall: {}\nF1: {}\n",
            digit.digit,
            digit.precision(),
            digit.recall(),
            digit.f1()
        );
    }

    let num_correct = results.iter().fold(
        0,
        |acc: u32, digits| {
            if digits.0 == digits.1 { acc + 1 } else { acc }
        },
    );

    // let weights = DVector::from_row_slice(&weights.weights);
    //
    // let scores = &test_data * &weights;
    // let x: Vec<f64> = scores.iter().cloned().collect();
    //
    // let y: Vec<f64> = tst_lbl
    //     .iter()
    //     .map(|digit| if *digit == trained_digit { 1.0 } else { 0.0 })
    //     .collect();
    //
    let total_scores = N_TESTING_SET;
    //
    // let mut num_correct = 0;
    // for i in 0..N_TESTING_SET as usize {
    //     if scores[i] >= 0.5 && y[i] == 1.0 || scores[i] < 0.5 && y[i] == 0.0 {
    //         num_correct += 1
    //     } else {
    //         // println!("digit: {}, \nscore: {}", y[i], scores[i]);
    //     }
    // }
    //
    let percent_correct = (num_correct as f32 / N_TESTING_SET as f32) * 100.0;
    //
    println!(
        "Total scores: {}\n Number correct: {}\n Percent Correct: {:.2}%",
        total_scores, num_correct, percent_correct
    );
    // //
    // score_scatterplot(x, y)?;
    // for i in 0..50 {
    //     println!("digit={} score={}", tst_lbl[i], scores[i]);
    // }

    Ok(())
}

// Creates a scatterplot where scores of whether the image is a specific digit are the x-axis, and
// whether is was the digit (1) or a different digit (0), on the y-axis
fn score_scatterplot(x_values: Vec<f64>, y_values: Vec<f64>) -> Result<()> {
    let root = BitMapBackend::new("digit_scatterplot.png", (1280, 960)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Digit Scatterplot", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(45)
        .y_label_area_size(45)
        .build_cartesian_2d(-1f64..2f64, -1f64..2f64)?;

    chart
        .configure_mesh()
        .x_desc("Score")
        .y_desc("Digit (1) or Other (0)")
        // .x_label_offset(40)
        // .y_label_offset(40)
        .axis_desc_style(("sans-serif", 20, &BLACK))
        .draw()?;

    chart.draw_series(
        x_values
            .iter()
            .zip(y_values.iter())
            .map(|(&x, &y)| Circle::new((x, y), 1, BLUE.filled())),
    )?;

    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // #[test]
    // fn test_svd_least_squares() {
    //     let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
    //     let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
    //     let result = svd_least_squares(&x, &y);
    //
    //     assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = 0.0001);
    // }

    #[test]
    fn test_svd_least_squares_lapack() {
        let x = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let y = DVector::<f64>::from_vec(vec![2.0, 3.0, 7.0]);
        let result = svd_least_squares_lapack(&x, &y, 0);

        assert_relative_eq!(result.weights[..], &vec![2.5, -1.0], epsilon = 0.0001);
    }
}
