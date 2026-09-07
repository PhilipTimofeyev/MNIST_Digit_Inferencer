use crate::f1_scatterplot;
use crate::preprocessing::pca;
use crate::{EPSILON, F1, Model, ModelType, N_TESTING_SET, N_TRAINING_SET};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};

pub fn inference(x: &DMatrix<f64>, w: &DMatrix<f64>) -> DVector<usize> {
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

pub fn digit_inference(tst_img: &[u8], tst_lbl: &[u8], model: Model) -> Result<()> {
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
