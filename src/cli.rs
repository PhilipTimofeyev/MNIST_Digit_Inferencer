use crate::{Library, Solver, get_weights, inference, models, preprocessing};
use anyhow::Result;
use dialoguer::{Confirm, FuzzySelect};

pub fn use_pca() -> dialoguer::Result<bool> {
    let pca: bool = Confirm::new()
        .with_prompt("Use Principle Component Analysis?")
        .interact()?;

    Ok(pca)
}

pub fn select_training_library() -> Result<Library> {
    let items = vec!["Faer", "nAlgebra"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Library::Faer),
        1 => Ok(Library::NAlgebra),
        _ => todo!(),
    }
}

pub fn select_solver() -> Result<Solver> {
    let items = vec!["SVD", "QR"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Solver::SVD),
        1 => Ok(Solver::QR),
        _ => Err(anyhow::anyhow!("No selection made")),
    }
}

pub enum ModelSelection {
    LinearRegression,
    LogisticRegression,
}

pub fn select_model() -> Result<ModelSelection> {
    let items = vec!["Linear Regression", "Logistic Regression"];
    let selection = FuzzySelect::new()
        .with_prompt("Select a model:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(ModelSelection::LinearRegression),
        1 => Ok(ModelSelection::LogisticRegression),
        _ => Err(anyhow::anyhow!("No selection made")),
    }
}

pub fn select_train_or_infer(
    trn_img: &[u8],
    trn_lbl: &[u8],
    tst_img: &[u8],
    tst_lbl: &[u8],
) -> Result<()> {
    loop {
        let items = vec!["Train", "Inference", "Build PCA", "Exit"];
        let selection = FuzzySelect::new()
            .with_prompt("Select and option:")
            .items(&items)
            .interact()?;

        match selection {
            0 => {
                let model = select_model()?;
                let library = select_training_library()?;
                let use_pca = use_pca()?;
                match model {
                    ModelSelection::LinearRegression => {
                        let method = select_solver()?;
                        models::linear_regression::train(
                            trn_img, trn_lbl, library, method, use_pca,
                        )?;
                    }
                    ModelSelection::LogisticRegression => {
                        models::logistic_regression::train(trn_img, trn_lbl, library, use_pca)?;
                    }
                }
            }
            1 => {
                let model = get_weights()?;
                inference::digit_inference(tst_img, tst_lbl, model)?;
            }
            2 => {
                let pca = preprocessing::pca::fit_pca(trn_img);
                preprocessing::pca::save_pca(pca)?;
            }
            _ => break,
        }
    }

    Ok(())
}
