use crate::{Library, Method};
use anyhow::{Context, Result};
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

pub fn select_training_method() -> Result<Method> {
    let items = vec!["SVD", "QR", "Logistic"];
    let selection = FuzzySelect::new()
        .with_prompt("Select and option:")
        .items(&items)
        .interact()?;

    match selection {
        0 => Ok(Method::SVD),
        1 => Ok(Method::QR),
        2 => Ok(Method::Logistic),
        _ => todo!(),
    }
}
