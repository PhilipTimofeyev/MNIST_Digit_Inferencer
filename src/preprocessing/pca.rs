use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pca {
    mean: DVector<f64>,
    components: DMatrix<f64>,
}

pub fn save_pca(pca: Pca) -> Result<()> {
    let path = std::path::Path::new("./pca");
    std::fs::create_dir_all(path)?;

    let filename = "pca/pca.json";
    let file = File::create(filename).context("Failed to create file at path")?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, &pca)
        .context("Failed to serialize PCA into JSON format")?;

    Ok(())
}

impl Pca {
    pub fn k_components(&mut self, k: usize) -> Pca {
        self.components = self.components.columns(0, k).into_owned();
        self.to_owned()
    }
}

pub fn open_pca() -> Result<Pca> {
    let filename = "pca/pca.json";
    let file = File::open(filename)?;
    let weights_json = BufReader::new(file);
    let weights =
        serde_json::from_reader(weights_json).context("Failed to deserialize weights JSON")?;
    Ok(weights)
}

pub fn pca_transform(pca: &mut Pca, matrix: &DMatrix<f64>, k_components: usize) -> DMatrix<f64> {
    let matrix = matrix.clone_owned();
    let centered = DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        matrix[(i, j)] - pca.mean[j]
    });

    let pca = pca.k_components(k_components);

    centered * &pca.components
}

pub fn fit_pca(trn_img: &[u8]) -> Pca {
    let matrix = DMatrix::from_row_slice(60000, 784, trn_img).map(|pixel| pixel as f64 / 255.0);
    let m = matrix.nrows();
    let n = matrix.ncols();

    // Calculate column means
    let mut means = vec![0.0; n];

    for j in 0..n {
        let mut sum = 0.0;

        for i in 0..m {
            sum += matrix[(i, j)];
        }

        means[j] = sum / m as f64;
    }

    // Center the data using the means
    let centered = DMatrix::from_fn(m, n, |i, j| matrix[(i, j)] - means[j]);

    // make sure not to divide by 0
    let denominater = if m > 1 { (m - 1) as f64 } else { 1.0 };
    // Build covariance matrix which represents how feature change together
    // The diagonal represents the variance of each specific feature
    let covariance_matrix = (&centered.transpose() * &centered) / denominater;

    // Get the eigendecomposition, eigenvectors represent the directions of maximum variance
    // The eigenvalues represent the amount of variance in an eigenvector
    let eigen = SymmetricEigen::new(covariance_matrix);

    let mut eigenpairs: Vec<(f64, DVector<f64>)> = (0..n)
        .map(|i| {
            (
                eigen.eigenvalues[i],
                eigen.eigenvectors.column(i).into_owned(),
            )
        })
        .collect();

    // Sort by eigenvalue, largest first to denote which direction has the most
    // variance/significance
    eigenpairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Sets the number of components
    let k = 784;

    let components = DMatrix::from_columns(
        &eigenpairs[..k]
            .iter()
            .map(|(_, v)| v.clone())
            .collect::<Vec<_>>(),
    );

    Pca {
        mean: DVector::from_vec(means),
        components,
    }
}
