use crate::EPSILON;
use crate::PCA_COMPONENTS;
use anyhow::Result;
use nalgebra::{DMatrix, DVector};

pub mod n_algebra {
    use super::*;

    pub mod pca {
        use super::*;
        use crate::preprocessing::pca;

        pub fn decompose(matrix: DMatrix<f64>) -> Result<DMatrix<f64>> {
            let mut pca = pca::open_pca()?;
            let transformed_matrix = pca::pca_transform(&mut pca, &matrix, PCA_COMPONENTS);
            let transformed_matrix = transformed_matrix.insert_column(0, 1.0);
            let pseudo_inverse = transformed_matrix.pseudo_inverse(EPSILON).unwrap();
            let rank = pseudo_inverse.rank(EPSILON);
            println!("Matrix Rank: {rank}");

            Ok(pseudo_inverse)
        }
    }

    // Builds the pseudoinverse using SVD
    pub fn decompose(matrix: DMatrix<f64>) -> Result<DMatrix<f64>> {
        let matrix = matrix.insert_column(0, 1.0);
        let svd = nalgebra_lapack::SVD::new(matrix).unwrap();
        let rank = svd.rank(EPSILON);
        println!("Matrix Rank: {rank}");

        let ut = svd.u.transpose();

        // Trim U^T
        let mut sigma_inv_ut = ut.rows(0, rank).into_owned();

        // Multiply U^T by the inverted singular values to give E^-1 * U^T
        for i in 0..rank {
            let inv_sigma = 1.0 / svd.singular_values[i];

            for j in 0..sigma_inv_ut.ncols() {
                sigma_inv_ut[(i, j)] *= inv_sigma;
            }
        }

        // Trim V^T
        let vt = svd.vt.rows(0, rank).transpose();

        // Create pseudo inverse by multiplying V^T*E^-1*U^T
        let pseudoinverse = vt * sigma_inv_ut;

        Ok(pseudoinverse)
    }

    pub fn solve(pseudo_inverse: &DMatrix<f64>, trn_lbl: &[u8]) -> Result<DMatrix<f64>> {
        let mut all_weights = DMatrix::zeros(pseudo_inverse.nrows(), 10);
        for i in 0..=9 {
            let train_label =
                DVector::from_row_slice(trn_lbl).map(|digit| if digit == i { 1.0 } else { 0.0 });
            let weights = pseudo_inverse * train_label;
            all_weights.set_column(i as usize, &weights);
        }

        Ok(all_weights)
    }
}

// Tolerance (Epsilon) is set internally by Faer
// fn svd_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
//     let svd = matrix.thin_svd().unwrap();
//
//     let pseudo_inverse = svd.pseudoinverse();
//     let solution = pseudo_inverse * vector;
//     let solution: Vec<f64> = solution.iter().copied().collect();
//
//     Weights::new(solution.as_slice(), digit, false)
// }
