use super::super::prepare_trn_lbl_nalgebra;
use crate::EPSILON;
use anyhow::Result;
use nalgebra::DMatrix;
use nalgebra_lapack::QrDecomposition;

pub mod n_algebra {
    use super::*;

    pub mod pca {
        use super::*;
        use crate::PCA_COMPONENTS;
        use crate::preprocessing::pca;

        // Returns the factorized A matrix as QR
        pub fn decompose(
            matrix: DMatrix<f64>,
        ) -> Result<nalgebra_lapack::QR<f64, nalgebra::Dyn, nalgebra::Dyn>> {
            let mut pca = pca::open_pca()?;
            let transformed_matrix = pca::pca_transform(&mut pca, &matrix, PCA_COMPONENTS);
            let transformed_matrix = transformed_matrix.insert_column(0, 1.0);
            let qr = nalgebra_lapack::QR::new(transformed_matrix)?;
            Ok(qr)
        }

        pub fn solve(
            qr: &nalgebra_lapack::QR<f64, nalgebra::Dyn, nalgebra::Dyn>,
            trn_lbl: &[u8],
        ) -> Result<DMatrix<f64>> {
            let mut all_weights = DMatrix::zeros(qr.ncols(), 10);
            for digit in 0..=9 {
                println!("Training {digit}");
                let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                let weights = qr.solve(train_label)?;
                all_weights.set_column(digit as usize, &weights);
            }

            Ok(all_weights)
        }
    }

    // QR nAlgebra with Column Pivoting
    // Returns a tuple containing a trimmed version of Q and R matrices
    pub fn decompose(
        x: DMatrix<f64>,
    ) -> (
        DMatrix<f64>,
        DMatrix<f64>,
        nalgebra::PermutationSequence<nalgebra::Dyn>,
    ) {
        let qr = x.col_piv_qr();

        let (q, rt, p) = qr.unpack();
        let qt = q.transpose();

        let rank = (0..rt.nrows().min(rt.ncols()))
            .take_while(|&i| rt[(i, i)].abs() > EPSILON)
            .count();

        let qt_trimmed = qt.rows(0, rank).into_owned();
        let rt_trimmed = rt.view((0, 0), (rank, rank)).into_owned();

        (qt_trimmed, rt_trimmed, p)
    }

    pub fn solve(
        trn_lbl: &[u8],
        n_features: usize,
        q: DMatrix<f64>,
        rt: DMatrix<f64>,
        p: nalgebra::PermutationSequence<nalgebra::Dyn>,
    ) -> Result<DMatrix<f64>> {
        let mut all_weights = DMatrix::zeros(n_features, 10);
        for digit in 0..=9 {
            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
            let qtb = &q * train_label;
            let weights = rt.solve_upper_triangular(&qtb).unwrap();
            let mut weights = weights.resize_vertically(n_features, 0.0);
            p.inv_permute_rows(&mut weights);
            all_weights.set_column(digit as usize, &weights);
        }
        Ok(all_weights)
    }
}

// fn qr_least_squares_faer(matrix: Mat<f64>, vector: Col<f64>, digit: u8) -> Weights {
//     let qr = matrix.col_piv_qr();
//     let q = qr.compute_thin_Q();
//     let rt = qr.R().to_owned();
//
//     let rank = (0..rt.nrows().min(rt.ncols()))
//         .take_while(|&i| rt[(i, i)].abs() > EPSILON)
//         .count();
//
//     println!("Rank: {rank}");
//
//     // Q^T * b
//     let qtb = q.transpose() * vector;
//
//     let (qtb_truncated, _discard) = qtb.split_at_row(rank);
//     let mut qtb_truncated = qtb_truncated.to_owned();
//
//     // R (upper right triangle matrix)
//     let rt = rt.submatrix(0, 0, rank, rank);
//
//     solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::rayon(0));
//     // solve_upper_triangular_in_place(rt, qtb_truncated.as_mat_mut(), Par::Seq);
//
//     let mut permutated_y = qtb_truncated;
//     permutated_y.resize_with(matrix.ncols(), |_| 0.0);
//
//     let p = qr.P();
//
//     let (forward_idx, _inverse_idx) = p.arrays();
//
//     let mut x = Mat::<f64>::zeros(permutated_y.nrows(), 1);
//
//     for (i, &orig_col) in forward_idx.iter().enumerate() {
//         x[(orig_col, 0)] = permutated_y[i];
//     }
//
//     Weights::new(x.col_as_slice(0), digit, false)
// }
