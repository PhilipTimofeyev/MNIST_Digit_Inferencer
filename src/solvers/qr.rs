use super::super::preprocessing::pca;
use crate::EPSILON;
use crate::PCA_COMPONENTS;
use anyhow::Result;
use nalgebra::DMatrix;

// Returns the factorized A matrix as QR
pub fn qr_nalgebra_lapack_pca(
    matrix: DMatrix<f64>,
) -> Result<nalgebra_lapack::QR<f64, nalgebra::Dyn, nalgebra::Dyn>> {
    let mut pca = pca::open_pca()?;
    let transformed_matrix = pca::pca_transform(&mut pca, &matrix, PCA_COMPONENTS);
    let transformed_matrix = transformed_matrix.insert_column(0, 1.0);
    let qr = nalgebra_lapack::QR::new(transformed_matrix)?;
    Ok(qr)
}

// QR nAlgebra with Column Pivoting
// Returns a tuple containing a trimmed version of Q and R matrices
pub fn qr_nalgebra_lapack(
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
