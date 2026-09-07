use super::super::{
    prepare_trn_img_nalgebra, prepare_trn_lbl_nalgebra, save_weights, svd_train_digits,
};
use crate::{EPSILON, Library, Model, ModelType, PCA_COMPONENTS, Solver, solvers};
use anyhow::Result;
use nalgebra::DMatrix;
use nalgebra_lapack::QrDecomposition;
use std::time::Instant;

pub fn train(
    trn_img: &[u8],
    trn_lbl: &[u8],
    library: Library,
    method: Solver,
    use_pca: bool,
) -> Result<()> {
    match library {
        Library::NAlgebra => {
            let train_data = prepare_trn_img_nalgebra(trn_img);
            match method {
                Solver::SVD => {
                    let start = Instant::now();

                    let pseudo_inverse = if use_pca {
                        solvers::svd::svd_nalgebra_lapack_pca(train_data)?
                    } else {
                        solvers::svd::svd_nalgebra_lapack(train_data)?
                    };

                    svd_train_digits(pseudo_inverse, trn_lbl, use_pca)?;
                    println!("Time elapsed: {:?}", start.elapsed());
                }
                Solver::QR => {
                    let start = Instant::now();

                    if use_pca {
                        let qr = solvers::qr::qr_nalgebra_lapack_pca(train_data)?;
                        let mut all_weights = DMatrix::zeros(qr.ncols(), 10);
                        for digit in 0..=9 {
                            println!("Training {digit}");
                            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                            let weights = qr.solve(train_label)?;
                            all_weights.set_column(digit as usize, &weights);
                        }
                        let model_type = ModelType::LinearRegression {
                            weights: all_weights,
                            epsilon: EPSILON,
                        };

                        let model = Model::new(qr.nrows(), Some(PCA_COMPONENTS), model_type);
                        save_weights(model)?;
                    } else {
                        let train_data = train_data.insert_column(0, 1.0);
                        let n_features = train_data.ncols();
                        let (q, rt, p) = solvers::qr::qr_nalgebra_lapack(train_data);
                        let mut all_weights = DMatrix::zeros(n_features, 10);
                        for digit in 0..=9 {
                            let train_label = prepare_trn_lbl_nalgebra(trn_lbl, digit);
                            let qtb = &q * train_label;
                            let weights = rt.solve_upper_triangular(&qtb).unwrap();
                            let mut weights = weights.resize_vertically(n_features, 0.0);
                            p.inv_permute_rows(&mut weights);
                            all_weights.set_column(digit as usize, &weights);
                        }
                        let model_type = ModelType::LinearRegression {
                            weights: all_weights,
                            epsilon: EPSILON,
                        };

                        let model = Model::new(n_features, None, model_type);
                        save_weights(model)?;
                    }

                    println!("QR elapsed: {:?}", start.elapsed());
                }
            }
        }
        Library::Faer => match method {
            Solver::SVD => {
                todo!();
                // let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
                // // let z = pca(train_data.clone());
                // svd_least_squares_faer(train_data, train_label, i)
            }
            Solver::QR => {
                todo!();
                // let (train_data, train_label) = prepare_train_data_faer(trn_img, trn_lbl, i)?;
                // qr_least_squares_faer(train_data, train_label, i)
            }
        },
    };

    Ok(())
}
