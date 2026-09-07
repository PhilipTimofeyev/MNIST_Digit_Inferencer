use super::super::{prepare_trn_img_nalgebra, save_weights};
use crate::{
    EPSILON, Library, Model, ModelType, PCA_COMPONENTS, Solver,
    solvers::{qr, svd},
};
use anyhow::Result;
use nalgebra_lapack::QrDecomposition;
use std::time::Instant;

pub fn train(
    trn_img: &[u8],
    trn_lbl: &[u8],
    library: Library,
    solver: Solver,
    use_pca: bool,
) -> Result<()> {
    match library {
        Library::NAlgebra => {
            let train_data = prepare_trn_img_nalgebra(trn_img);
            let start = Instant::now();
            match solver {
                Solver::SVD => {
                    let pseudo_inverse = if use_pca {
                        svd::n_algebra::pca::decompose(train_data)?
                    } else {
                        svd::n_algebra::decompose(train_data)?
                    };
                    let weights = svd::n_algebra::solve(&pseudo_inverse, trn_lbl)?;
                    let model_type = ModelType::LinearRegression {
                        weights,
                        epsilon: EPSILON,
                    };

                    let pca = if use_pca { Some(PCA_COMPONENTS) } else { None };
                    let model = Model::new(pseudo_inverse.nrows(), pca, model_type);
                    save_weights(model)?;
                }
                Solver::QR => {
                    if use_pca {
                        let qr = qr::n_algebra::pca::decompose(train_data)?;
                        let weights = qr::n_algebra::pca::solve(&qr, trn_lbl)?;
                        let model_type = ModelType::LinearRegression {
                            weights,
                            epsilon: EPSILON,
                        };

                        let model = Model::new(qr.nrows(), Some(PCA_COMPONENTS), model_type);
                        save_weights(model)?;
                    } else {
                        let train_data = train_data.insert_column(0, 1.0);
                        let n_features = train_data.ncols();
                        let (q, rt, p) = qr::n_algebra::decompose(train_data);
                        let weights = qr::n_algebra::solve(trn_lbl, n_features, q, rt, p)?;
                        let model_type = ModelType::LinearRegression {
                            weights,
                            epsilon: EPSILON,
                        };
                        let model = Model::new(n_features, None, model_type);
                        save_weights(model)?;
                    }
                }
            }
            println!("Time elapsed: {:?}", start.elapsed());
        }
        Library::Faer => match solver {
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
