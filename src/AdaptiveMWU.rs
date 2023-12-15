#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};
use std::intrinsics::sqrtf32;
use ndarray_rand::rand_distr::num_traits::real::Real;
// use std::intrinsics::{sqrtf32, logf32};

pub fn Adaptive_MWU_algorithm(input_data: &mut Array2<f64>) -> Array1<f64> {
    let (T, N) = input_data.dim();
    let mut Adaptive_MWU_loss = Array::zeros(T);
    let mut Adaptive_MWU_single = 0.;
    let mut accumulated_loss = Array::<f64, _>::zeros((T));
    let mut weight = Array::<f64, _>::zeros((N));

    for _i in 0..N {
        weight[_i] = 1. / (N as f64);
    }

    for _i in 0..T {
        //recompute eta
        let mut eta = sqrtf32(N.log()/ (1 + ))

        // let mut eta = sqrtf32(logf32(N) / (1 ));
        //choose an arm and get feedback
        
        Adaptive_MWU_loss[_i] = Adaptive_MWU_single;
        //renew weight
        let mut entropy = 0.;
        for _j in 0..N {
            entropy += weight[_j] * weight[_j].log();
        }
        for _j in 0..N {

        }
    }
    Adaptive_MWU_loss
}