#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn MWU_algorithm(input_data: &mut Array2<f64>) -> Array1<f64> {
    let (T, N) = input_data.dim();
    let mut MWU_loss = Array::<f64, _>::zeros((T));
    let mut accumulated_loss = Array::<f64, _>::zeros((N));
    let mut weight = Array::<f64, _>::[1/N ; N];
    for _i in 0..T {

    }
    MWU_loss
}