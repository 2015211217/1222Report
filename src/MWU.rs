#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn MWU_algorithm(input_data: &mut Array2<f64>) -> Array1<f64> {
    let (T, N) = input_data.dim();
    let mut MWU_loss = Array::zeros(T);
    
    MWU_loss
}