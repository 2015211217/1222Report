#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn FTL_Algorithm_loss(input_data: &mut Array2<f64>) -> Array1<f64>  {
    let (T, N) = input_data.dim();
    let mut FTL_loss = Array::zeros(T);
    let mut accumulated_loss_vector = Array::zeros(N);
    let mut chosen_rounds = Array::zeros(N);
    // Pull each arm once at first
    let mut accumulated_loss_single = 0.;
    for _i in 0..N:
        chosen_rounds[_i] += 1;
        accumulated_loss_vector[_i] += input_data[[_i, _i]];
        accumulated_loss_single += accumulated_loss_vector[_i];
        FTL_loss[_i] = accumulated_loss_single;
    // Follow the arm with smallest average loss
    for _i in N..T:
        let argmin = accumulated_loss_vector.argmin().unwrap();
        accumulated_loss_vector[argmin] =  (accumulated_loss_vector[argmin] * chosen_rounds[argmin] +
            input_data[[_i, argmin]]) / (chosen_rounds[argmin] + 1);
        chosen_rounds[argmin] += 1;
        accmulated
        FTL_loss[_i] = accumulated_loss_single;
    FTL_loss
}