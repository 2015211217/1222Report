#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn FTL_Algorithm_loss(input_data: &mut Array2<f64>) -> Array1<f64>  {
    let (T, N) = input_data.dim();
    let mut FTL_loss = Array::zeros((T));
    let mut chosen_rounds = Array::<f64, _>::zeros((N));
    let mut accumulated_loss= Array::<f64, _>::zeros((N));

    // Pull each arm once at first
    let mut accumulated_loss_single = 0.;

    for _i in 0..N {
        chosen_rounds[_i] += 1.0;
        accumulated_loss[_i] += input_data[[_i, _i]];
        accumulated_loss_single += input_data[[_i, _i]];
        FTL_loss[_i] = accumulated_loss_single / chosen_rounds[_i];
    }
    // Follow the arm with smallest average loss
    for _i in N..T {
        let argmin = accumulated_loss.argmin().unwrap();
        accumulated_loss[argmin] = accumulated_loss[argmin] + input_data[[_i, argmin]];
        chosen_rounds[argmin] += 1.0;
        accumulated_loss_single += input_data[[_i, argmin]];

        FTL_loss[_i] = accumulated_loss_single / chosen_rounds[argmin];
    }
    FTL_loss
}