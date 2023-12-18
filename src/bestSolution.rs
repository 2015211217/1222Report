#![allow(non_snake_case)]
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn best_solution_loss(input_data: Array2<f64>) -> Array1<f64>  {
    let (T, N) = input_data.dim();
    let mut accumulated_loss_vector= input_data.sum_axis(Axis(0));

    let argmin = accumulated_loss_vector.argmin().unwrap();
    //calculate the feedback sequence over T rounds
    let mut best_loss = Array::zeros(T);
    let mut accumulated_loss_single = 0.;

    for _i in 0..T {
        accumulated_loss_single  = accumulated_loss_single + input_data[[_i, argmin]];
        best_loss[_i] = accumulated_loss_single;
    }
    best_loss
}