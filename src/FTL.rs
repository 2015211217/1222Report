use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

fn best_solution_loss(N: i32, T:i32, input_data: Array2<f64>) -> ArrayBase<OwnedRepr<_>, i32> {
    let mut accumulated_loss= input_data.sum_axis(Axis(0));
    println!("{:?}", accumulated_loss.shape());
    let argmax = accumulated_loss.argmin().unwrap();
    //calculate the feedback sequence over T rounds
    let mut best_loss = Array::zero(T);

    best_loss
}