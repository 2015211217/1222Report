use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};

pub fn best_solution_loss(N: usize, T: usize, input_data: Array2<f64>) -> ArrayBase<OwnedRepr<_>, Ix2> {
    let mut accumulated_loss_vector= input_data.sum_axis(Axis(0));
    println!("{:?}", accumulated_loss.shape());
    let argmax = accumulated_loss.argmin().unwrap();
    //calculate the feedback sequence over T rounds
    let mut best_loss = Array2::zeros((0, T));
    let mut accumulated_loss_single = 0;
    for _i in 0..T {
        accumulated_loss_single += input_data.row_mut(_i)
        best_loss.append(Axis(0), input_data[_i][argmax]);
    }

    best_loss
}