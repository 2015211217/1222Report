#![allow(non_snake_case)]
#![feature(collections)]

use ndarray::prelude::*;
use ndarray::OwnedRepr;

use ndarray_stats::{Quantile1dExt, QuantileExt};
use rand::seq::SliceRandom;
use ndarray_rand::rand::prelude::IteratorRandom;
// use std::intrinsics::{sqrtf32, logf32, expf64};
// use std::cmp;
use std::f64::consts::E;
use std::cmp::min;
use rand::Rng;
// use yew::preclude::*;


pub fn MWU_algorithm(input_data: Array2<f64>) -> Array1<f64> {
    let (T, N) = input_data.dim();
    let mut MWU_loss = Array::<f64, _>::zeros((T));
    // let mut accumulated_loss = Array::<f64, _>::zeros((N));
    let mut weight = Array::<f64, _>::zeros((N));
    for _i in 0..N {
        weight[_i] = 1. / (N as f64);
    }
    let mut eta = 2.0 * (f64::ln(N as f64)).sqrt() / T as f64;
    if eta < 1. {eta = 1.;}

    let mut accumulated_loss_single = 0.;
    for _i in 0..T {
        // choose one arm
        let mut rng = rand::thread_rng();
        let rng_number = rng.gen();
        let mut mediate = 0.;
        let mut random_arg= 1;

        for _m in 1..N{
            mediate += weight[_m];

            if mediate >= rng_number && mediate - weight[_m] <= rng_number {
                random_arg = _m;
                break;
            }
        }

        accumulated_loss_single += input_data[[_i, random_arg]];

        MWU_loss[_i] = accumulated_loss_single;
        // renew the weight
        for _j in 0..N {
            weight[_j] = ((-1.) * eta * input_data[[_i,_j]]).exp();
        }
        let weight_sum = weight.sum();
        for _j in 0..N {
            weight[_j] = weight[_j] / weight_sum;
        }
        //probably precise problem
    }
    MWU_loss
}