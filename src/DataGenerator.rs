
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::{OwnedArcRepr, OwnedRepr};
use plotters::prelude::*;
// use yew::prelude::*;
use charming::{component::{Axis, Title}, element::AxisType, series::Line, Chart};

pub fn data_generator(N: usize, T:usize, C:f64) -> Array2<f64> {
    let mut offline_data = Array2::zeros((T, N));
    let mut offline_data_stochastic = Array2::zeros((0, N));
    let mut offline_data_adversarial = Array2::zeros((T, N));

    for _i in 0..T {
        //stochastic data generator
        let array = Array::random( N , Uniform::new(0., 1.));
        offline_data.push_row((&array).into()).unwrap();
    }
    //adversarial data generator
    let mut delta = 0.05;
    let mut interator = 0.;
    let mut flag:bool = true;
    for _i in 0.. (T/2) as usize {
        if _i == (T/2) as usize {
            flag = !flag;
        }
        for _j in 0..N {
            if flag {
                offline_data_adversarial[[_i, _j]] = offline_data_stochastic[[_i, _j]];
                flag = false;
            } else {
                offline_data_adversarial[[_i, _j]] = 0.95 - delta * interator;
                interator += 1.;
                interator %= 10.;
                flag = true;
            }
        }
    }

    if C > 0.0 {
        for _i in 0..T {
            for _j in 0..N {
                offline_data[[_i, _j]] = offline_data_stochastic[[_i, _j]] * (1. - C)
                    + offline_data_adversarial[[_i, _j]] * C;
            }
        }
    }
    if C == 0.0 {
        offline_data = offline_data_stochastic;
    }
    //adversarial data generator
    offline_data
}