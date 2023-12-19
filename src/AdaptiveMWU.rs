#![allow(non_snake_case)]

use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_stats::{Quantile1dExt, QuantileExt};
use std::f64::consts::E;
use ndarray_rand::rand_distr::num_traits::real::Real;
use rand::Rng;
use std::cmp::min;
use gurobi::*;
// use good_lp::{constraint};


pub fn Adaptive_MWU_algorithm(input_data: Array2<f64>) -> Array1<f64> {
    let (T, N) = input_data.dim();
    let mut Adaptive_MWU_loss = Array::zeros(T);
    let mut Adaptive_MWU_single = 0.;
    let mut accumulated_loss = Array::<f64, _>::zeros((T));
    let mut weight = Array::<f64, _>::zeros((N));
    let mut z_s = 0.;

    for _i in 0..N {
        weight[_i] = 1. / (N as f64);
    }

    for _i in 0..T {
        //recompute eta
        let mut eta = (1 / 4) as f64;

        // let mut eta = sqrtf32(logf32(N) / (1 ));
        //choose an arm and get feedback
        let mut rng = rand::thread_rng();
        let rng_number = rng.gen();
        let mut mediate = 0.;
        let mut random_arg= 1;

        for _m in 1..N{
            mediate += weight[_m];
            if mediate <= rng_number && mediate - weight[_m] >= rng_number {
                random_arg = _m;
                break;
            }
        }
        Adaptive_MWU_single += input_data[[_i, random_arg]];
        Adaptive_MWU_loss[_i] = Adaptive_MWU_single;
        //renew weight -- gurobi
        let mut entropy = 0.;
        for _j in 0..N {
            accumulated_loss[_j] += input_data[[_i, _j]];
            entropy += weight[_j] * f64::ln(weight[_j]) * (1. / eta);
        }

        let env = gurobi::Env::new("log").unwrap();
        // let mut model = env.new_model("model1").unwrap();
        // let mut weight_iterator = Vec::new();
        // // let mut weight_iterator_2 = Vec::new();
        //
        // for _m in 0..N {
        //     weight_iterator.push(model.add_var("probabilities", Continuous, 0.0, 0.0, 1.0, &[], &[] ).unwrap());
        //     // weight_iterator_2.push(model.add_var("probabilities", Continuous, 0.0, 0.0, 1.0, &[], &[] ).unwrap());
        // }
        // model.update().unwrap();
        // //constraints: sum is one;
        // // let mut iter:LinExpr;
        // //
        // // for _m in 0..N {
        // //     let _iter = iter;
        // //     iter = _iter + weight_iterator[_m];
        // // }
        // let mut objective = LinExpr::new();
        // // for _m in 1..N {
        // //     objective += accumulated_loss[_m] * weight_iterator[_m];
        // // }
        // let mut iter = LinExpr::new();
        // for _m in 0..N {
        //     iter += weight_iterator[_m].clone();
        // }
        // let constr = model.add_constr("c0", iter, Equal, 1.0).unwrap();
        // model.update().unwrap();
        //
        // // let objective = accumulated_loss * weight_iterator;
        // model.set_objective(objective + (1.0 / eta) * entropy, Minimize).unwrap();
        // model.optimize().unwrap();
        // let result = model.get_values(attr::X, &weight_iterator).unwrap();
        // for _m in 0..N {
        //     weight[_m] = result[_m];
        // }

    }
    Adaptive_MWU_loss
}