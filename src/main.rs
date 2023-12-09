mod bestSolution;
mod FTL;
mod MWU;
mod AdaptiveMWU;

#[macro_use]
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;

use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::{OwnedArcRepr, OwnedRepr};

fn main() {
    let  T = 1000;
    let  N = 200;
    //generate the input data, every arm holds different average values
    let mut offline_data = Array2::zeros((0 , N));
    for _i in 0..T {
        let array = Array::random( N , Uniform::new(0., 1.));
        offline_data.push_row((&array).into()).unwrap();
    }

    //check the input data
    // println!("{:?}", offline_data);
    //get the best solution
    let best_solution = bestSolution::best_solution_loss(N, T, offline_data);
    // plot the data
}

