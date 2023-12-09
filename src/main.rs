#[macro_use]
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;

use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::{OwnedArcRepr, OwnedRepr};

// #[]
fn main() {
    let  T = 1000;
    let  N = 200;
    //generate the input data, every arm holds different average values
    let mut data = Array2::zeros((0 , N));
    for _i in 0..T {
        let array = Array::random( N , Uniform::new(0., 1.));
        data.push_row((&array).into()).unwrap();
    }
    
    //check the input data
    println!("{:?}", data);
}

