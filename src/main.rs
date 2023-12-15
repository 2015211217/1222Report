#![allow(non_snake_case)]
mod bestSolution;
mod FTL;
mod MWU;
mod AdaptiveMWU;

#[macro_use]
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate plotters;

use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::{OwnedArcRepr, OwnedRepr};
use plotters::prelude::*;
// use yew::prelude::*;
use charming::{component::{Axis, Title}, element::AxisType, series::Line, Chart};

fn main() {
    let  T = 1000;
    let  N = 200;
    if N > T {
        panic!("Come on! Too little rounds!!");
    }
    //generate the input data, every arm holds different average values
    let mut offline_data = Array2::zeros((0 , N));
    for _i in 0..T {
        let array = Array::random( N , Uniform::new(0., 1.));
        offline_data.push_row((&array).into()).unwrap();
    }
    //check the input data
    // println!("{:?}", offline_data);
    //get the best solution
    let best_solution = bestSolution::best_solution_loss(&mut offline_data);
    let FTL = FTL::FTL_Algorithm_loss(&mut offline_data);
    let MWU = MWU::MWU_algorithm(&mut offline_data);
    let AdaptiveMWU = AdaptiveMWU::Adaptive_MWU_algorithm(&mut offline_data);
    // plot the data

    // let root = BitMapBackend::new("figures/regret.png", (640, 480)).into_drawing_area();
    // let graph = yew_hooks::use_async::<_, _, ()>({
    //
    // });
    //
    // let renderer = WasmRenderer::new(600,400);
    // async move {
    //     renderer.render("chart", &chart).unwarp();
    //     Ok(());
    // }
    // root.present();
    // Ok::<(), E>(());
}

