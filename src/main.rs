#![allow(non_snake_case)]
mod bestSolution;
mod FTL;
mod MWU;
mod AdaptiveMWU;
mod DataGenerator;

#[macro_use]
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
// extern crate plotters;
// extern crate gurobi;

use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::{OwnedArcRepr, OwnedRepr};
// use plotters::prelude::*;
// use yew::prelude::*;
// use charming::{component::{Axis, Title}, element::AxisType, series::Line, Chart};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter, IoSlice};


fn main() {
    let  T = 1000;
    let  N = 200;
    let C = 0.;
    if N > T {
        panic!("Come on! Too little rounds!!");
    }
    //generate the input data, every arm holds different average values
    //new generating technics
    let offline_data = DataGenerator::data_generator(N, T, C);
    //check the input data
    // println!("{:?}", offline_data);
    //get the best solution
    let best_solution = bestSolution::best_solution_loss(offline_data.clone());
    let FTL_loss = FTL::FTL_Algorithm_loss(offline_data.clone()) - best_solution.clone();
    let MWU_loss = MWU::MWU_algorithm( offline_data.clone()) - best_solution.clone();
    // let AdaptiveMWU_loss = AdaptiveMWU::Adaptive_MWU_algorithm(offline_data.clone()) - best_solution.clone();
    // plot the data
    println!("{}", FTL_loss);
    println!("{}", MWU_loss);
    // println!("{}", AdaptiveMWU_loss);
    // let mut stream = BufWriter::new(FTL_loss).unwrap();
    let mut file = File::create("Regret.txt").unwrap();
    file.write(b"[");
    for _i in 0..T {
        let s1 = format!("{}, ", FTL_loss[_i]);
        file.write(s1.as_bytes());
    }
    file.write(b"]");
    file.write(b"\n");
    file.write(b"[");
    for _i in 0..T {
        let s2 = format!("{}, ", MWU_loss[_i]);
        file.write(s2.as_bytes());
    }
    file.write(b"]");

    // file.write(b"[");
    // for _i in 0..T {
    //     let s3 = format!("{}, ", AdaptiveMWU_loss[_i]);
    //     file.write(s3.as_bytes());
    // }
    // file.write(b"]");
    // // for _i in 0..T {
    // //     for _j in 0..N {
    // //         file.write(FTL_loss[[_i, _j]]);
    // //     }
    // // }
    // let io_slice1 = IoSlice::new(FTL_loss);
    // file.write_all(&io_slice1);
    //
    // file.write(b"Hello World!");
    //
    // // file.write(FTL_loss)?;
    // file = OpenOptions::new().append(true).open("Regret.txt").unwrap();
    // file.write(b"Hello World!");

    // file.write(MWU_loss)?;
    // file.write(AdaptiveMWU_loss)?;
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

