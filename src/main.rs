#![feature(nll)]
extern crate ndarray;

use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::{ArrayView, Ix1, Ix2};


fn f<'a>(y: &'a Col<'a>) {}

struct Col<'a> {
    values: ArrayView<'a, f64, Ix1>,
}

pub fn x<'x>(
    y: ArrayView<'x, f64, Ix1>
) {
    let f_y = Col{values: y};

    let cols: Vec<Col> = unimplemented!();
    
    f(&f_y);
}

fn main() {
    let data_y = Vec::<f64>::new();
    let y: Array<f64, Ix1> = Array::from_vec(data_y);

    let _res = x(y.view());
}