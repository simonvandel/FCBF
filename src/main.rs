#![feature(nll)]
extern crate ndarray;

use std::hash::Hash;
use csv::ReaderBuilder;

use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::ShapeBuilder;
use ndarray::{ArrayView, Ix1, Ix2};
use std::collections::HashMap;

extern crate csv;
extern crate fnv;
extern crate indexmap;
extern crate noisy_float;
extern crate num_traits;
extern crate ordered_float;


mod information_theory;
use information_theory::*;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

type ValueType = f64;

fn symmetric_uncertainty_cached<'a>(cacher: &mut MyCacher, x: &'a Col<'a>, y: &'a Col<'a>) -> ValueType {
    symmetric_uncertainty(&x.values, &y.values)
    // println!("{} {}", x.idx, y.idx);
    // let entropy_x = cacher.get_entropy(x.idx, || entropy(&x.values));
    // let entropy_y = entropy(&y.values);
    // let cond_entr = cacher.get_cond_entropy((x.idx, y.idx), ||conditional_entropy(&x.values, &y.values));
    // let ig_x_given_y = entropy_x - cond_entr;

    // let term = checked_div(ig_x_given_y, entropy_x + entropy_y).unwrap_or(0.0);
    // let res = 2.0 * term;
    // debug_assert!(res >= 0.0 && res <= 1.0);
    // res
}

enum CacheElem {
    SU
}

type MyCacher = Cacher<usize, ValueType>;

struct Cacher<K, V> {
    entropy_cache: HashMap<K,V>,
    cond_entropy_cache: HashMap<(K, K),V>
}

impl<K,V> Cacher<K,V> where K: Hash + Eq {
    fn new() -> Self {
        let entropy_cache = HashMap::new();
        let cond_entropy_cache = HashMap::new();
        Cacher{entropy_cache, cond_entropy_cache}
    }

    fn get_entropy<F>(&mut self, k: K, fn_if_not_found: F) -> &V where F: Fn() -> V {
        self.entropy_cache.entry(k).or_insert_with(fn_if_not_found)
    }

    fn get_cond_entropy<F>(&mut self, k: (K, K), fn_if_not_found: F) -> &V where F: Fn() -> V {
        self.cond_entropy_cache.entry(k).or_insert_with(fn_if_not_found)
    }
}

#[derive(Clone, Debug)]
struct Col<'a> {
    idx: usize,
    values: ArrayView<'a, ValueType, Ix1>,
}

#[derive(Clone, Debug)]
struct Feature<'a> {
    col: Col<'a>,
    su_target: ValueType,
}

/// Fast Correlation Based Feature Selection
///
/// `x` is a matrix of features (rows=instances, cols=features).
/// `y` is a matrix of class labels
///
/// The returned view into the subarray of features,
/// has the same lifetime as the original array of features.
pub fn fcbf<'x>(
    x: ArrayView<'x, ValueType, Ix2>,
    y: ArrayView<'x, ValueType, Ix1>,
    threshold: f64,
) -> Array<ValueType, Ix2> {

    let mut cacher = Cacher::new();

    // keeps indices of selected features along with their su_i_c value
    let mut s_list: Vec<Feature> = Vec::new();

    // index of target label is num_features
    let f_y = Col{idx: x.cols(), values: y};

    // gen col structures
    let cols: Vec<Col> = x.gencolumns()
    .into_iter()
    .enumerate()
    .map(|(idx, f_i)| {
        Col {
            idx,
            values: f_i,
        }}).collect();
    

    // calculate all C-correlations (feature-class correlations)
    for c in cols {
        let su_i_c = symmetric_uncertainty_cached(&mut cacher, &c, &f_y);
        if su_i_c >= threshold {
            s_list.push(Feature{col: c, su_target: su_i_c});
        }
    }

    // order s_list in descending SU i_c value
    s_list.sort_unstable_by(|a, b| b.su_target.partial_cmp(&a.su_target).unwrap());

    // TODO: drain_filter sounds nice
    let mut cur_feature_idx = 0;

    while let Some(f_p) = s_list.get(cur_feature_idx) {
        // check all features after the current index
        let unexplored = s_list.iter().skip(cur_feature_idx + 1).filter(|f_q| {
            assert_ne!(f_p.col.idx, f_q.col.idx);
            let su_p_q = symmetric_uncertainty_cached(&mut cacher,
                    &f_q.col,
                    &f_p.col,
                );
            let f_q_su_target = symmetric_uncertainty_cached(&mut cacher, &f_q.col, &f_y);

            // we want to keep f_q if its SU against the class is stronger than the rendundancy score between f_p and f_q
            f_q_su_target > su_p_q
        });

        // unexplored contains every feature that is still interesting to see.

        let predominant: Vec<Feature> =
            s_list.iter().take(cur_feature_idx + 1).cloned().collect();

        let mut new_s_list: Vec<Feature> = Vec::new();
        new_s_list.extend(predominant);

        new_s_list.extend(unexplored.cloned());
        s_list = new_s_list.clone();

        // next iteration we want to use at the next unexplored feature as a reference
        cur_feature_idx += 1;
    }

    s_list.sort_by(|a, b| {
        b.su_target.partial_cmp(&a.su_target).unwrap()
        });
    for x in s_list.iter() {
        println!("{}\t\t {}", x.su_target, x.col.idx);
    }


    // resulting array of selected features can be constructed from the indices in s_list
    // non-contigious slicing is not available, so we need to copy the features into a new array
    // TODO: maybe copy of data can be avoided?
    let feature_views: Vec<_> = s_list.iter().map(|f| f.col.idx).collect();
    x.select(Axis(1), &feature_views)
}

fn main() {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path("data/oil.csv")
        .unwrap();

    let mut data_y: Vec<ValueType> = Vec::new();
    let mut num_features: usize = 0;
    let mut num_instances: usize = 0;
    let mut col_order_data: Vec<Vec<ValueType>> = Vec::new();
    let mut total_col_order: Vec<ValueType> = Vec::new();

    let mut iter = rdr.deserialize();
    while let Some(result) = iter.next() {
        let record: Vec<f64> = result.unwrap();
        num_features = record.len() - 1;
        if col_order_data.len() == 0 {
            for _ in 0..num_features {
                col_order_data.push(Vec::new());
            }
        }
        for (idx, x) in record.iter().take(num_features).enumerate() {
            col_order_data[idx].push(*x);
        }

        data_y.push(*record.last().unwrap());
        num_instances += 1;
    }

    for col in col_order_data {
        total_col_order.extend(col);
    }

    let shape_builder = (num_instances, num_features).f();
    let x: Array<ValueType, Ix2> =
        ArrayBase::from_shape_vec(shape_builder, total_col_order).unwrap();
    let y: Array<ValueType, Ix1> = ArrayBase::from_vec(data_y);

    let _res = fcbf(x.view(), y.view(), 0.0);
}


