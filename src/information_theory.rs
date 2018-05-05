use fnv::FnvBuildHasher;
use indexmap::IndexMap;
use indexmap::IndexSet;
use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Zero;
use ordered_float::OrderedFloat;
use std::iter::Sum;

pub fn symmetric_uncertainty<'a, I, F>(x: &'a I, y: &'a I) -> F
where
    I: IntoIterator<Item = &'a F> + Clone,
        I::IntoIter: Clone,
    F: 'a + Float + FromPrimitive + Zero + Sum,
{
    let entropy_x = entropy(x);
    let entropy_y = entropy(y);
    let cond_entr = conditional_entropy(x, y);
    let ig_x_given_y = entropy_x - cond_entr;

    let term = checked_div(ig_x_given_y, entropy_x + entropy_y).unwrap_or(F::zero());
    let res = F::from_f64(2.0).unwrap() * term;
    debug_assert!(res >= F::from_f64(0.0).unwrap() && res <= F::from_f64(1.0).unwrap());
    res
}

/// Guards agains division by 0
pub fn checked_div<F>(numerator: F, denominator: F) -> Option<F>
where
    F: Float + Zero,
{
    if denominator == Zero::zero() || -denominator == Zero::zero() {
        None
    } else {
        Some(numerator / denominator)
    }
}

pub fn entropy<'a, I, F>(x: &'a I) -> F
where
    I: IntoIterator<Item = &'a F> + Clone,
    F: 'a + Float + FromPrimitive + Sum,
{
    let proba_x = calc_probabilities(x);
    // TODO: can use dot product for speedup?
    let sum: F = proba_x
        .into_iter()
        .map(|prob| {
            let log_expr = if prob == F::from_f64(0.0).unwrap() {
                F::from_f64(0.0).unwrap()
            } else {
                F::log2(prob)
            };

            prob * log_expr
        })
        .sum();
    -sum
}

// TODO: return iterator instead of allocating Vec
pub fn calc_probabilities<'a, I, F>(x: &'a I) -> Vec<F>
where
    I: IntoIterator<Item = &'a F> + Clone,
    F: 'a + Float + FromPrimitive,
{
    let mut instances = 0;
    // calculate probabilities of every class
    // We use an IndexMap to preserve the order of the classes
    let mut frequency = IndexMap::with_hasher(FnvBuildHasher::default());
    for val in x.clone() {
        instances += 1;
        *frequency.entry(OrderedFloat(*val)).or_insert(0) += 1;
    }

    frequency
        .values()
        .map(|count| F::from_u64(*count).unwrap() / F::from_u64(instances).unwrap())
        .collect()
}

/// Returns unique values in x. Keeps order
pub fn unique<'a, F: 'a, I>(x: &'a I) -> Vec<F>
where
    I: IntoIterator<Item = &'a F> + Clone,
    F: Float,
{
    let mut uniq: IndexSet<OrderedFloat<F>> = IndexSet::new();
    for val in x.clone() {
        uniq.insert(OrderedFloat(*val));
    }

    let mut ret: Vec<F> = Vec::new();
    for x in uniq.iter() {
        ret.push(**x);
    }
    ret
}

// TODO: support different lifetimes for x and y

/// H(X | Y)
pub fn conditional_entropy<'a, F, I>(x: &'a I, y: &'a I) -> F
where
    I: IntoIterator<Item = &'a F> + Clone,
    I::IntoIter: Clone,
    F: 'a + Float + Zero + FromPrimitive + Sum,
{
    let proba_y = calc_probabilities(y);
    let mut sum = Zero::zero();
    let classes_y = unique(y);
    for (class_y, p_y) in classes_y.iter().zip(proba_y) {
        let to_entropy = (x.clone().into_iter()).zip(y.clone())
            .filter(|(_, y_val)| *y_val == class_y)
            .map(|(x_val, _)| x_val);

        sum = sum + p_y * entropy(&to_entropy)
    }

    sum
}

#[cfg(test)]
mod tests {

    extern crate float_cmp;
    use self::float_cmp::{ApproxEq, ApproxEqRatio, Ulps};
    use quickcheck::TestResult;

    use super::*;

    // TODO: use discard features
    quickcheck! {
        fn symmetric_uncertainty_between_0_1(x: Vec<f64>, y: Vec<f64>) -> TestResult {
            if x.len() != y.len() {
                return TestResult::discard();
            }
            let res = symmetric_uncertainty(&x.iter(), &y.iter());
            let b = res >= 0.0 && res <= 1.0;
            return TestResult::from_bool(b);
        }
    }

    // TODO: use discard features
    quickcheck! {
        fn symmetric_uncertainty_symmetric(x: Vec<f64>, y: Vec<f64>) -> TestResult {
            if x.len() != y.len() {
                return TestResult::discard();
            }
            let res1 = symmetric_uncertainty(&x.iter(), &y.iter());
            let res2 = symmetric_uncertainty(&x.iter(), &y.iter());
            TestResult::from_bool(res1 == res2)
        }
    }

    // TODO: use discard features
    quickcheck! {
        fn entropy_no_crash(x: Vec<f64>) -> bool {
            entropy(&x.iter());
            return true;
        }
    }

    // TODO: use discard features
    quickcheck! {
        fn calc_probabilities_sum_1(x: Vec<f64>) -> TestResult {
            if x.len() == 0 {
                return TestResult::discard();;
            }
            let res: Vec<f64> = calc_probabilities(&x.iter());
            let sum: f64 = res.into_iter().sum();
            TestResult::from_bool(sum.approx_eq_ratio(&1.0, 0.0001))
        }
    }
}
