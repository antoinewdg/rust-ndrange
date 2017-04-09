extern crate rayon;
extern crate ndarray;

mod types;
mod iter;

pub use types::{NdRange, Range1, Range2, Range3, Range4, Range5};
pub use iter::{IntoIter, IntoParIter};

