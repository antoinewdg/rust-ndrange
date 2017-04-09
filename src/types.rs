use ndarray::{Dimension, IntoDimension, Ix1, Ix2, Ix3, Ix4, Ix5};

/// A multi-dimensional range.
///
/// Similar to `std::ops::Range` but can be used for
/// multiple dimensions, using `ndarray`'s `Dimension` trait.
///
/// # Examples
///
/// `NdRange` can be used like a regular `Range`.
///
/// ```
/// use ndrange::Range2;
///
/// let range = Range2::new((0,0), (2,3));
/// let mut vec = Vec::new();
/// for (a, b) in range {
///     vec.push((b, a));
/// }
/// assert_eq!(vec, vec![(0, 0), (1, 0), (2, 0),
///                      (0, 1), (1, 1), (2, 1)]);
/// ```
///
/// It can also be used with `rayon`'s `ParallelIterator`.
///
/// ```
/// extern crate rayon;
/// extern crate ndrange;
///
/// use rayon::prelude::*;
/// use ndrange::Range2;
///
/// fn main() {
///     let range = Range2::new((0, 0), (2, 3));
///     let vec = range.into_par_iter().collect::<Vec<_>>();
///     assert_eq!(vec, vec![(0, 0), (0, 1), (0, 2),
///                          (1, 0), (1, 1), (1, 2)]);
/// }
/// ```
///
#[derive(Clone, Debug)]
pub struct NdRange<D> {
    /// The first element of the range
    pub start: D,
    /// The end value (not included) for each dimension
    pub end: D,
}


impl<D> NdRange<D> where D: Dimension {
    /// Create a new `NdRange`.
    pub fn new<I>(start: I, end: I) -> Self where I: IntoDimension<Dim=D> {
        let start = start.into_dimension();
        let end = end.into_dimension();
        NdRange { start: start.clone(), end: end }
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        (self.end.clone() - self.start.clone()).size()
    }
}

pub type Range1 = NdRange<Ix1>;
pub type Range2 = NdRange<Ix2>;
pub type Range3 = NdRange<Ix3>;
pub type Range4 = NdRange<Ix4>;
pub type Range5 = NdRange<Ix5>;


#[cfg(test)]
mod tests {
    use ::Range3;

    #[test]
    fn len() {
        assert_eq!(Range3::new((0, 0, 0), (1, 1, 1)).len(), 1);
        assert_eq!(Range3::new((0, 0, 0), (1, 1, 5)).len(), 5);
        assert_eq!(Range3::new((0, 0, 0), (1, 5, 1)).len(), 5);
        assert_eq!(Range3::new((0, 0, 0), (5, 1, 1)).len(), 5);
        assert_eq!(Range3::new((1, 1, 1), (5, 2, 3)).len(), 8);
        assert_eq!(Range3::new((5, 6, 7), (10, 10, 10)).len(), 5 * 4 * 3);
        assert_eq!(Range3::new((100, 10, 100), (102, 19, 133)).len(), 2 * 9 * 33);
    }
}