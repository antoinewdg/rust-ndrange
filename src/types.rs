use ndarray::{Dimension, IntoDimension, Ix1, Ix2, Ix3, Ix4, Ix5};


#[derive(Clone, Debug)]
pub struct NdRange<D> {
    pub start: D,
    pub end: D,
}


impl<D> NdRange<D> where D: Dimension {
    pub fn new<I>(start: I, end: I) -> Self where I: IntoDimension<Dim=D> {
        let start = start.into_dimension();
        let end = end.into_dimension();
        NdRange { start: start.clone(), end: end }
    }

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
    use super::*;

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