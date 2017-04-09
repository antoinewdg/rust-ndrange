use ndarray::{Dimension};

use types::NdRange;
use super::state::IterState;


pub struct IntoIter<D> {
    state: IterState<D>
}

pub fn into_iter_from_state<D>(state: IterState<D>) -> IntoIter<D> {
    IntoIter { state: state }
}

impl<D> IntoIterator for NdRange<D> where D: Dimension {
    type Item = D::Pattern;
    type IntoIter = IntoIter<D>;

    fn into_iter(self) -> IntoIter<D> {
        IntoIter {
            state: IterState::from_range(self)
        }
    }
}

impl<D> Iterator for IntoIter<D> where D: Dimension {
    type Item = D::Pattern;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.len == 0 { return None; }

        let value = self.state.head.clone().into_pattern();

        self.state.increment();
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.state.len;
        (len, Some(len))
    }
}


impl<D> ExactSizeIterator for IntoIter<D> where D: Dimension {
    fn len(&self) -> usize {
        self.state.len
    }
}

impl<D> DoubleEndedIterator for IntoIter<D> where D: Dimension {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.state.len == 0 { return None; }
        self.state.increment_back();
        Some(self.state.tail.clone().into_pattern())
    }
}

#[cfg(test)]
mod tests {
    use ::{Range2, Range3};


    #[test]
    fn into_iter() {
        let expected = vec![(1, 2, 3), (1, 2, 4), (1, 2, 5),
                            (1, 3, 3), (1, 3, 4), (1, 3, 5),
                            (2, 2, 3), (2, 2, 4), (2, 2, 5),
                            (2, 3, 3), (2, 3, 4), (2, 3, 5),
                            (3, 2, 3), (3, 2, 4), (3, 2, 5),
                            (3, 3, 3), (3, 3, 4), (3, 3, 5),
                            (4, 2, 3), (4, 2, 4), (4, 2, 5),
                            (4, 3, 3), (4, 3, 4), (4, 3, 5)];
        let actual = Range3::new((1, 2, 3), (5, 4, 6)).into_iter().collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn into_iter_rev() {
        let expected = vec![(1, 2, 3), (1, 2, 4), (1, 2, 5),
                            (1, 3, 3), (1, 3, 4), (1, 3, 5),
                            (2, 2, 3), (2, 2, 4), (2, 2, 5),
                            (2, 3, 3), (2, 3, 4), (2, 3, 5),
                            (3, 2, 3), (3, 2, 4), (3, 2, 5),
                            (3, 3, 3), (3, 3, 4), (3, 3, 5),
                            (4, 2, 3), (4, 2, 4), (4, 2, 5),
                            (4, 3, 3), (4, 3, 4), (4, 3, 5)];
        let expected = expected.into_iter().rev().collect::<Vec<_>>();
        let actual = Range3::new((1, 2, 3), (5, 4, 6)).into_iter().rev().collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn len() {
        let mut iter = Range2::new((7, 1), (9, 4)).into_iter();

        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next(), Some((7, 1)));

        assert_eq!(iter.len(), 5);
        assert_eq!(iter.next(), Some((7, 2)));

        assert_eq!(iter.len(), 4);
        assert_eq!(iter.next(), Some((7, 3)));

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some((8, 1)));

        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some((8, 2)));

        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some((8, 3)));

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn len_reverse() {
        let mut iter = Range2::new((7, 1), (9, 4)).into_iter();

        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next_back(), Some((8, 3)));

        assert_eq!(iter.len(), 5);
        assert_eq!(iter.next_back(), Some((8, 2)));

        assert_eq!(iter.len(), 4);
        assert_eq!(iter.next(), Some((7, 1)));

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next_back(), Some((8, 1)));

        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some((7, 2)));

        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some((7, 3)));

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
    }
}
