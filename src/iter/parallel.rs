use rayon::prelude::*;
use rayon::iter::internal::*;
use ndarray::Dimension;

use types::NdRange;
use super::state::IterState;
use super::sequential::{IntoIter, into_iter_from_state};

struct IterProducer<D> {
    state: IterState<D>
}

pub struct IntoParIter<D> {
    state: IterState<D>
}


impl<D> Producer for IterProducer<D> where D: Dimension {
    type Item = <NdRange<D> as IntoIterator>::Item;
    type IntoIter = IntoIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        into_iter_from_state(self.state)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.state.split_at(index);

        (IterProducer { state: left },
         IterProducer { state: right })
    }
}

impl<D> ParallelIterator for IntoParIter<D> where D: Dimension, D::Pattern: Send {
    type Item = D::Pattern;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: UnindexedConsumer<Self::Item>
    {
        bridge(self, consumer)
    }

    fn opt_len(&mut self) -> Option<usize> {
        Some(self.state.len)
    }
}

impl<D> BoundedParallelIterator for IntoParIter<D> where D: Dimension, D::Pattern: Send {
    fn upper_bound(&mut self) -> usize {
        self.state.len
    }

    fn drive<'c, C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }
}

impl<D> ExactParallelIterator for IntoParIter<D> where D: Dimension, D::Pattern: Send {
    fn len(&mut self) -> usize {
        self.state.len
    }
}

impl<D> IndexedParallelIterator for IntoParIter<D> where D: Dimension, D::Pattern: Send {
    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where CB: ProducerCallback<Self::Item>
    {
        callback.callback(IterProducer { state: self.state })
    }
}


impl<D> IntoParallelIterator for NdRange<D> where D: Dimension, D::Pattern: Send {
    type Item = D::Pattern;
    type Iter = IntoParIter<D>;

    fn into_par_iter(self) -> Self::Iter {
        IntoParIter {
            state: IterState::from_range(self)
        }
    }
}


#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use types::{Range3};


    #[test]
    fn into_par_iter() {
        let expected = vec![(1, 2, 3), (1, 2, 4), (1, 2, 5),
                            (1, 3, 3), (1, 3, 4), (1, 3, 5),
                            (2, 2, 3), (2, 2, 4), (2, 2, 5),
                            (2, 3, 3), (2, 3, 4), (2, 3, 5),
                            (3, 2, 3), (3, 2, 4), (3, 2, 5),
                            (3, 3, 3), (3, 3, 4), (3, 3, 5),
                            (4, 2, 3), (4, 2, 4), (4, 2, 5),
                            (4, 3, 3), (4, 3, 4), (4, 3, 5)];
        let actual = Range3::new((1, 2, 3), (5, 4, 6)).into_par_iter().collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }
}