use ndarray::{Dimension};

use types::NdRange;

#[derive(Clone)]
pub struct IterState<D> {
    pub range: NdRange<D>,
    pub head: D,
    pub tail: D,
    pub len: usize
}

impl<D> IterState<D> where D: Dimension {
    pub fn increment(&mut self) {
        let start_end_iter = self.range.start.slice().iter().zip(self.range.end.slice().iter());
        let iter = self.head.slice_mut().iter_mut().zip(start_end_iter);

        for (c, (s, e)) in iter.rev() {
            *c += 1;
            if *c < *e {
                break;
            }
            *c = s.clone()
        }
        self.len -= 1;
    }

    pub fn increment_back(&mut self) {
        let start_end_iter = self.range.start.slice().iter().zip(self.range.end.slice().iter());
        let iter = self.tail.slice_mut().iter_mut().zip(start_end_iter);

        for (c, (s, e)) in iter.rev() {
            *c -= 1;
            if *c >= *s {
                break;
            }
            *c = e.clone() - 1
        }
        self.len -= 1;
    }

    pub fn from_range(range: NdRange<D>) -> Self {
        let mut tail = range.start.clone();
        tail[0] = range.end[0];

        IterState {
            head: range.start.clone(),
            tail: tail,
            len: range.len(),
            range: range,
        }
    }

    pub fn at(&self, index: usize) -> D {
        let mut mid = self.head.clone();
        let mut index = index;

        for i in (0..mid.ndim()).rev() {
            let size = self.range.end[i] - self.range.start[i];
            let offset = self.head[i] - self.range.start[i] + index;

            mid[i] = (offset % size) + self.range.start[i];
            index = offset / size;
        }

        mid
    }

    pub fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.at(index);
        let left = IterState {
            range: self.range.clone(),
            head: self.head.clone(),
            tail: mid.clone(),
            len: index
        };
        let right = IterState {
            range: self.range,
            head: mid,
            tail: self.tail,
            len: self.len - index
        };
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IntoDimension;

    use super::*;
    use types::{Range3};
    use iter::sequential::into_iter_from_state;

    #[test]
    fn at() {
        let mut state = IterState::from_range(Range3::new((2, 2, 1), (5, 3, 5)));
        assert_eq!(state.at(0).into_pattern(), (2, 2, 1));
        assert_eq!(state.at(1).into_pattern(), (2, 2, 2));
        assert_eq!(state.at(2).into_pattern(), (2, 2, 3));
        assert_eq!(state.at(3).into_pattern(), (2, 2, 4));
        assert_eq!(state.at(4).into_pattern(), (3, 2, 1));
        assert_eq!(state.at(5).into_pattern(), (3, 2, 2));
        assert_eq!(state.at(6).into_pattern(), (3, 2, 3));
        assert_eq!(state.at(7).into_pattern(), (3, 2, 4));
        assert_eq!(state.at(8).into_pattern(), (4, 2, 1));
        assert_eq!(state.at(9).into_pattern(), (4, 2, 2));
        assert_eq!(state.at(10).into_pattern(), (4, 2, 3));
        assert_eq!(state.at(11).into_pattern(), (4, 2, 4));

        state.head = (3, 2, 2).into_dimension();
        state.tail = (4, 2, 4).into_dimension();
        state.len = 6;

        assert_eq!(state.at(0).into_pattern(), (3, 2, 2));
        assert_eq!(state.at(1).into_pattern(), (3, 2, 3));
        assert_eq!(state.at(2).into_pattern(), (3, 2, 4));
        assert_eq!(state.at(3).into_pattern(), (4, 2, 1));
        assert_eq!(state.at(4).into_pattern(), (4, 2, 2));
        assert_eq!(state.at(5).into_pattern(), (4, 2, 3));
    }

    fn get_real_length<D>(state: &IterState<D>) -> usize where D: Dimension {
        into_iter_from_state(state.clone()).fold(0, |acc, _| acc + 1)
    }

    #[test]
    fn split_at() {
        let state = IterState::from_range(Range3::new((0, 3, 8), (4, 5, 12)));
        assert_eq!(state.len, 32);
        assert_eq!(state.head.into_pattern(), (0, 3, 8));
        assert_eq!(state.tail.into_pattern(), (4, 3, 8));
        assert_eq!(get_real_length(&state), 32);

        let (left, right) = state.split_at(3);
        assert_eq!(left.len, 3);
        assert_eq!(left.head.into_pattern(), (0, 3, 8));
        assert_eq!(left.tail.into_pattern(), (0, 3, 11));
        assert_eq!(get_real_length(&left), 3);
        assert_eq!(right.len, 29);
        assert_eq!(right.head.into_pattern(), (0, 3, 11));
        assert_eq!(right.tail.into_pattern(), (4, 3, 8));
        assert_eq!(get_real_length(&right), 29);

        let (left, right) = right.split_at(15);
        assert_eq!(left.len, 15);
        assert_eq!(left.head.into_pattern(), (0, 3, 11));
        assert_eq!(left.tail.into_pattern(), (2, 3, 10));
        assert_eq!(get_real_length(&left), 15);
        assert_eq!(right.len, 14);
        assert_eq!(right.head.into_pattern(), (2, 3, 10));
        assert_eq!(right.tail.into_pattern(), (4, 3, 8));
        assert_eq!(get_real_length(&right), 14);
    }
}