use rand::{rng, seq::IndexedRandom};

pub fn select_some(start: usize, end: usize, count: usize) -> Vec<usize> {
    let ans = (start..end).collect::<Vec<_>>();  // 修改为不包含 end 的范围
    let mut rng = rng();
   ans.sample(&mut rng, count).cloned().collect::<Vec<_>>()
}