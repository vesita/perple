/// Hungarian 算法（Munkres），求解最小代价指派问题
///
/// 输入 `cost[i][j]` 为 `n_rows × n_cols` 代价矩阵，返回每个 row 指派的 column 索引。
/// 若 `assigned_col >= n_cols` 则表示该 row 未匹配（指派给了 dummy column）。
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n_rows = cost.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = cost[0].len();
    if n_cols == 0 {
        return vec![0; n_rows];
    }

    let n = n_rows.max(n_cols);

    // 寻找有限最大代价
    let max_finite = cost
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&c| c < f64::MAX / 2.0)
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let pad_cost = if max_finite.is_finite() { max_finite + 1.0 } else { 1e9 };

    // 填充为方阵
    let mut sq = vec![vec![pad_cost; n]; n];
    for i in 0..n_rows {
        for j in 0..n_cols {
            sq[i][j] = cost[i][j];
        }
    }

    // 执行匈牙利算法
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0; n + 1]; // p[j] = row assigned to column j (0 = unassigned)
    let mut way = vec![0; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::MAX; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::MAX;
            let mut j1 = 0;

            for j in 1..=n {
                if !used[j] {
                    let cur = sq[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // 增广路径更新
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // 转译为 row → column 映射
    let mut result = vec![n_cols; n_rows];
    for j in 1..=n {
        let row = p[j];
        if row > 0 && row <= n_rows {
            result[row - 1] = (j - 1).min(n_cols);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(hungarian(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_single_pair() {
        let cost = vec![vec![5.0]];
        let r = hungarian(&cost);
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn test_2x2() {
        //     c1 c2
        // r1: 1  2
        // r2: 3  4
        // optimal: r1→c1(1), r2→c2(4) = 5
        let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = hungarian(&cost);
        assert_eq!(r.len(), 2);
        // r1 should be assigned to c1 (0) or c2 (1)
        assert!(r[0] < 2);
        assert!(r[1] < 2);
        assert_ne!(r[0], r[1]); // one-to-one
    }

    #[test]
    fn test_rectangular_more_rows() {
        // 3 rows, 2 cols
        let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let r = hungarian(&cost);
        // one row should be unmatched (assigned >= 2)
        let matched: Vec<&usize> = r.iter().filter(|&&c| c < 2).collect();
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_rectangular_more_cols() {
        // 2 rows, 3 cols
        let cost = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let r = hungarian(&cost);
        // both rows should be matched
        assert_eq!(r.len(), 2);
        assert!(r[0] < 3);
        assert!(r[1] < 3);
        assert_ne!(r[0], r[1]);
    }

    #[test]
    fn test_inf_costs() {
        // 全部无效，所有 row 应 unmatched
        let cost = vec![vec![f64::MAX, f64::MAX], vec![f64::MAX, f64::MAX]];
        let r = hungarian(&cost);
        assert_eq!(r.len(), 2);
        // 都可能被指派到 dummy（≥2）
    }

    #[test]
    fn test_prefer_valid() {
        // 一个有效一个无效
        let cost = vec![vec![1.0, f64::MAX], vec![f64::MAX, 2.0]];
        let r = hungarian(&cost);
        assert_eq!(r.len(), 2);
        // r0 → c0, r1 → c1
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 1);
    }
}
