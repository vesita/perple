// todo
// 索引数学组合枚举

pub fn combine_all(all: usize, select: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut path = Vec::new();
    pick(all, select, &mut path, &mut result);
    result
}

fn pick(target: usize, take: usize, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    let select = take - path.len();
    if select == 0 {
        result.push(path.clone());
        return;
    }
    if target > select {
        pick(target - 1, take, path, result);
    }
    path.push(target);
    pick(target - 1, take, path, result);
    path.pop();
}

// impl Solution {
//     pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
//         let mut ans = Vec::new();
//         let mut path = Vec::new();
//         fn pick(all: usize, start: usize, select: usize, path: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
//             if select == 0 {
//                 ans.push(path.clone());
//                 return;
//             }
//             for now in start..=(all - select + 1) {
//                 path.push(now as i32);
//                 pick(all, now + 1, select - 1, path, ans);
//                 path.pop();
//             }
//         }

//         pick(n as usize, 1, k as usize, &mut path, &mut ans);
//         ans
//     }
// }
