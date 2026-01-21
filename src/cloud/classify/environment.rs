use crate::utils::boxes::Box3D;
use crate::{cloud::CldBud, config::fixif};

use crate::utils::random::select_some;
use rand::{RngExt, rng};

/// 使用比例过滤和法向量检验的方法提取地面点
/// 实现三阶段过滤：
/// 1. 第一过滤：阈值抽样法向量检验
/// 2. 第二过滤：参数截取默认置信
/// 3. 第三过滤：非置信区间，采用通过与置信区间随机抽样法向量检验的方法判断该点是否应当被扩展
pub fn pick_ground(cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>) {
    if cloud.is_empty() {
        return (0, vec![]);
    }

    // 首先按照z坐标排序
    cloud.sort_unstable_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

    // 获取配置参数
    let config = fixif();
    let ground_filter_threshold = config.ground_filter_threshold;
    let cross_product_patience = config.ground_cross_product_patience;
    let sample_test_count = config.ground_sample_test_count;

    // 1. 按照最大值和最小值的一定比例过滤点 (第一过滤)
    let z_min = cloud[0][2];
    let z_max = cloud[cloud.len() - 1][2];
    let z_threshold = z_min + (z_max - z_min) * ground_filter_threshold;

    let mut filtered = 0;
    for elem in &*cloud {
        if elem[2] > z_threshold {
            break;
        }
        filtered += 1;
    }

    if filtered < 3 {
        return (0, vec![]);
    }

    // 使用迭代器和all方法检查所有随机抽样是否都满足要求
    if !(0..sample_test_count).all(|_| {
        let candidate_indices = random_sampling(&cloud, 0, filtered, 3);
        // 检查抽样点数是否足够
        candidate_indices.len() >= 3
    }) {
        return (0, vec![]);
    }

    // 2. 参数截取默认置信 (第二过滤)
    // 取过滤区间的前30%作为置信区域
    let trusted_end = (filtered as f32 * 0.3) as usize;
    let trusted_end = trusted_end.max(3).min(filtered); // 至少需要3个点，但不超过filtered

    // 直接使用切片而不是复制点
    let trusted_points = &cloud[..trusted_end];

    // 创建一个布尔数组来标记哪些点已经被添加
    let mut added = vec![false; cloud.len()];
    for i in 0..trusted_end {
        added[i] = true;
    }

    // 3. 非置信区间，采用通过与置信区间随机抽样法向量检验的方法判断该点是否应当被扩展 (第三过滤)
    let mut last_added_index = trusted_end;
    for i in trusted_end..cloud.len() {
        // 跳过已经被加入到地面点集合的点
        if added[i] {
            continue;
        }

        if can_add_by_normal_check_optimized(
            trusted_points,
            cloud[i],
            cross_product_patience,
            sample_test_count,
        ) {
            added[i] = true;
            last_added_index = i + 1;
        }
    }

    // 构造地面包围盒并直接返回
    let mut ground_box = Box3D::empty_box();
    // 只使用added数组中标记为true的点来构建包围盒
    let ground_points: Vec<[f32; 3]> = cloud
        .iter()
        .enumerate()
        .filter(|(i, _)| added[*i])
        .map(|(_, point)| *point)
        .collect();

    ground_box.cloud2box(&ground_points);

    // 创建表示地面的CldBud对象
    let ground_cld_bud = CldBud::new(
        ground_box,
        0,                    // 地面类别ID
        "ground".to_string(), // 地面类别名称
        1.0,                  // 置信度
    );

    (last_added_index, vec![ground_cld_bud])
}

/// 使用RANSAC算法检测墙面点，利用墙面法向量与Z轴垂直的特性加速检测
pub fn pick_wall(cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>) {
    if cloud.is_empty() {
        return (0, vec![]);
    }

    let mut wall_cld_buds = Vec::new();

    // 获取配置参数
    let ransac_iterations = 100; // RANSAC迭代次数
    let distance_threshold = 0.2; // 进一步放宽点到平面的距离阈值
    let min_wall_points = 30; // 进一步降低构成墙面所需的最少点数
    let vertical_tolerance = 0.3; // 进一步放宽法向量与Z轴垂直的容忍度

    // 为了不改变原始点云顺序，我们创建一个索引向量进行操作
    let indices: Vec<usize> = (0..cloud.len()).collect();
    let mut used: Vec<bool> = vec![false; cloud.len()]; // 标记已被使用的点
    // 可能存在多个墙面，所以我们循环检测直到找不到更多墙面
    loop {
        // let mut best_plane: Option<([f32; 3], f32)> = None; // (法向量, 距离)
        let mut best_support_count = 0;
        let mut best_inliers = Vec::new();

        // 收集未被使用的点的索引
        let unused_indices: Vec<usize> = indices.iter().filter(|&&i| !used[i]).copied().collect();

        // 如果剩余点太少，停止检测
        if unused_indices.len() < min_wall_points / 2 {
            break;
        }

        // RANSAC迭代
        for _ in 0..ransac_iterations {
            // 使用select_some函数随机选择3个不重复的点
            let sampled_indices_raw = select_some(0, unused_indices.len(), 3);
            if sampled_indices_raw.len() < 3 {
                continue;
            }

            let sampled_indices: Vec<usize> = sampled_indices_raw
                .iter()
                .map(|&i| unused_indices[i])
                .collect();

            // 从三个点计算平面方程
            let p1 = cloud[sampled_indices[0]];
            let p2 = cloud[sampled_indices[1]];
            let p3 = cloud[sampled_indices[2]];

            // 计算两个边向量
            let edge1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let edge2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

            // 计算法向量（通过叉积）
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];

            // 检查法向量是否有效
            let normal_magnitude =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if normal_magnitude < 1e-6 {
                continue;
            }

            // 归一化法向量
            let normalized_normal = [
                normal[0] / normal_magnitude,
                normal[1] / normal_magnitude,
                normal[2] / normal_magnitude,
            ];

            // 利用墙面法向量应与Z轴垂直的先验知识进行快速过滤
            // Z轴向量是(0, 0, 1)，墙面法向量与其点积应该接近0
            let dot_with_z = normalized_normal[2].abs(); // 与Z轴的点积绝对值
            if dot_with_z > vertical_tolerance {
                // 如果法向量与Z轴不够垂直，则不是墙面，跳过
                continue;
            }

            // 计算平面到原点的距离
            let distance = normalized_normal[0] * p1[0]
                + normalized_normal[1] * p1[1]
                + normalized_normal[2] * p1[2];

            // 统计内点数量（距离平面小于阈值的点）
            let mut inliers = Vec::new();
            for &idx in &unused_indices {
                let point = cloud[idx];
                let dist = (normalized_normal[0] * point[0]
                    + normalized_normal[1] * point[1]
                    + normalized_normal[2] * point[2]
                    - distance)
                    .abs();

                if dist < distance_threshold {
                    inliers.push(idx);
                }
            }

            // 更新最佳平面模型
            if inliers.len() > best_support_count && inliers.len() >= min_wall_points / 2 {
                best_support_count = inliers.len();
                // best_plane = Some((normalized_normal, distance));
                best_inliers = inliers;
            }
        }

        // 如果没有找到合适的平面，尝试放宽条件再检测一次
        if best_support_count < min_wall_points && min_wall_points > 15 {
            // 临时放宽条件再次尝试
            let relaxed_min_wall_points = min_wall_points / 3;
            let relaxed_distance_threshold = distance_threshold * 2.0;

            // 再次尝试寻找满足放宽条件的墙面
            for _ in 0..(ransac_iterations * 2) {
                // 增加迭代次数
                // 使用select_some函数随机选择3个不重复的点
                let sampled_indices_raw = select_some(0, unused_indices.len(), 3);
                if sampled_indices_raw.len() < 3 {
                    continue;
                }

                let sampled_indices: Vec<usize> = sampled_indices_raw
                    .iter()
                    .map(|&i| unused_indices[i])
                    .collect();

                let p1 = cloud[sampled_indices[0]];
                let p2 = cloud[sampled_indices[1]];
                let p3 = cloud[sampled_indices[2]];

                let edge1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
                let edge2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

                let normal = [
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0],
                ];

                let normal_magnitude =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if normal_magnitude < 1e-6 {
                    continue;
                }

                let normalized_normal = [
                    normal[0] / normal_magnitude,
                    normal[1] / normal_magnitude,
                    normal[2] / normal_magnitude,
                ];

                let dot_with_z = normalized_normal[2].abs();
                if dot_with_z > vertical_tolerance * 2.0 {
                    // 进一步放宽垂直度要求
                    continue;
                }

                let distance = normalized_normal[0] * p1[0]
                    + normalized_normal[1] * p1[1]
                    + normalized_normal[2] * p1[2];

                let mut inliers = Vec::new();
                for &idx in &unused_indices {
                    let point = cloud[idx];
                    let dist = (normalized_normal[0] * point[0]
                        + normalized_normal[1] * point[1]
                        + normalized_normal[2] * point[2]
                        - distance)
                        .abs();

                    if dist < relaxed_distance_threshold {
                        inliers.push(idx);
                    }
                }

                if inliers.len() > best_support_count && inliers.len() >= relaxed_min_wall_points {
                    best_support_count = inliers.len();
                    // best_plane = Some((normalized_normal, distance));
                    best_inliers = inliers;
                }
            }
        }

        // 如果还是没有找到合适的平面，结束检测
        if best_support_count < min_wall_points / 3 {
            break;
        }

        // 根据最佳平面提取墙面点
        let wall_points: Vec<[f32; 3]> = best_inliers.iter().map(|&i| cloud[i]).collect();

        // 标记这些点为已使用
        for &idx in &best_inliers {
            used[idx] = true;
        }

        // 创建墙面包围盒
        let mut wall_box = Box3D::empty_box();
        wall_box.cloud2box(&wall_points);

        // 创建墙面CldBud对象
        let wall_cld_bud = CldBud::new(
            wall_box,
            2,                                       // 墙面类别ID
            format!("wall_{}", wall_cld_buds.len()), // 墙面类别名称
            if best_support_count >= min_wall_points {
                0.9
            } else if best_support_count >= min_wall_points / 2 {
                0.7
            } else {
                0.5
            }, // 根据点数确定置信度
        );

        wall_cld_buds.push(wall_cld_bud);
    }

    // 添加一个简单的启发式检测方法作为备选
    // 如果RANSAC方法没有找到墙面，则尝试基于点密度的简单方法
    if wall_cld_buds.is_empty() {
        // 寻找密集的垂直平面区域
        let unused_indices: Vec<usize> = indices.iter().filter(|&&i| !used[i]).copied().collect();

        if !unused_indices.is_empty() {
            // 按X坐标排序，寻找垂直Y-Z平面
            let mut sorted_by_x = unused_indices.clone();
            sorted_by_x.sort_by(|&a, &b| cloud[a][0].partial_cmp(&cloud[b][0]).unwrap());

            // 查找X方向上的密集区域
            if sorted_by_x.len() > 20 {
                let x_min = cloud[sorted_by_x[0]][0];
                let x_max = cloud[sorted_by_x[sorted_by_x.len() - 1]][0];
                let x_range = x_max - x_min;

                if x_range > 0.1 {
                    // 确保有足够的范围
                    // 在X方向上分割为几个区域，寻找点密度高的区域
                    let num_bins = 10;
                    let bin_width = x_range / num_bins as f32;
                    let mut bins = vec![Vec::new(); num_bins];

                    for &idx in &sorted_by_x {
                        let bin_idx = ((cloud[idx][0] - x_min) / bin_width) as usize;
                        let bin_idx = bin_idx.min(num_bins - 1);
                        bins[bin_idx].push(idx);
                    }

                    // 寻找点数最多的几个bins
                    for (bin_idx, bin) in bins.iter().enumerate() {
                        if bin.len() >= 20 {
                            // 如果某个bin中有足够多的点
                            let wall_points: Vec<[f32; 3]> =
                                bin.iter().map(|&i| cloud[i]).collect();

                            let mut wall_box = Box3D::empty_box();
                            wall_box.cloud2box(&wall_points);

                            let wall_cld_bud = CldBud::new(
                                wall_box,
                                2,
                                format!("wall_heuristic_{}", bin_idx),
                                0.4, // 启发式方法的置信度较低
                            );

                            wall_cld_buds.push(wall_cld_bud);
                        }
                    }
                }
            }
        }
    }

    // 我们不移动任何点，所以slice_index始终为0
    (0, wall_cld_buds)
}

pub fn single_pick_ground(cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>) {
    let ixi = fixif();
    cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

    // 原始方法：使用固定阈值
    // let filter = ixi.ground_filter_threshold;
    // let result = filter * cloud.len() as f32;
    // return (result as usize, Vec::new())

    // 新方法：基于均值和最小值的自适应筛选
    let filter = ixi.ground_filter_threshold;
    let initial_count = (filter * cloud.len() as f32) as usize;

    if initial_count >= cloud.len() || initial_count == 0 {
        return (initial_count, Vec::new());
    }

    // 计算初始阈值区间内的均值和最小值
    let mut sum = 0.0f32;
    let min_value = cloud[0][2];
    for i in 0..initial_count {
        sum += cloud[i][2];
    }
    let mean_value = sum / initial_count as f32;

    // 使用delta值确定终止条件（这里使用配置中的cross_product_patience作为delta）
    let delta = mean_value - min_value;
    let threshold = mean_value + delta;

    // 向后遍历直到数值超过均值+delta时停止
    let mut final_count = initial_count;
    for i in (initial_count / 2)..cloud.len() {
        if cloud[i][2] > threshold {
            break;
        }
        final_count = i + 1;
    }

    (final_count, Vec::new())
}

/// 随机抽样函数
fn random_sampling(cloud: &[[f32; 3]], start: usize, end: usize, count: usize) -> Vec<[f32; 3]> {
    let indexes = select_some(start, end, count);
    indexes.iter().map(|&i| cloud[i]).collect()
}

/// 计算三点构成平面的法向量与参考法向量的一致性
pub fn aboat_normal(plane: [[f32; 3]; 3], reference_normal: [f32; 3]) -> f32 {
    // 计算两个边向量
    let edge1 = [
        plane[1][0] - plane[0][0],
        plane[1][1] - plane[0][1],
        plane[1][2] - plane[0][2],
    ];

    let edge2 = [
        plane[2][0] - plane[0][0],
        plane[2][1] - plane[0][1],
        plane[2][2] - plane[0][2],
    ];

    // 计算叉积得到平面法向量
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];

    // 归一化法向量
    let magnitude = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if magnitude == 0.0 {
        return 0.0;
    }

    let normalized_normal = [
        normal[0] / magnitude,
        normal[1] / magnitude,
        normal[2] / magnitude,
    ];

    // 计算与参考法向量的点积（一致性度量）
    normalized_normal[0] * reference_normal[0]
        + normalized_normal[1] * reference_normal[1]
        + normalized_normal[2] * reference_normal[2]
}

/// 检查点集是否满足平面分布要求
pub fn satisfy_plane(points: &[[f32; 3]], tolerance: f32, sample_test_count: usize) -> bool {
    if points.len() < 3 {
        return false;
    }

    let config = fixif();
    let reference_normal = config.default_ground_vector;

    // 多次抽样检查平面一致性
    for _ in 0..sample_test_count {
        let sample_indices = select_some(0, points.len(), 3);
        if sample_indices.len() >= 3 {
            let samples: Vec<[f32; 3]> = sample_indices.iter().map(|&i| points[i]).collect();
            // 如果任何一个样本不满足条件，则认为不满足平面要求
            if aboat_normal([samples[0], samples[1], samples[2]], reference_normal) <= tolerance {
                return false;
            }
        }
    }

    true
}

/// 检查新点是否可以通过法向量检验加入到现有点集中
pub fn can_add_by_normal_check(
    trusted_points: &[[f32; 3]],
    new_point: [f32; 3],
    tolerance: f32,
    sample_test_count: usize,
) -> bool {
    if trusted_points.len() < 3 {
        return false;
    }

    // 多次抽样检查新点是否与可信区域的法向量一致
    for _ in 0..sample_test_count {
        // 从可信区域随机选取3个点
        let sample_indices = select_some(0, trusted_points.len(), 3);
        if sample_indices.len() >= 3 {
            let samples: Vec<[f32; 3]> =
                sample_indices.iter().map(|&i| trusted_points[i]).collect();

            // 计算可信区域样本的法向量
            let trusted_normal = calculate_normal([samples[0], samples[1], samples[2]]);

            // 从可信区域再选取2个点和新点组成三角形
            let sample_indices2 = select_some(0, trusted_points.len(), 2);
            if sample_indices2.len() >= 2 {
                let samples2: Vec<[f32; 3]> =
                    sample_indices2.iter().map(|&i| trusted_points[i]).collect();
                let test_normal = calculate_normal([samples2[0], samples2[1], new_point]);

                // 比较两个法向量的夹角
                let dot_product = trusted_normal[0] * test_normal[0]
                    + trusted_normal[1] * test_normal[1]
                    + trusted_normal[2] * test_normal[2];

                // 如果夹角过大，则不添加该点
                if dot_product.abs() < tolerance {
                    return false;
                }
            }
        }
    }

    true
}

/// 计算三点构成平面的法向量
fn calculate_normal(plane: [[f32; 3]; 3]) -> [f32; 3] {
    // 计算两个边向量
    let edge1 = [
        plane[1][0] - plane[0][0],
        plane[1][1] - plane[0][1],
        plane[1][2] - plane[0][2],
    ];

    let edge2 = [
        plane[2][0] - plane[0][0],
        plane[2][1] - plane[0][1],
        plane[2][2] - plane[0][2],
    ];

    // 计算叉积得到平面法向量
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];

    // 归一化法向量
    let magnitude = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if magnitude == 0.0 {
        return [0.0, 0.0, 1.0]; // 默认法向量
    }

    [
        normal[0] / magnitude,
        normal[1] / magnitude,
        normal[2] / magnitude,
    ]
}

/// 检查新点是否可以通过法向量检验加入到现有点集中
fn can_add_by_normal_check_optimized(
    trusted_points: &[[f32; 3]],
    new_point: [f32; 3],
    tolerance: f32,
    sample_test_count: usize,
) -> bool {
    if trusted_points.len() < 3 {
        return false;
    }

    let mut rng = rng();

    // 多次抽样检查新点是否与可信区域的法向量一致
    for _ in 0..sample_test_count {
        // 从可信区域随机选取3个点
        let i1 = rng.random_range(0..trusted_points.len());
        let i2 = rng.random_range(0..trusted_points.len());
        let i3 = rng.random_range(0..trusted_points.len());

        // 确保不选择相同的点
        if i1 == i2 || i2 == i3 || i1 == i3 {
            continue;
        }

        let p1 = trusted_points[i1];
        let p2 = trusted_points[i2];
        let p3 = trusted_points[i3];

        // 计算可信区域样本的法向量
        let trusted_normal = calculate_normal([p1, p2, p3]);

        // 再选取两个不同的点
        let i4 = rng.random_range(0..trusted_points.len());
        let i5 = rng.random_range(0..trusted_points.len());

        if i4 == i5 || i4 == i1 || i4 == i2 || i4 == i3 || i5 == i1 || i5 == i2 || i5 == i3 {
            continue;
        }

        let p4 = trusted_points[i4];
        let p5 = trusted_points[i5];

        // 计算包含新点的法向量
        let test_normal = calculate_normal([p4, p5, new_point]);

        // 比较两个法向量的夹角
        let dot_product = trusted_normal[0] * test_normal[0]
            + trusted_normal[1] * test_normal[1]
            + trusted_normal[2] * test_normal[2];

        // 如果夹角过大，则不添加该点
        if dot_product.abs() < tolerance {
            return false;
        }
    }

    true
}

/// 检查新点是否可以加入到现有平面中
pub fn can_add_to_plane(
    existing_points: &[[f32; 3]],
    new_point: [f32; 3],
    tolerance: f32,
    sample_test_count: usize,
) -> bool {
    // 创建包含新点的临时点集
    let mut temp_points = existing_points.to_vec();
    temp_points.push(new_point);

    // 检查加入新点后是否仍满足平面要求
    satisfy_plane(&temp_points, tolerance, sample_test_count)
}
