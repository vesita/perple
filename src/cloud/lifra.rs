use std::ops::Index;

use pcd_rs::DynReader;

use crate::config::*;

#[derive(Clone)]
pub struct Lifra {
    points: Vec<[f32; 3]>,
    count: usize,
}

impl Lifra {

    pub fn new() -> Self {
        Lifra {
            points: Vec::with_capacity(POINTS_CAPACITY),
            count: 0,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Lifra {
            points: Vec::with_capacity(capacity),
            count: 0,
        }
    }
    
    /// 从DynReader直接构造Lifra实例
    pub fn init<R>(reader: &mut DynReader<R>) -> Self 
    where 
        R: std::io::BufRead,
    {
        let mut lifra = Lifra::new();
        lifra.update(reader);
        lifra
    }

    pub fn update<R>(&mut self, reader: &mut DynReader<R>) 
    where 
        R: std::io::BufRead,
    {
        self.count = 0;
        while let Some(result) = reader.next() {
            if let Ok(point) = result {
                if let Some(now) = point.to_xyz() {
                    if !self.reject(&now) {
                        self.push(now);
                    }
                }
            }
        }
    }

    /// 拒绝无效点
    /// 
    /// 过滤掉包含NaN或无穷大的点
    pub fn reject(&self, point: &[f32; 3]) -> bool {
        point.iter().any(|&x| x.is_nan() || x.is_infinite())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// 将点添加到点云中
    /// 
    /// 如果点云已满，则不会添加新点
    pub fn push(&mut self, point: [f32; 3]) {
        if self.count < POINTS_CAPACITY {
            self.points.push(point);
            self.count += 1;
        }
    }
    
    /// 从另一个Lifra实例更新数据
    pub fn update_from_lifra(&mut self, other: &Lifra) {
        self.count = 0;
        for point in other.iter().take(other.len()) {
            self.push(*point);
        }
    }

    /// 提供只读引用迭代器
    pub fn iter(&self) -> std::slice::Iter<'_, [f32; 3]> {
        self.points.iter()
    }
    
    /// 提供可变引用迭代器
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, [f32; 3]> {
        self.points.iter_mut()
    }

    /// 获取点云数据的只读引用
    pub fn points(&self) -> &Vec<[f32; 3]> {
        &self.points
    }
    
    /// 使用给定的点云数据创建新的Lifra实例
    pub fn from_points(points: Vec<[f32; 3]>) -> Self {
        let count = points.len().min(POINTS_CAPACITY);
        let mut points_vec = Vec::with_capacity(POINTS_CAPACITY);
        points_vec.extend_from_slice(&points[..count]);
        Lifra {
            points: points_vec,
            count,
        }
    }

    /// 清空点云数据
    pub fn clear(&mut self) {
        self.points.clear();
        self.count = 0;
    }
}

// 实现IntoIterator，支持所有权转移的迭代
impl IntoIterator for Lifra {
    type Item = [f32; 3];
    type IntoIter = std::vec::IntoIter<[f32; 3]>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter()
    }
}

// 实现引用的IntoIterator
impl<'a> IntoIterator for &'a Lifra {
    type Item = &'a [f32; 3];
    type IntoIter = std::slice::Iter<'a, [f32; 3]>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

// 实现可变引用的IntoIterator
impl<'a> IntoIterator for &'a mut Lifra {
    type Item = &'a mut [f32; 3];
    type IntoIter = std::slice::IterMut<'a, [f32; 3]>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter_mut()
    }
}

impl Index<usize> for Lifra {
    type Output = [f32; 3];

    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

// 实现Default trait
impl Default for Lifra {
    fn default() -> Self {
        Self::new()
    }
}