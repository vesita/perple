use std::ops::Index;

use pcd_rs::DynReader;

use crate::config::*;

pub struct Lifra {
    points: [[f32; 3]; POINTS_CAPACITY],
    count: usize,
}

impl Lifra {

    pub fn new() -> Self {
        Lifra {
            points: [[0.0; 3]; POINTS_CAPACITY],
            count: 0,
        }
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

    pub fn reject(&mut self, point: &[f32; 3]) -> bool {
        point[0] == 0.0 && 
        point[1] == 0.0 && 
        point[2] == 0.0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, point: [f32; 3]) {
        if self.count < POINTS_CAPACITY {
            self.points[self.count] = point;
            self.count += 1;
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
}

// 实现IntoIterator，支持所有权转移的迭代
impl IntoIterator for Lifra {
    type Item = [f32; 3];
    type IntoIter = std::array::IntoIter<[f32; 3], POINTS_CAPACITY>;

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