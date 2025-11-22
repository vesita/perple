use std::ops::Index;

use nalgebra::iter;
use pcd_rs::{DynReader, DynRecord};

use crate::{config::{DBSCAN_MIN_POINTS, POINTS_CAPACITY, RESOLUTION}, lidar::{claster::Claster, lifra::Lifra, tag::Tag3D}};




pub struct Squeeze {
    records: Lifra,
    targets: Claster,
    y_length: usize,
    x_length: usize,
    resolution: f32,
    x_offset: usize,
    y_offset: usize,
}


impl Squeeze { 
    pub fn new(resolution: f32, x_length: usize, y_length: usize) -> Self {
        Squeeze {
            records: Lifra::new(),
            targets: Claster::new(),
            y_length: y_length,
            x_length: x_length,
            resolution: resolution,
            x_offset: POINTS_CAPACITY / 2 - x_length / 2,
            y_offset: POINTS_CAPACITY / 2 - y_length / 2
        }
    }
    
    pub fn claster(&mut self) {
        for point in self.records.iter() {
            self.targets.submit(point);
        }
    }


    /// 获取内部records的不可变引用
    pub fn records(&self) -> &Lifra {
        &self.records
    }
    
    /// 获取内部records的可变引用
    pub fn records_mut(&mut self) -> &mut Lifra {
        &mut self.records
    }
    
    /// 获取聚类结果的不可变引用
    pub fn targets(&self) -> &Claster {
        &self.targets
    }

    pub fn coord2index(&self, x: f32, y: f32) -> Option<usize> {
        let x_index = ((x / self.resolution) as isize + self.x_offset as isize) as usize;
        let y_index = ((y / self.resolution) as isize + self.y_offset as isize) as usize;
        if x_index < self.x_length && y_index < self.y_length {
            Some(y_index * self.x_length + x_index)
        } else {
            None
        }
    }

    pub fn index2coord(&self, index: usize) -> (f32, f32) {
        let x_index = index % self.x_length;
        let y_index = index / self.x_length;
        (x_index as f32 * self.resolution, y_index as f32 * self.resolution)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}



impl IntoIterator for Squeeze {
    type Item = [f32; 3];
    type IntoIter = std::array::IntoIter<Self::Item, POINTS_CAPACITY>;
    fn into_iter(self) -> Self::IntoIter {
        self.records.into_iter()
    }
}

impl<'a> IntoIterator for &'a Squeeze {
    type Item = &'a [f32; 3];
    type IntoIter = std::slice::Iter<'a, [f32; 3]>;
    fn into_iter(self) -> Self::IntoIter {
        self.records.iter()
    }
}

impl<'a> IntoIterator for &'a mut Squeeze {
    type Item = &'a mut [f32; 3];
    type IntoIter = std::slice::IterMut<'a, [f32; 3]>;
    fn into_iter(self) -> Self::IntoIter {
        self.records.iter_mut()
    }
}


pub struct Hist {
    records: [usize; POINTS_CAPACITY],
    count: usize,
    resolution: f32,
    offset: usize,  // 添加偏移量，用于处理负坐标值
    axis: usize,
}

impl Hist {
    pub fn new(resolution: f32, axis: usize) -> Self {
        Hist {
            records: [0; POINTS_CAPACITY],
            count: 0,
            resolution: RESOLUTION,
            offset: POINTS_CAPACITY / 2,  // 设置偏移为中心点
            axis: axis,
        }
    }
    
    pub fn count(&mut self, frame: & Lifra) {
        for point in frame {
            // 获取目标坐标轴的坐标对应的索引
            if let Some(index) = self.coord2index(point[self.axis]) {
                self.records[index] += 1;
                self.count += 1;
            }
        }
    }


    // 添加坐标到索引的转换方法
    pub fn coord2index(&self, coord: f32) -> Option<usize> {
        let index = (coord / self.resolution) as isize + self.offset as isize;
        if index >= 0 && index < POINTS_CAPACITY as isize {
            Some(index as usize)
        } else {
            None  // 坐标超出范围
        }
    }
    
    // 添加索引到坐标的转换方法
    pub fn index2coord(&self, index: usize) -> f32 {
        ((index as isize - self.offset as isize) as f32) * self.resolution
    }
}


impl Index<usize> for Squeeze {
    type Output = [f32; 3];
    fn index(&self, index: usize) -> &Self::Output {
        &self.records[index]
    }
}