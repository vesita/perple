
extern crate nalgebra as na;

use na::{Matrix4, Vector3, Vector4};
use ndarray::{Array3, ArrayD};
use std::collections::HashMap;

pub struct World {
    equips: Vec<Box<dyn Location>>,
    equip_id: HashMap<String, usize>,
    points: Vec<Vector3<f32>>,
}

pub trait Location {
    fn on_world(&self) -> Matrix4<f32>;
    fn set_by_degrees(&mut self, x: f32, y: f32, z: f32);
    fn set_by_arcs(&mut self, x: f32, y: f32, z: f32);
    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>);
}

pub struct Camera {
    position: Matrix4<f32>,
    intrinsicinti: Matrix4<f32>,
    desitc: Matrix4<f32>,
}

pub struct Lidar {
    position: Matrix4<f32>,
}

impl World { 
    pub fn new() -> Self {
        Self {
            equips: vec![],
            equip_id: HashMap::new(),
            points: vec![],
        }
    }
}

impl Location for Camera {
    fn on_world(&self) -> Matrix4<f32> {
        self.position
    }
    fn set_by_degrees(&mut self, x: f32, y: f32, z: f32) {
        self.position = Matrix4::new_rotation(Vector3::new(x, y, z));
    }
    fn set_by_arcs(&mut self, x: f32, y: f32, z: f32) {
        self.position = Matrix4::new_rotation(Vector3::new(x, y, z));
    }
    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>) {
        self.position = *matrix;
    }
}

impl Location for Lidar {
    fn on_world(&self) -> Matrix4<f32> {
        self.position
    }
    fn set_by_degrees(&mut self, x: f32, y: f32, z: f32) {
        self.position = Matrix4::new_rotation(Vector3::new(x, y, z));
    }
    fn set_by_arcs(&mut self, x: f32, y: f32, z: f32) {
        self.position = Matrix4::new_rotation(Vector3::new(x, y, z));
    }
    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>) {
        self.position = *matrix;
    }
}