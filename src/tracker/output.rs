use crate::utils::boxes::Box3D;

#[derive(Clone)]
pub struct Target {
    pub the_box: Box3D,
    pub class_type: String,
    pub id: usize,
    pub velocity: [f32; 3],
    pub speed: f32,
    pub is_dynamic: bool,
    pub classification: String,
}
