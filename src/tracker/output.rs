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

impl Target {
    pub fn new(the_box: Box3D, class_type: String, id: usize) -> Self {
        Self {
            the_box,
            class_type,
            id,
            velocity: [0.0; 3],
            speed: 0.0,
            is_dynamic: false,
            classification: "unknown".to_string(),
        }
    }

    pub fn default() -> Self {
        Self {
            the_box: Box3D::empty_box(),
            class_type: String::new(),
            id: 0,
            velocity: [0.0; 3],
            speed: 0.0,
            is_dynamic: false,
            classification: "unknown".to_string(),
        }
    }
}
