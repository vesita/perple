use std::sync::{Arc, Mutex};
use bevy::prelude::*;

use crate::{cloud::Lifra, Swapl};

#[derive(Resource)]
pub struct VisResource {
    pub clouds: Arc<Mutex<Lifra>>,
    pub swapl: Arc<Mutex<Swapl>>,
}